"""CPU-only completion audit; publish aggregates, never patient rows or paths."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


def read(path):
    return json.loads(path.read_text())


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n_patients': 0, 'mean_difference': None, 'ci95': None, 'valid_replicates': 0}
    rng = np.random.default_rng(42)
    draws = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    return {'n_patients': len(values), 'mean_difference': float(values.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(), 'valid_replicates': 2000}


def main():
    # One runnable check for the added aggregation logic, including empty cohorts.
    assert mean_ci([])['mean_difference'] is None
    assert mean_ci([0, 0])['ci95'] == [0., 0.]
    assert mean_ci([2])['ci95'] == [2., 2.]
    base = Path(os.environ['PILOT_ROOT'])
    root, public = base/'outputs', base/'code/reports'
    ledger = read(root/'gpu_ledger.json')
    assert ledger['status'] == 'completed' and not ledger['active']
    assert ledger['charged_seconds'] <= ledger['limit_seconds'] == 21600
    records = ledger['records']
    events = [(v['started'], 1) for v in records] + [(v['started']+v['charged_seconds'], -1) for v in records]
    active = maximum = 0
    for _, change in sorted(events):
        active += change
        maximum = max(maximum, active)
    assert maximum <= 3
    training = {}
    initial = {}
    for stage in ('M0', 'W', 'M1', 'M2'):
        p = root/stage
        summary = read(p/'summary.json')
        assert summary['status'] == 'completed' and summary['steps'] == summary['planned_steps']
        lines = [json.loads(line) for line in (p/'train.jsonl').read_text().splitlines()]
        assert len(lines) == summary['steps'] and all(np.isfinite(v['gradient_norm']) for v in lines)
        initial[stage] = read(p/'initial_identity.json')
        summary['initial_identity'] = initial[stage]
        summary['checkpoints'] = {q.name: read(q/'identity.json') for q in sorted(p.glob('step_*'))}
        assert all((p/k/'adapter_model.safetensors').is_file() and (p/k/'training_state.pt').is_file() for k in summary['checkpoints'])
        summary['gradient_norm_pre_clip_quantiles'] = np.quantile([v['gradient_norm'] for v in lines], [0, .5, 1]).tolist()
        summary['mean_EOS_NLL'] = {k: float(np.mean([v[k] for v in lines if k in v])) for k in ('full_eos', 'loc_eos', 'crop_eos') if any(k in v for v in lines)}
        summary['mean_pair_backward_seconds'] = float(np.mean([v.get('pair_backward_s', 0) for v in lines]))
        for values in summary['exposures'].values():
            repetitions = list(values.pop('patient_repetitions').values())
            values['patient_repetition_min_max'] = [min(repetitions), max(repetitions)] if repetitions else None
        training[stage] = summary
    fork = training['W']['checkpoints']['step_0256']
    assert initial['M0'] == initial['W'] and initial['M1'] == initial['M2'] == fork
    assert training['M1']['exposures'] == training['M2']['exposures']
    assert all(v['test_pixels_read'] == 0 for v in training.values())
    evaluation, rows = {}, {}
    for stage in ('M0', 'M1', 'M2'):
        p = root/('eval_'+stage)
        evaluation[stage] = read(p/'summary.json')
        assert evaluation[stage]['status'] == 'completed' and evaluation[stage]['patients'] == 256
        rows[stage] = sorted([json.loads(v) for v in (p/'predictions.jsonl').read_text().splitlines()], key=lambda v:v['case_id'])
        assert len(rows[stage]) == len({v['case_id'] for v in rows[stage]}) == 256
        access = [json.loads(v) for v in (p/'data_access.jsonl').read_text().splitlines()]
        assert all(v['split'] == 'validation' for v in access)
        assert evaluation[stage]['test_pixels_read'] == 0
    assert [v['case_id'] for v in rows['M1']] == [v['case_id'] for v in rows['M2']]
    supplement = {}
    for label in ('positive', 'negative'):
        for scale in ('2.0', '2.5'):
            key = 'negative_crop_p_'+scale
            paired = [(a, b) for a, b in zip(rows['M1'], rows['M2']) if ('crops' in a if label=='positive' else key in a)]
            assert all(('crops' in b if label=='positive' else key in b) for a,b in paired)
            def score(row):
                return float(np.mean([v[scale] for v in row['crops']])) if label=='positive' else row[key]
            supplement[f'{label} / crop_{scale} / M2 minus M1'] = mean_ci([score(b)-score(a) for a,b in paired])
        for kind in ('gray', 'blur', 'blank'):
            paired = [(a,b) for a,b in zip(rows['M1'],rows['M2']) if 'views' in a and a['views']['positive']==(label=='positive')]
            for key in ('S_positive_rate', 'u_original') + (('view_false_positive_rate',) if label=='negative' else ()):
                def score(row):
                    if key=='u_original': return row['p']
                    view=row['views'][kind]
                    if key=='S_positive_rate': return float(view['S']>0)
                    return float(np.mean([view[v]>=.5 for v in ('keep','hide')]))
                supplement[f'{label} / {kind} / {key} / M2 minus M1'] = mean_ci([score(b)-score(a) for a,b in paired])
    finish = max(v['started']+v['charged_seconds'] for v in records)
    output = {'status': 'completed', 'training': training, 'evaluation': evaluation,
              'actual_W_fork_identity_equal': True, 'actual_branch_exposures_equal': True,
              'charged_gpu_hours': ledger['gpu_hours'], 'charged_gpu_seconds': ledger['charged_seconds'],
              'max_concurrent_gpu_jobs': maximum, 'gpu_time_by_stage_seconds': {v['stage']:v['charged_seconds'] for v in records},
              'finished_GPU_UTC': datetime.fromtimestamp(finish, timezone.utc).isoformat(),
              'supplementary_paired_intervals': supplement,
              'aggregation_provenance': 'CPU-only completion supplement added after results; frozen training/evaluation/source lock unchanged; no new inference, view, threshold, sample or checkpoint selection',
              'statistical_unit': 'Patient; crop boxes and sham views averaged within patient; 2000 paired resamples seed42; no training-seed uncertainty'}
    (public/'medevidence_p1_training.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = ['# MedEvidence pilot-1 实际训练与完成审计', '',
             f"全部预定训练与统一评价 completed，GPU阶段结束UTC {output['finished_GPU_UTC']}。累计保守计费 {ledger['gpu_hours']:.6f} GPU小时，包含加载、smoke及两次0更新preflight失败；上限6小时，实际最多{maximum}个独立单卡作业。", '',
             '| 阶段 | 实际/计划更新 | 累计更新 | full | L | crop | control | dep诊断 | hinge活跃率 | 峰值GiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in training.items():
        loss=s['mean_losses'];hinge='NA' if s['hinge_active_rate'] is None else f"{s['hinge_active_rate']:.4f}"
        lines.append(f"| {name} | {s['steps']}/{s['planned_steps']} | {s['cumulative_step']} | {loss['full_loss']:.6f} | {loss['loc_loss']:.6f} | {loss['crop_loss']:.6f} | {loss['control_loss']:.6f} | {loss['dep_loss']:.6f} | {hinge} | {s['peak_memory_gib']:.3f} |")
    lines += ['', 'M1/M2表内loss为256步续训均值，W只训练一次；dep均值按全部更新计算，hinge分母各128个适用阳性更新。M1 dep仅诊断、系数0；M2系数.2。目标full + .5 L + .5 crop + .1 control，原图4槽平均，辅助不再除4。', '',
              '实际W step256分叉检查PASS：adapter、optimizer、RNG、lr、排程位置完全相等；M0/W初始状态完全相等。M1/M2暴露逐项相等：续训各1024原图、256 L、256 crop和256 pair，pair128阳性/128阴性，来自3阳性和32阴性患者。3个阳性各重复42或43次，不能当128个独立患者。', '',
              '全部实际梯度有限；训练梯度范数、EOS单项NLL、视图/视觉forward计数、患者重复范围及身份摘要见相邻JSON。训练阶段记录总梯度范数；dep独立梯度只在smoke审计，正式阶段未逐步额外分解。视觉冻结与dep连接验证见engineering报告。', '',
              '预定checkpoints均存在adapter和optimizer/RNG状态：M0局部128/256/512，W128/256，M1/M2局部128/256（累计384/512）。评价固定最后checkpoint，无最佳点选择。全部评估各256患者；测试像素访问0，实际访问日志均validation。', '',
              '| GPU计费阶段 | 秒 |', '|---|---:|']
    lines += [f"| {v['stage']} | {v['charged_seconds']:.3f} |" for v in records]
    lines += ['', '| 补充患者配对比较 | 患者数 | M2−M1 | 95%区间 |', '|---|---:|---:|---|']
    for key,value in supplement.items():
        lines.append(f"| {key} | {value['n_patients']} | {value['mean_difference']} | {value['ci95']} |")
    lines += ['', '以上补充统计只读取已冻结评价输出，无新增GPU计算。多框crop和阴性两个遮挡先在患者内平均，再配对重抽样；有效重复数各2000，空队列NA。此完成聚合脚本在结果后加入；原冻结训练、评价与主分析脚本保持原版本。患者图像、标识、逐例排程/坐标/审核、预测、checkpoint、私有路径及原始日志不公开。']
    (public/'medevidence_p1_training.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'completed_audit', 'GPU_hours':ledger['gpu_hours'], 'fork_equal':True, 'patients_per_model':256}))


if __name__ == '__main__':
    main()
