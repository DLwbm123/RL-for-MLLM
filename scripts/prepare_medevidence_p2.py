"""Replay existing private outputs, audit coverage and freeze localization probe."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import ImageDraw, Image
from src.data import smart_size
from src.medevidence import normalized, parse_boxes, matching, exposures
from src.medevidence_p2 import select_sets, loc_schedule, enrich, summarize, token_contract, distribution
from src.v2 import save, DevelopmentData


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def exposure_audit(plan, rows, tokenizer, method):
    train={k:r for k,r in rows.items() if r['split']=='train'}
    counts=Counter(v['loc'] for v in plan)
    result={}
    for label in ('yes','no'):
        pool={k:r for k,r in train.items() if r['pathology']==label}
        repetitions=[counts[k] for k in pool]
        lengths=[len(tokenizer.encode(r['loc_target'],add_special_tokens=False))+1 for r in pool.values()]
        box_repetitions=[counts[k] for k,r in pool.items() for _ in r['boxes']]
        result[label]={'eligible_patients':len(pool),'exposures':sum(repetitions),'different_patients':sum(n>0 for n in repetitions),
                       'patient_repetition_histogram':dict(Counter(repetitions)), 'repetitions_0_1_2_more':{str(i):sum(v==i for v in repetitions) for i in (0,1,2)} | {'more_than_2':sum(v>2 for v in repetitions)},
                       'original_frame_repetition_histogram':dict(Counter(box_repetitions)),
                       'GT_target_length_including_EOS':distribution(lengths),
                       'exposure_weighted_target_length':distribution([len(tokenizer.encode(rows[v['loc']]['loc_target'],add_special_tokens=False))+1 for v in plan if rows[v['loc']]['pathology']==label])}
    result['single_box_exposures']=sum(len(rows[v['loc']]['boxes'])==1 for v in plan)
    result['multi_box_exposures']=sum(len(rows[v['loc']]['boxes'])>1 for v in plan)
    result['loss_weight']=.5
    result['normalization']='each auxiliary patient mean-token NLL incl EOS, multiplied .5; not divided by four classification slots'
    return result


def coverage(rows, cfg, draft, review, audit):
    rule=cfg['dep_applicability'];result={};geometry=Counter()
    proposals=draft['positive_proposals'];cases=review['cases']
    for split in ('train','validation'):
        positives={k:r for k,r in rows.items() if r['split']==split and r['boxes']}
        single={k:r for k,r in positives.items() if len(r['boxes'])==1}
        eligible={}
        for k,r in single.items():
            b=r['boxes'][0];area=(b[2]-b[0])*(b[3]-b[1])/(r['width']*r['height'])
            if rule['min_target_area_fraction']<=area<=rule['max_target_area_fraction'] and min(b[2]-b[0],b[3]-b[1])>=rule['min_target_side_pixels']:eligible[k]=r
        geometry['multiple_boxes']+=len(positives)-len(single);geometry['nonlocal_or_too_small']+=len(single)-len(eligible)
        proposed={k:v for k,v in proposals.items() if v['split']==split}
        reviewed={k:cases[k] for k in proposed if k in cases}
        accepted={k:v for k,v in reviewed.items() if v['decision']=='accept'}
        reasons=Counter(v['reason'] for v in reviewed.values() if v['decision']!='accept')
        mapped=Counter()
        for v in reviewed.values():
            if v['decision']=='accept':continue
            if v['reason']=='other_or_nonlocal_visible_information':mapped['multifocal_diffuse_or_target_information_outside_box']+=1
            elif v['reason']=='uncertain_target_or_control':mapped['unknown_target_vs_control_uncertainty']+=1
            else:mapped['unknown_target_vs_control_vs_anatomical_context']+=1
        result[split]={'original_positive_patients':len(positives),'original_single_box':len(single),'original_multi_box':len(positives)-len(single),
                       'area_side_eligible':len(eligible),'area_side_excluded':len(single)-len(eligible),
                       'geometry_attempted_source_loop':len(eligible),'geometry_failed_recorded_all_splits':audit['geometry_exclusions'].get('no_matched_control',0),
                       'geometry_success_inferred_from_source_and_aggregate':len(eligible) if audit['geometry_exclusions'].get('no_matched_control',0)==0 else None,
                       'unattempted_geometry':0,'actual_proposals':len(proposed),'eligible_not_proposed_or_reviewed':len(eligible)-len(proposed),
                       'review_completed':len(reviewed),'proposed_unreviewed':len(proposed)-len(reviewed),'accepted':len(accepted),
                       'reject_or_unknown_combined':sum(v['decision']=='reject_or_unknown' for v in reviewed.values()),
                       'pure_rejected_vs_uncertain':'NA: prior records combine reject_or_unknown; cannot separate retrospectively',
                       'original_primary_reasons':dict(reasons),'supported_coarse_reason_mapping':dict(mapped),
                       'semantic_review_missing_among_geometry_eligible':len(eligible)-len(reviewed),
                       'crop_excluded_reviewed':sum(not v.get('crop_usable',True) for v in reviewed.values()),
                       'crop_eligible_positive_patients':sum(k in draft['crops'] and (k not in cases or cases[k].get('crop_usable',True)) for k in positives),
                       'use_levels':{'original_region_supervision':len(positives),
                           'region_finding_correspondence':'All annotated positives are candidates; semantic crop applicability not exhaustively validated',
                           'paired_restricted_evidence_accepted':len(accepted)},
                       'overlapping_multilabel_reasons':'Not recorded; only one primary reason per old reviewed patient; unrecorded causes remain unknown'}
    assert dict(geometry)=={k:audit['geometry_exclusions'][k] for k in geometry}
    return result


def main():
    base=Path(os.environ['PILOT_ROOT']);code=base/'code';root=base/'outputs';old=Path(os.environ['REFERENCE_OUTPUT_ROOT'])
    p=root/'protocol';p.mkdir(exist_ok=True);public=code/'reports';public.mkdir(exist_ok=True)
    if (p/'lock.json').exists() or (p/'CPU_checks.json').exists():raise FileExistsError('Already prepared; no repeated selection or overwrite')
    cfgpath=code/'configs/medevidence_p2.json';cfg=read(cfgpath);oldcfg=read(code/'configs/medevidence_p1.json')
    assert cfg['prompts']=={k:oldcfg['prompts'][k] for k in ('A','L')} and cfg['revision']==oldcfg['revision']
    lock=read(old/'protocol/lock.json')
    assert digest(code/'configs/medevidence_p1.json')==lock['config_sha256']
    for name,h in lock['hashes'].items():assert digest(old/'protocol'/name)==h,name
    for name,h in lock['source_hashes'].items():assert digest(code/name)==h,name
    frame=pd.read_csv(old/'protocol/manifest.csv',keep_default_na=False,dtype={'case_id':str,'image_id':str})
    if frame.case_id.duplicated().any() or set(frame.split)!={'train','validation'}:raise ValueError('Patient/split identity error')
    rows=frame.set_index('image_id').to_dict('index')
    for r in rows.values():r['boxes']=json.loads(r['boxes'])
    checkpoints={}
    from safetensors.torch import load_file
    from src.medevidence_run import tensor_digest
    for stage in ('W','M1','M2','M0'):
        summary=read(old/stage/'summary.json')
        assert summary['status']=='completed' and summary['steps']==summary['planned_steps']
        path=old/stage/('step_0512' if stage=='M0' else 'step_0256')
        if not (path/'adapter_model.safetensors').exists():
            checkpoints[stage]={'status':'blocked_checkpoint'}
            continue
        identity=read(path/'identity.json')
        weights=load_file(str(path/'adapter_model.safetensors'))
        weights={k.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight'):v for k,v in weights.items()}
        assert tensor_digest(weights)==identity['adapter_digest'],stage+' CPU adapter identity'
        checkpoints[stage]={'identity':identity,'checkpoint_path':str(path),'adapter_bytes':(path/'adapter_model.safetensors').stat().st_size,
                            'training_state_bytes':(path/'training_state.pt').stat().st_size}
    del weights
    save(p/'reference_checkpoints.json',checkpoints)
    from transformers import AutoProcessor
    model_base=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    identity=read(old/'protocol/identity.json')
    for name,v in identity['base']['files'].items():
        st=(model_base/name).stat();assert (st.st_size,st.st_mtime_ns)==(v['bytes'],v['mtime_ns'])
    processor=AutoProcessor.from_pretrained(model_base,local_files_only=True,use_fast=False,
        min_pixels=cfg['min_visual_tokens']*784,max_pixels=cfg['max_visual_tokens']*784)
    tokenizer=processor.tokenizer
    assert tokenizer.eos_token_id==identity['eos_id']
    assert read(old/'protocol/token_contract.json')['max_new_tokens']==cfg['max_new_tokens']==80
    replay={};private_replay={};oldfolds=read(old/'protocol/folds.json')
    oldresults=read(old.parent/'code/reports/medevidence_p1_results.json')
    for stage in ('M1','M2'):
        summary=read(old/('eval_'+stage)/'summary.json');assert summary['patients']==256 and summary['status']=='completed'
        records=[json.loads(line) for line in (old/('eval_'+stage)/'predictions.jsonl').read_text().splitlines()]
        assert len(records)==len({r['case_id'] for r in records})==256
        assert all(rows[r['image_id']]['split']=='validation' for r in records)
        enriched=[enrich(r,rows[r['image_id']]) for r in records];private_replay[stage]=enriched
        replay[stage]=summarize(enriched)
        thresholds=oldresults['models'][stage]['classification']['fold_thresholds']
        replay[stage]['historical_crossfit_supplement']=summarize([{**r,'class_prediction':r['p']>=thresholds[oldfolds[r['image_id']]]} for r in enriched])
        replay[stage]['historical_crossfit_supplement']['rule']='Reused stored pilot-1 fold thresholds and fold membership; no new fitting or threshold search'
    save(root/'D0_patient_replay.json',private_replay)
    plan=read(old/'protocol/schedule.json');views=read(old/'protocol/views.json');training_exposure={}
    for stage in ('W','M1','M2'):
        selected_plan=plan[:256] if stage=='W' else plan[256:]
        logged=[json.loads(v) for v in (old/stage/'train.jsonl').read_text().splitlines()]
        assert len(logged)==256 and [v['cumulative_step'] for v in logged]==[v['step'] for v in selected_plan]
        assert exposures(selected_plan,rows,views,stage)==read(old/stage/'summary.json')['exposures']
        training_exposure[stage]=exposure_audit(selected_plan,rows,tokenizer,stage)
    paired=Counter(k for v in plan[256:] if v['pair_positive'] for k in [v['pair']])
    assert len(paired)==3 and sum(paired.values())==128
    replay['training_exposure']=training_exposure
    save(public/'medevidence_p2_empty_output_audit.json',replay)
    coverage_audit=coverage(rows,oldcfg,read(old/'protocol/views_draft.json'),read(old/'protocol/AI_review.json'),read(old/'protocol/data_audit.json'))
    save(public/'medevidence_p2_coverage_audit.json',coverage_audit)
    sets=select_sets(rows,cfg['seed']);schedule=loc_schedule(sets['Fit32'],rows,cfg['steps'])
    assert Counter(k for update in schedule for k in update)==Counter({k:32 for k in sets['Fit32']})
    save(p/'sets.json',sets);save(p/'schedule.json',schedule);frame.to_csv(p/'manifest.csv',index=False)
    save(p/'reference_identity.json',identity)
    contracts={k:token_contract(tokenizer,rows[k]['loc_target']) for k in sets['Fit32']+sets['Ref32']}
    save(p/'token_contracts.json',contracts)
    scoring_errors=[];maximum_error=0.
    for k,r in rows.items():
        gt=normalized(r['boxes'],r['width'],r['height']);parsed,error=parse_boxes(r['loc_target'])
        if error or parsed!=gt or not matching(gt,parsed)['strict']:scoring_errors.append(k)
        for a,b in zip(sorted(r['boxes']),gt):
            error=np.abs(np.asarray(a)-np.asarray(b)*np.asarray([r['width'],r['height'],r['width'],r['height']])/1000)
            assert np.all(error<=np.asarray([r['width'],r['height'],r['width'],r['height']])/2000+1e-9)
            maximum_error=max(maximum_error,float(error.max()))
    assert not scoring_errors
    data=DevelopmentData(os.environ['DATA_ROOT'],frame,root/'CPU_data_access.jsonl');data.install_guard()
    try:(Path(os.environ['DATA_ROOT'])/'not_authorized_test_pixel.png').read_bytes()
    except PermissionError:pass
    else:raise AssertionError('Test access not blocked')
    images={};alignment_checks=[];overlay=root/'alignment';overlay.mkdir(exist_ok=True)
    show=[k for k in sets['Fit32'] if rows[k]['pathology']=='yes'][:4]
    canvas=Image.new('RGB',(1024,512*len(show)))
    for k in sets['Fit32']+sets['Ref32']:
        r=rows[k];im=data.image(k);assert im.size==(r['width'],r['height'])
        source=Path(os.environ['DATA_ROOT'])/r['image_path'];st=source.stat()
        images[k]={'image_sha256':digest(source),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'size':list(im.size),'sop':r['sop']}
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':cfg['prompts']['L']}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch=processor(text=[text],images=[im],return_tensors='pt');grid=batch['image_grid_thw'][0].tolist()
        assert grid[0]==1 and (grid[1]*14,grid[2]*14)==smart_size(im.height,im.width,cfg['min_visual_tokens'],cfg['max_visual_tokens'])
        image_token_id=read(model_base/'config.json')['image_token_id']
        assert int((batch['input_ids']==image_token_id).sum())==grid[1]*grid[2]//4
        alignment_checks.append({'image_id':k,'image_grid_thw':grid,'original_size':list(im.size),'orientation':'stored original pixels, no flip/transpose/overlay transform'})
        if k in show:
            j=show.index(k);marked=im.copy();d=ImageDraw.Draw(marked)
            for box in r['boxes']:d.rectangle(box,outline='red',width=5)
            canvas.paste(im.resize((512,512)),(0,j*512));canvas.paste(marked.resize((512,512)),(512,j*512))
    canvas.save(overlay/'GT_alignment.jpg',quality=95)
    save(p/'selected_images.json',images);save(p/'alignment_checks.json',alignment_checks)
    save(p/'CPU_checks.json',{'status':'passed','historical_replay_parser_matcher_equal':True,'adapter_identities_exact_CPU':True,
        'Fit32_Ref32_disjoint':True,'patients':64,'max_coordinate_roundtrip_error_pixels':maximum_error,'GT_self_match_patients':1280,
        'processor_grid_checks':64,'test_access_blocked':True,'test_pixels_read':0,'target_tokenization_verified':64,'overlay_engineering_review':'pending'})
    selected_summary={s:{'n':len(v),'positive':sum(bool(rows[k]['boxes']) for k in v),'negative':sum(not rows[k]['boxes'] for k in v),
                        'target_length_including_EOS':{label:distribution([len(contracts[k]['target_ids']) for k in v if rows[k]['pathology']==label]) for label in ('yes','no')}} for s,v in sets.items()}
    freeze={'historical_reference_commit':cfg['reference_commit'],'historical_source_lock_verified':True,
            'checkpoint_adapter_identities':{k:v['identity'] if 'identity' in v else {'status':v['status']} for k,v in checkpoints.items()},'selected_summary':selected_summary,
            'selection_independent_of_scores_AI_and_reward':True,'actual_L_only_schedule':{'steps':256,'exposures':1024,'positive':512,'negative':512,'per_patient':32},
            'CPU_checks':read(p/'CPU_checks.json')}
    save(code/'protocol/medevidence_p2/freeze_summary.json',freeze)
    lines=['# MedEvidence pilot-2 历史逐例重放','','D0仅CPU读取既有M1/M2逐例输出，严格解析/匹配与历史记录一致；没有新模型生成。原始0.5阈值为主，invalid保留。','',
           '| 模型 | 阳性空列表 | 分类yes且空列表（阳性） | 阳性区域召回 | 单框严格成功 | 阴性非空 | 分类—定位不一致 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in ('M1','M2'):
        r=replay[name];cross=r['classification_loc_cross_table'];yesempty=next(v['patients'] for v in cross if v['true'] and v['class_yes'] and v['loc_state']=='valid_empty')
        lines.append(f"| {name} | {r['positive']['valid_empty']}/37 | {yesempty} | {sum(v['GT_regions']-v['missed_GT_regions'] for v in r['positive_failure_categories'].values())}/51 | {r['single_strict_success_count']}/25 | {r['negative']['valid_nonempty']}/219 | {r['class_loc_inconsistent_valid_count']}/256 |")
    lines+=['','总体联合正确为补充：M1 164/256、M2 165/256；始终no+[]参照为219/256（85.55%），由相同37阳性/219阴性分母得到。多数类参照强调总体指标的局限，不表示该预测器有临床能力。','',
            '精确分类×定位三状态交叉表、五种漏检类别的患者和区域分母、框数量/尺寸/中心/模板统计以及W和续训定位暴露直方图见相邻JSON。近邻诊断每个预测选最高IoU GT，tie按中心距离，不计正式检出；正式检出仍原一对一IoU≥.5。','',
            'W及M1/M2每256步各128阳性、128阴性L监督；每患者、每原始框重复分布包含未出现者。L实际权重.5，按患者完整目标含EOS的平均token NLL，不额外除原图4槽。排程比例、短序列权重与任务冲突均是待检验假设，不能直接据此认定空列表失败原因。']
    (public/'medevidence_p2_empty_output_audit.md').write_text('\n'.join(lines)+'\n')
    lines=['# MedEvidence pilot-2 证据覆盖审计','','E0只读取元数据、实际源码流程与旧审核记录，未重新采样对照、未重审。','',
           '| 阶段 | 训练 | 开发 |','|---|---:|---:|']
    for key in ('original_positive_patients','original_single_box','original_multi_box','area_side_eligible','geometry_attempted_source_loop','geometry_success_inferred_from_source_and_aggregate','actual_proposals','review_completed','accepted','eligible_not_proposed_or_reviewed'):
        lines.append(f"| {key} | {coverage_audit['train'][key]} | {coverage_audit['validation'][key]} |")
    lines+=['','实际旧源码先为每个面积/边长合格病例调用control_box，再检查训练提议上限32。因此未进入提议的病例不是完全未尝试几何：它们做过几何搜索但结果没有逐例保存。旧全量几何失败记录为0，可以据执行源码和聚合收据推断所有满足面积/边长者通过几何；对未提议者不能声称完成语义审核。','',
            '主要缩减来自单框/局部面积的任务定义、训练首32提议范围以及已审核病例中的语义适用性。不同阶段和unknown原因见JSON；旧reject_or_unknown无法再拆成纯拒绝与不确定。37个总计框外/非局部信息原因有记录，其余目标/control/解剖上下文之间不能精确归因；多标签重叠未记录，不把推测补成审核事实。','',
            'crop资格与dep资格不同：dep未通过不删除原图L监督。原始区域监督、区域—发现对应、成对证据受限比较分别保留各自适用边界，不共用总gate。','',
            '后续证据任务的首选建议：改为较少依赖“框内目标信息完全消失”的区域—发现对应任务。依据是当前最明显排除原因为框外/非局部信息，增加相同删除式候选不能消除定义限制；需要验证区域与发现的对应标注和可观察上下文。该任务只能证明局部对应，不能证明删除目标后的因果答案变化。本轮不执行新证据训练或数据扩展；定位探针后的总研发优先级另见最终decision。']
    (public/'medevidence_p2_coverage_audit.md').write_text('\n'.join(lines)+'\n')
    save(root/'stage_status.json',{'D0':{'status':'completed','models':2,'patients_per_model':256},'E0':{'status':'completed','new_reviews':0},
          'D1':{'status':'not_authorized'},'L1_loc_probe':{'status':'not_authorized'},'CPU':{'status':'ready','patients':64,'overlay_review':'pending'}})
    print(json.dumps({'D0':'completed','E0':'completed','fixed_sets':selected_summary,'CPU_checks':'passed','overlay_review':'pending','GPU_launched':False}))


if __name__=='__main__':main()
