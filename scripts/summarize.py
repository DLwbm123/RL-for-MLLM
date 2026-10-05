"""Render honest aggregate reports from existing outputs; never invent missing results."""
import json
import os
from pathlib import Path


def main():
    root=Path(os.environ['OUTPUT_ROOT']);out=root/'reports';out.mkdir(exist_ok=True)
    status=json.loads((root/'pipeline_status.json').read_text()) if (root/'pipeline_status.json').exists() else {'status':'not_started'}
    lines=['# Pilot 实际执行报告','',f"状态：{status['status']}。只统计已写出的真实结果；空白表示尚未完成。",'',
           '| 方法 | 状态 | 更新数 | 验证 AUROC | GPU 小时 | 峰值显存 GiB |','|---|---|---:|---:|---:|---:|']
    for method in ['B0','B1','B2','B3','B4','B5','B6']:
        file=root/method/'summary.json'
        if file.exists():
            d=json.loads(file.read_text());auroc=(d.get('selected') or {}).get('auroc','')
            lines.append(f"| {method} | 完成短验证 | {d.get('steps','')} | {auroc} | {d.get('gpu_hours','')} | {d.get('peak_memory_gib','')} |")
        elif method=='B0' and (root/'audits/B0/summary.json').exists():
            d=json.loads((root/'audits/B0/summary.json').read_text())
            lines.append(f"| B0 | 120 Case 审计子集 | 0 | {d['classification']['auroc']} | {d['gpu_hours']} | {d['peak_memory_gib']} |")
        else:lines.append(f'| {method} | 尚未完成/尚未运行，见状态记录 | | | | |')
    lines+=['','B0 审计子集与训练 checkpoint 的完整 validation 指标分母不同，不直接作性能差值。',
            '数据、图像、逐样本预测、权重和原始日志仅保存在执行数据盘。测试集未开启。',
            '工程测试和短暂拟合不能证明医学有效性，不能证明 RL 必要。', '', '```json',json.dumps(status,indent=2,ensure_ascii=False),'```']
    (out/'pilot_report.md').write_text('\n'.join(lines)+'\n')
    audit=['# 冻结证据审计','', '只报告 validation 中预先分层选定的 Case，每 Case 固定一张图；不按是否答对选择。','']
    for tag in ['B0','B1']:
        file=root/'audits'/tag/'summary.json'
        audit.append('## '+tag)
        if file.exists():audit+=['','```json',json.dumps(json.loads(file.read_text()),ensure_ascii=False,indent=2),'```','']
        else:audit+=['','尚未完成，无可报告的模型证据结果。','']
    (out/'frozen_audit.md').write_text('\n'.join(audit)+'\n')


if __name__=='__main__':main()
