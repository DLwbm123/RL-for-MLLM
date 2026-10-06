"""Aggregate saved v3 decompositions and residuals; no new scoring or sampling."""
import json
import os
from pathlib import Path
import numpy as np
from src.v3 import distribution,errors


def main():
    root=Path(os.environ['OUTPUT_ROOT'])/'reward_replay'
    groups=json.loads((root/'group_advantage_replay.json').read_text())
    cases=json.loads((root/'case_decomposition.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text())
    if summary['status']!='completed':raise ValueError('Completed bounded replay required')
    reversed_rows=[g for g in groups if g['stratum']=='geometry_only_positive' and g['versions']['R-H']['mixed'] and g['versions']['R-H']['sign_b']<0]
    by_k={}
    for k in range(1,8):
        rows=[r['versions']['R-H'] for r in reversed_rows if r['versions']['R-H']['correct_candidates']==k]
        by_k[str(k)]={'groups':len(rows),'patients':len({g['case_id'] for g in reversed_rows if g['versions']['R-H']['correct_candidates']==k}),
                      'float32_Delta_A':distribution([r['normalizations']['historical_float32']['Delta_A'] for r in rows]),
                      'float32_formula_error':distribution([r['normalizations']['historical_float32']['reversal_formula_error'] for r in rows]),
                      'float64_no_epsilon_formula_error':distribution([r['normalizations']['float64_without_epsilon']['reversal_formula_error'] for r in rows])}
    all_cached=[];reward_errors=[]
    for case in cases:
        for pack in case['labels'].values():
            cached=pack['cached']
            all_cached.extend([cached[k] for k in ['reward','D_E','raw_M','m','std','gate']]+cached['D_N'])
            reward_errors.append(errors([pack['float32']['final_reward']],[cached['reward']],cfg['relative_error_denominator_floor']))
    anomalies=[]
    for group in groups:
        for name,result in group['versions'].items():
            n=result['normalizations']['historical_float32']
            if n['constant_input_roundoff_advantage']:
                anomalies.append({'variant':name,'correct_candidates':result['correct_candidates'],'stored_input_constant':True,
                                  'float32_std':n['reward_std'],'advantage':distribution(n['reward_advantage']),
                                  'float64_with_epsilon_zero':result['normalizations']['float64_with_epsilon']['zero_advantage'],
                                  'float64_without_epsilon_zero':result['normalizations']['float64_without_epsilon']['zero_advantage']})
    result={'analysis_type':'aggregation of already completed v3 CPU replay; no additional measurements',
            'reversed_groups_by_correct_count':by_k,
            'recomputed_reward_error':{'max_absolute':max(x['max_absolute'] for x in reward_errors),'max_relative_with_floor':max(x['max_relative_with_floor'] for x in reward_errors),'relative_floor':cfg['relative_error_denominator_floor']},
            'cached_numeric_serialization':{'numbers_checked':len(all_cached),'non_float32_representable_numbers':sum(float(np.float32(x))!=x for x in all_cached),'interpretation':'checks stored cache representation only; precision before caching remains NA'},
            'constant_input_numerical_artifacts':anomalies,
            'ordering_protection_gap':{v:distribution([c['version_gaps'][v] for c in cases]) for v in cfg['reward_versions']},
            'same_patient_mixed_order_changes':sum(len({g['versions']['R-H']['order'] for g in groups if g['case_id']==c['case_id'] and g['versions']['R-H']['mixed']})>1 for c in cases),
            'float32_vs_float64_advantage_max_error':{v:max(g['versions'][v]['float32_vs_float64_advantage_error']['max_absolute'] for g in groups) for v in cfg['reward_versions']}}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
