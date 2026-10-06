"""Read frozen private P4 groups and emit aggregate-only distributions; no model calls."""
import json
import os
from collections import Counter
from pathlib import Path
import numpy as np


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return {'n':0,'min':None,'p25':None,'median':None,'p75':None,'p95':None,'max':None,'mean':None}
    if not np.isfinite(a).all():raise ValueError('Nonfinite stored Delta_A')
    quantiles=np.quantile(a,[0,.25,.5,.75,.95,1]).tolist()
    return {'n':len(a),**dict(zip(['min','p25','median','p75','p95','max'],quantiles)),'mean':float(a.mean())}


def summarize(rows):
    variants={}
    for variant in ['correctness_only','legacy_B4','revised_CED']:
        mixed=[r for r in rows if r[variant]['all_legal_mixed']]
        usable=[r for r in rows if r[variant]['usable']]
        reversed_rows=[r for r in rows if r[variant]['ordering_reversed']]
        subsets={'all_groups':rows,'legal_mixed_groups':mixed,'positive_order_nonzero_mixed_groups':usable,'reversed_groups':reversed_rows}
        variants[variant]={
            'groups':len(rows),'patients':len({r['case_id'] for r in rows}),
            'legal_mixed_groups':len(mixed),'positive_order_nonzero_mixed_groups':len(usable),
            'zero_advantage_groups':sum(r[variant]['zero_advantage'] for r in rows),
            'zero_advantage_fraction':sum(r[variant]['zero_advantage'] for r in rows)/len(rows) if rows else None,
            'reversed_groups':len(reversed_rows),'patients_with_reversal':len({r['case_id'] for r in reversed_rows}),
            'reversed_fraction_of_mixed':len(reversed_rows)/len(mixed) if mixed else None,
            'delta_A':{name:distribution([r[variant]['delta_A'] for r in group]) for name,group in subsets.items()},
            'usable_delta_A_lt_1e4':sum(r[variant]['delta_A']<1e-4 for r in usable),
        }
    return variants


def main():
    # One deterministic check of quantile interpolation and undefined empty groups.
    assert distribution([0,1,2,3])['median']==1.5 and distribution([])['median'] is None
    rows=json.loads(Path(os.environ['P4_GROUPS_FILE']).read_text())
    if len(rows)!=256 or set(Counter(r['case_id'] for r in rows).values())!={4}:raise ValueError('Unexpected frozen group coverage')
    result={'model':'historical B1 step_0256','scope':'post-run descriptive aggregation of existing frozen samples; no resampling or training',
            'quantiles':'linear interpolation; these are descriptive group quantiles, not confidence intervals',
            'dependence':'four groups per patient are dependent; groups are not independent clinical cases',
            'strata':{'all':summarize(rows)}}
    for stratum in sorted({r['stratum'] for r in rows}):result['strata'][stratum]=summarize([r for r in rows if r['stratum']==stratum])
    # Reconcile to the run's original frozen aggregate, rather than silently redefining usable groups.
    old=json.loads(Path(os.environ['P4_SUMMARY_FILE']).read_text())['models']['B1']['variants']
    for stratum,variants in result['strata'].items():
        for name,item in variants.items():
            prior=old[name][stratum]
            for new,key in [('groups','groups'),('zero_advantage_groups','zero_advantage_groups'),('positive_order_nonzero_mixed_groups','usable_groups'),('reversed_groups','ordering_reversals')]:
                if item[new]!=prior[key]:raise ValueError('Stored group/aggregate mismatch')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
