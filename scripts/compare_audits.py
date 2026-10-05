"""Paired B0/B1 changes and the fixed intersection of correctly classified cases."""
import json
import os
from pathlib import Path
import pandas as pd
from src.data import write_json
from src.evaluation import paired_auroc_difference,cluster_bootstrap


def main():
    root=Path(os.environ['OUTPUT_ROOT']);tags=os.environ.get('AUDIT_TAGS','B0,B1').split(',')
    records={tag:[json.loads(x) for x in (root/'audits'/tag/'predictions.jsonl').read_text().splitlines()] for tag in tags}
    maps={tag:{r['image_id']:r for r in rows} for tag,rows in records.items()}
    ids=set.intersection(*(set(m) for m in maps.values()))
    if any(len(m)!=len(ids) for m in maps.values()):raise ValueError('Audits must have identical fixed selections')
    correct={k for k in ids if all(m[k]['correct'] for m in maps.values())}
    eligible={k for k in ids if all(m[k]['eligible'] for m in maps.values())}
    report={'tags':tags,'n_all':len(ids),'n_eligible':len(eligible),'n_common_correct':len(correct),
            'n_common_correct_eligible':len(correct&eligible),'methods':{}}
    for tag,m in maps.items():
        report['methods'][tag]={}
        for name,subset in [('all_eligible',eligible),('common_correct_eligible',eligible&correct)]:
            report['methods'][tag][name]={'n':len(subset)}
            for kind in ['feature','pixel']:
                rows=pd.DataFrame([{'case_id':m[k]['case_id'],'M':m[k][kind]['M']} for k in sorted(subset)])
                report['methods'][tag][name][kind]={'mean_M':float(rows.M.mean()),**cluster_bootstrap(rows,lambda x:x.M.mean())} if len(rows) else None
    if len(tags)==2:
        first,second=[pd.DataFrame(records[tag]) for tag in tags]
        report['paired_auroc_difference']=paired_auroc_difference(first,second)
        for kind in ['feature','pixel']:
            rows=pd.DataFrame([{'case_id':maps[tags[0]][k]['case_id'],'difference':maps[tags[1]][k][kind]['M']-maps[tags[0]][k][kind]['M']} for k in sorted(eligible)])
            report[kind+'_margin_difference']={'mean':float(rows.difference.mean()),**cluster_bootstrap(rows,lambda x:x.difference.mean())} if len(rows) else None
    write_json(root/'audits/comparison.json',report)


if __name__=='__main__':main()
