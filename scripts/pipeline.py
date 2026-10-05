"""Bounded, detached pilot. Every stage stops on error; no formal run or monitoring daemon."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from src.data import write_json


def main():
    root=Path(os.environ['OUTPUT_ROOT']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pilot.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads(Path(os.environ['RUN_CONFIG']).read_text());started=time.time()
    history=[]
    stages=[('audit_B0',{'ACTION':'audit','TAG':'B0'}),('shortfit',{'ACTION':'train','METHOD':'shortfit'}),
            ('SFT',{'ACTION':'train','METHOD':'B1'}),('audit_B1',{'ACTION':'audit','TAG':'B1'})]
    def run(name,changes):
        # Fixed, previously selected GPU; stop if the resource margin has disappeared.
        gpu=os.environ['CUDA_VISIBLE_DEVICES']
        free=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<30000:raise RuntimeError('Insufficient GPU memory margin before '+name)
        status={'status':'running','stage':name,'started':started,'completed_stages':history}
        write_json(root/'pipeline_status.json',status)
        env=os.environ.copy();env.update(changes)
        script="from src.experiment import stage_main; stage_main()"
        logs=root/'logs';logs.mkdir(exist_ok=True)
        with (logs/(name+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'-u','-'],input=script,text=True,stdout=log,stderr=subprocess.STDOUT,env=env)
        history.append({'stage':name,'exit_code':result.returncode,'finished':time.time()})
        if result.returncode:raise RuntimeError(name+' failed; see stage log')
    try:
        smoke=json.loads((root/'smoke/summary.json').read_text())
        if smoke['status']!='passed':raise RuntimeError('Model smoke did not pass')
        for name,changes in stages:
            run(name,changes)
            if name=='shortfit' and not json.loads((root/'shortfit/summary.json').read_text())['fit_gate_passed']:
                raise RuntimeError('32-case short-fit loss did not decrease by the predeclared 10%; inspect before common SFT')
        from scripts.compare_audits import main as compare
        compare()
        summary=json.loads((root/'audits/B1/summary.json').read_text())
        review=json.loads((root/'gates/geometry_review.json').read_text())
        reasons=[]
        if review.get('status')!='passed':reasons.append('geometry_review_not_passed')
        # Conservative pilot screening criteria, frozen before any model audit.
        criteria=cfg['gate_criteria']
        if summary['n_eligible']<criteria['min_eligible_cases']:reasons.append('insufficient_eligible_validation_cases')
        for kind in ['feature','pixel']:
            s=summary[kind]
            if not s.get('M',{}).get('ci95') or s['M']['ci95'][0]<=0:reasons.append(kind+'_margin_interval_not_positive')
            if not s.get('evidence_minus_wrong',{}).get('ci95') or s['evidence_minus_wrong']['ci95'][0]<=0:reasons.append(kind+'_wrong_region_separation_not_positive')
            for field in ['M_area_spearman','M_tokens_spearman']:
                if s.get(field) is None or abs(s[field])>=criteria['max_abs_area_or_token_spearman']:reasons.append(kind+'_'+field+'_needs_review')
        gate={'status':'failed' if reasons else 'passed','reasons':reasons,'criteria':'pilot_screen_v1_predeclared_not_a_medical_standard',
              'test_predictions_used':False,'baseline':'B1'}
        write_json(root/cfg['post_gate_file'],gate)
        if reasons:
            write_json(root/'pipeline_status.json',{'status':'completed_with_gate_stop','completed_stages':history,'gate':gate,'elapsed_s':time.time()-started,
                'not_run':['B2','B3','B4','B5','B6'],'next':'review failure evidence without changing sealed test protocol'})
            return
        for method in ['B2','B3','B4','B5','B6']:run(method,{'ACTION':'train','METHOD':method})
        write_json(root/'pipeline_status.json',{'status':'completed','completed_stages':history,'elapsed_s':time.time()-started})
    except Exception as error:
        write_json(root/'pipeline_status.json',{'status':'failed','error':str(error),'completed_stages':history,'elapsed_s':time.time()-started})
        raise
    finally:
        from scripts.summarize import main as summarize
        summarize()


if __name__=='__main__':main()
