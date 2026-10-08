"""Interrupted extraction retains complete records and resumes without changing features."""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from src.medevidence_p7 import Run as FeatureRun
from src.medevidence_p9_resume import FeatureChunks,diagnose,report
from src.v2 import save


class FakeModel:
    def __init__(self,*args):self.model=torch.nn.Linear(1,1).requires_grad_(False)
    def prompt(self,image,task='A'):return {},(1,1)
    def encode(self,inputs):return torch.tensor([[1.,2.,3.,4.]])


def main():
    ids=['a','b','c'];identity={'feature_dimension':16,'candidate_version':'fixed'}
    run=FeatureRun.__new__(FeatureRun);run.update=lambda *a,**k:None
    run.image=lambda k:Image.new('RGB',(28,28));run.rows={k:{'boxes':[[0,0,28,28]]} for k in ids}
    run.candidates={k:{'boxes':[[0,0,28,28]],'scores':[.5]} for k in ids};calls=[]
    def interrupt():
        calls.append(1)
        if len(calls)==3:raise TimeoutError('synthetic budget stop')
    with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'MODEL_ROOT':'unused'}),patch('src.model.Model',FakeModel):
        path=Path(folder)/'chunks';cache=FeatureChunks(path,ids,identity);run.tick=interrupt
        try:run.features(ids,cache=cache)
        except TimeoutError:pass
        else:raise AssertionError('Synthetic interruption was not reached')
        restored=FeatureChunks(path,ids,identity);assert list(restored.records)==ids[:2] and restored.persisted==2
        (path/'chunk_0000.pt.tmp').write_bytes(b'incomplete temporary write')
        run.tick=lambda:None;resumed=run.features(ids,cache=restored);assert restored.persisted==3
        reference=run.features(ids)
        for k in ids:
            for field in reference[k]:assert torch.equal(reference[k][field],resumed[k][field])
        with patch('src.model.Model',side_effect=AssertionError('Complete cache must skip model loading')):
            assert set(run.features(ids,cache=FeatureChunks(path,ids,identity)))==set(ids)
        try:FeatureChunks(path,ids,{**identity,'candidate_version':'changed'})
        except ValueError:pass
        else:raise AssertionError('Mismatched candidate provenance accepted')
        # A full block is durable before explicit final flush.
        full=FeatureChunks(Path(folder)/'full',[str(i) for i in range(33)],identity)
        for i in range(32):full.add(str(i),reference['a'])
        assert full.persisted==32 and len(FeatureChunks(full.path,full.ids,identity).records)==32
        d=diagnose(reference,ids,np.array([[.8,.7],[.2,.3],[.8,.1]]),.6)
        assert d['probability_conflict']['count']==1 and d['groups']['top1_supported_positive']['rules']['utility']['abstain']==1
        base=Path(folder)/'report';(base/'outputs').mkdir(parents=True);(base/'code/reports').mkdir(parents=True)
        with patch.dict(os.environ,{'PILOT_ROOT':str(base)}):
            save(base/'outputs/FINAL.json',{'status':'stopped_budget','detector_steps':0})
            save(base/'outputs/gpu_ledger.json',{'new_charged_seconds':1,'combined_GPU_hours':1.7})
            save(base/'outputs/resume_receipt.json',{'reused_detector_steps':2400});report()
            assert '/0.75' in (base/'code/reports/medevidence_p9_resume_decision.md').read_text()
    assert not torch.cuda.is_initialized();print('P9 chunk durability/interruption/equivalence/provenance/diagnostics CPU checks passed')


if __name__=='__main__':main()
