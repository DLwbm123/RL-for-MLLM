import io
import json
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pandas as pd
import pytest
import torch
from src.data import bbox,transform_box,group_rows,split_frame
from src.regions import region_tokens,replace_features,pixel_blur,build_regions
from src.objectives import token_logps,correctness,advantages,grpo_loss,ced_reward,evidence_loss,stability_loss
from src.evaluation import box_iou,cluster_bootstrap,fit_calibration,select_checkpoint
from scripts.download_busbra import verify,extract,download


def test_archive_integrity_and_safety(tmp_path):
    bad=tmp_path/'bad.zip';bad.write_bytes(b'not a zip')
    with pytest.raises(ValueError,match='MD5 mismatch'):verify(bad)
    with pytest.raises(zipfile.BadZipFile):extract(bad,tmp_path/'out')
    with zipfile.ZipFile(tmp_path/'traversal.zip','w') as z:z.writestr('../escape','bad')
    with pytest.raises(ValueError,match='Unsafe'):extract(tmp_path/'traversal.zip',tmp_path/'out')
    with zipfile.ZipFile(tmp_path/'symlink.zip','w') as z:
        info=zipfile.ZipInfo('link');info.create_system=3;info.external_attr=0o120777<<16;z.writestr(info,'/tmp')
    with pytest.raises(ValueError,match='Unsafe'):extract(tmp_path/'symlink.zip',tmp_path/'out')
    with zipfile.ZipFile(tmp_path/'ok.zip','w') as z:z.writestr('folder/file','hello')
    extract(tmp_path/'ok.zip',tmp_path/'out');assert (tmp_path/'out/folder/file').read_text()=='hello'
    corrupted=tmp_path/'crc.zip';corrupted.write_bytes((tmp_path/'ok.zip').read_bytes().replace(b'hello',b'hXllo'))
    with pytest.raises(ValueError,match='Corrupt'):extract(corrupted,tmp_path/'out2')


def test_download_resume_and_range_rejection(tmp_path,monkeypatch):
    import urllib.request
    path=tmp_path/'file.zip';part=tmp_path/'file.zip.part';part.write_bytes(b'abc')
    class Response(io.BytesIO):
        status=206
        headers={'Content-Range':'bytes 3-5/6'}
    def open_request(request,timeout):
        assert request.headers['Range']=='bytes=3-'
        return Response(b'def')
    monkeypatch.setattr(urllib.request,'urlopen',open_request)
    download('https://example.invalid/file',path,6,retries=1)
    assert path.read_bytes()==b'abcdef'
    path.unlink();part.write_bytes(b'abc')
    Response.headers={'Content-Range':'bytes 0-2/6'}
    with pytest.raises(ValueError,match='Invalid resume'):download('https://example.invalid/file',path,6,retries=1)


def test_grouped_split_duplicate_transitivity():
    rows=[{'case_id':str(i),'pathology':'benign' if i<20 else 'malignant','image_sha256':str(i),'pixel_sha256':str(i)} for i in range(40)]
    rows.append({**rows[0],'case_id':'40'});rows.append({**rows[1],'case_id':'40'})
    frame=pd.DataFrame(rows);frame['duplicate_group']=group_rows(frame);frame['split']=split_frame(frame)
    assert frame.loc[frame.case_id.isin(['0','1','40']),'split'].nunique()==1
    for col in ['case_id','image_sha256','pixel_sha256']:assert frame.groupby(col).split.nunique().max()==1


def test_coordinates_and_grid():
    mask=np.zeros((75,123),bool);mask[13:31,7:25]=1
    box=bbox(mask);assert box==[7,13,25,31]
    norm=transform_box(box,123,75);back=transform_box(norm,123,75,inverse=True)
    assert np.max(np.abs(np.array(back)-box))<=.123
    assert region_tokens([0,0,25,25],100,100,4,4)==[0]
    assert region_tokens([25,0,50,25],100,100,4,4)==[1]
    assert region_tokens([0,0,0,20],100,100,4,4)==[]
    assert region_tokens([0,0,100,100],100,100,3,5)==list(range(15))
    assert bbox(np.zeros((3,4))) is None


def test_intervention_isolation_and_gradients():
    original=torch.arange(24,dtype=torch.float32).reshape(6,4).requires_grad_()
    before=original.detach().clone();out=replace_features(original,[1,2],[4,5])
    assert torch.equal(out[[0,3,4,5]],original[[0,3,4,5]])
    assert torch.equal(original,before)
    assert torch.equal(replace_features(original,[],[]),original)
    out.sum().backward();assert original.grad is not None
    with pytest.raises(ValueError):replace_features(original,[1],[1])


def test_answer_span_padding_and_mean():
    logits=torch.tensor([[[0.,1.,2.],[2.,0.,1.],[1.,0.,2.],[5.,0.,0.]]])
    ids=torch.tensor([[0,2,0,1]]);mask=torch.tensor([[0,1,1,0]])
    result=token_logps(logits,ids,mask)
    expected=torch.log_softmax(logits[0,:2],-1)[[0,1],[2,0]]
    assert result['length'].item()==2
    assert torch.allclose(result['sum'],expected.sum())
    assert torch.allclose(result['mean'],expected.mean())
    with pytest.raises(ValueError):token_logps(logits,ids,torch.zeros_like(mask))


def test_rewards_and_zero_variance():
    assert correctness('benign','benign')==1
    assert correctness('malignant','benign')==0
    assert correctness('benign.','benign')==-.1
    assert torch.equal(advantages(torch.ones(8)),torch.zeros(8))
    a=advantages(torch.tensor([1.,0.,1.,0.]));assert torch.isfinite(a).all()
    r,d=ced_reward(1.,torch.tensor(.1,requires_grad=True),torch.tensor([0.,0.,0.]))
    assert not r.requires_grad and d['std']==0 and torch.isfinite(r)
    new=torch.tensor([-.5,-.7],requires_grad=True)
    loss=grpo_loss(new,new.detach(),torch.tensor(0.));loss.backward();assert (new.grad==0).all()


def test_auxiliary_gradient_and_stopgrad():
    # A trainable low-rank projection exercises differentiability through all image views.
    a=torch.nn.Parameter(torch.tensor([[.3,.4]]));b=torch.nn.Parameter(torch.tensor([[.2],[.1]]))
    x=torch.tensor([1.,2.]);e=torch.tensor([.5,.8]);n=torch.tensor([[.8,2.],[1.,1.8],[.9,1.9]])
    project=lambda z:(z @ (b@a).T)
    orig=project(x);evid=project(e);ctrl=project(n)
    loss,_=evidence_loss(orig,evid,ctrl,0,delta_rel=2,delta_abs=2)
    loss.backward();assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    a.grad=None;b.grad=None
    zero,_=evidence_loss(project(x),project(e),project(n),0,weight=0)
    zero.backward();assert a.grad.abs().sum()==0 and b.grad.abs().sum()==0
    orig=torch.tensor([1.,2.],requires_grad=True);ctrl=torch.tensor([[2.,0.],[1.,1.]],requires_grad=True)
    stability_loss(orig,ctrl).backward();assert orig.grad is None and ctrl.grad.abs().sum()>0
    orig.grad=None;(-orig.log_softmax(-1)[0]).backward();assert orig.grad.abs().sum()>0


def test_invalid_boxes_and_case_cluster_bootstrap():
    assert box_iou('nonsense',[0,0,1000,1000])==0
    assert box_iou('[10,10,5,5]',[0,0,1000,1000])==0
    assert box_iou('[0,0,1000,1000]',[0,0,1000,1000])==1
    df=pd.DataFrame({'case_id':['a','a','b'],'x':[1,1,0]})
    def stat(sample):
        # Cases always carry both correlated images when resampled.
        assert (sample.case_id=='a').sum()%2==0
        return sample.x.mean()
    out=cluster_bootstrap(df,stat,repeats=25);assert out['valid_replicates']==25
    with pytest.raises(ValueError):fit_calibration([[1,0],[0,1]],[0,1],'test')
    assert select_checkpoint([{'auroc':.8,'step':20},{'auroc':.8,'step':10}])['step']==10


def test_pixel_intervention_keeps_outside():
    from PIL import Image
    a=np.arange(10000,dtype=np.uint8).reshape(100,100);im=Image.fromarray(a).convert('RGB')
    out=np.asarray(pixel_blur(im,[20,30,60,70]))
    assert np.array_equal(out[:30],np.asarray(im)[:30])
    assert np.array_equal(out[70:],np.asarray(im)[70:])
    assert not np.array_equal(out[30:70,20:60],np.asarray(im)[30:70,20:60])


def test_control_pixel_and_token_area_matching():
    from PIL import Image
    rng=np.random.default_rng(3);image=Image.fromarray(rng.integers(20,160,(300,400),dtype=np.uint8)).convert('RGB')
    mask=np.zeros((300,400),bool);mask[120:145,175:205]=True
    region=build_regions(image,mask,[175,120,205,145])
    assert len(region['controls'])==3
    for control in region['controls']:
        box=control['box'];assert np.isclose((box[2]-box[0])*(box[3]-box[1]),30*25)
        assert len(control['tokens'])==len(region['evidence']['tokens'])
        assert set(control['tokens']).isdisjoint(region['evidence']['tokens'])
