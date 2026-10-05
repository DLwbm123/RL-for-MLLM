"""Build a private, offline review form. All sufficient/insufficient labels start unset."""
import base64
import io
import json
import os
from pathlib import Path
from src.experiment import context,fixed_cases,sample_image
from src.regions import pixel_blur

if __name__=='__main__':
    root,data,_,cfg,frame,regions=context()
    rows=fixed_cases(frame,'train',32).to_dict('records');panels=[]
    for row in rows:
        region=regions.get(row['image_id'],{})
        if not region.get('eligible'):continue
        image=sample_image(data,row)
        views={'original':image,'lesion_weakened':pixel_blur(image,region['evidence']['box']),
               'control_weakened':pixel_blur(image,region['controls'][0]['box'])}
        for name,im in views.items():
            buffer=io.BytesIO();im.save(buffer,format='PNG');uri='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
            key=row['image_id']+'/'+name
            panels.append(f'<section data-key="{key}"><h3>{key}</h3><img src="{uri}"><p><select><option value="">尚未审核</option><option value="sufficient">足够</option><option value="insufficient">不足</option><option value="uncertain">不确定</option></select></p><textarea placeholder="可见替代证据、伪影及判断依据"></textarea></section>')
    html='''<!doctype html><meta charset="utf-8"><title>Private evidence review</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}main{display:flex;flex-wrap:wrap;gap:20px}section{width:30%}img{max-width:100%;height:320px;object-fit:contain}textarea{width:95%;height:70px}</style><h1>乳腺超声证据审核</h1><p>当前没有证据充分性真值。本地审核须由具备阅片资质者完成；削弱区域不等于病灶消失。多图同 Case 不自动代表同病灶或独立证据。</p><label>审核者代码 <input id="reviewer"></label><label>资质说明 <input id="qualification"></label><button id="export">导出本地审核 JSON</button><main>'''+''.join(panels)+'''</main><script>document.querySelector('#export').onclick=()=>{const reviewer=document.querySelector('#reviewer').value, qualification=document.querySelector('#qualification').value;if(!reviewer||!qualification){alert('请填写审核者代码和资质说明');return}const rows=[...document.querySelectorAll('section')].map(s=>({key:s.dataset.key,label:s.querySelector('select').value||null,notes:s.querySelector('textarea').value}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({reviewer,qualification,rows,created:new Date().toISOString()},null,2)],{type:'application/json'}));a.download='expert_review.json';a.click();URL.revokeObjectURL(a.href)}</script>'''
    p=root/'review/expert_review.html';p.write_text(html);print(p)
