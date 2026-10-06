"""Freeze identities, then replay existing private rewards on CPU. Never starts a model."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter,defaultdict
import numpy as np
import pandas as pd
import torch
from src.v3 import (require_cpu_config,cpu_replay_guard,components,reward_versions,order,group_audit,
                    errors,distribution,aggregate_groups,cluster_intervals)
from src.v2 import save


ARTIFACTS=['P4/B1/ced_diagnostics.json','P4/B1/groups.json','P4/B1/samples.jsonl','P4/B1/frozen_identity.json',
           'P4/B1/effective_generation.json','P4/summary.json','P1/summary.json','P3/summary.json','stage_status.json',
           'protocol/regions.json','protocol/subsets.json','protocol/manifest.csv','protocol/identity.json','protocol/protocol_lock.json']
OWN_SOURCE=['src/v3.py','scripts/replay_rsna_v3.py','tests/test_v3.py']


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths():
    root=Path(os.environ['OUTPUT_ROOT']);old=Path(os.environ['V2_OUTPUT_ROOT']);cfgpath=Path(os.environ['RUN_CONFIG'])
    cfg=read(cfgpath);require_cpu_config(cfg)
    if root.resolve()==old.resolve() or root.resolve().is_relative_to(old.resolve()):raise PermissionError('v3 must have independent outputs')
    return root,old,cfgpath,cfg


def prepare():
    root,old,cfgpath,cfg=paths();protocol=root/'protocol';protocol.mkdir(exist_ok=True)
    if (protocol/'protocol_lock.json').exists():raise FileExistsError('v3 R1 protocol already frozen')
    oldlock=read(old/'protocol/protocol_lock.json');oldcfg=old.parent/'code/configs/rsna_v2.json'
    if sha(oldcfg)!=oldlock['config_sha256']:raise ValueError('v2 config changed')
    for name,digest in oldlock['hashes'].items():
        if sha(old/'protocol'/name)!=digest:raise ValueError('v2 manifest identity changed: '+name)
    for name,digest in oldlock['source_hashes'].items():
        if sha(old.parent/'code'/name)!=digest:raise ValueError('v2 executed source changed: '+name)
    identity=read(old/'protocol/identity.json');base=Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct'
    for name,info in identity['base']['files'].items():
        st=(base/name).stat()
        if st.st_size!=info['bytes'] or st.st_mtime_ns!=info['mtime_ns']:raise ValueError('Base model metadata changed')
    adapter=Path(os.environ['V1_OUTPUT_ROOT'])/'B1'/identity['historical_B1']['checkpoint']
    if sha(adapter/'adapter_model.safetensors')!=identity['historical_B1']['sha256']:raise ValueError('B1 adapter identity changed')
    if sha(adapter/'adapter_config.json')!=identity['historical_B1']['config_sha256']:raise ValueError('B1 adapter configuration changed')
    oldconfig=read(oldcfg)
    for key in ['ced_epsilon','advantage_epsilon','near_zero_std']:
        if cfg[key]!=oldconfig['P4'][key]:raise ValueError('Frozen epsilon contract differs')
    if read(old/'P4/B1/effective_generation.json')!={**oldconfig['P4']['generation'],'do_sample':True,'eos_token_id':read(old/'P4/B1/effective_generation.json')['eos_token_id'],'pad_token_id':read(old/'P4/B1/effective_generation.json')['pad_token_id'],'use_cache':True}:raise ValueError('Unexpected effective generation fields')
    code=Path(__file__).resolve().parents[1]
    lock={'version':cfg['protocol_version'],'reference_commit':cfg['v2_reference_commit'],'origin_HEAD':os.environ['V3_ORIGIN_HEAD'],
          'config_sha256':sha(cfgpath),'v2_config_sha256':sha(oldcfg),'v2_protocol_lock_sha256':sha(old/'protocol/protocol_lock.json'),
          'artifact_hashes':{name:sha(old/name) for name in ARTIFACTS},'source_hashes':{name:sha(code/name) for name in OWN_SOURCE},
          'created_at':time.time(),'tolerances_fixed_before_replay':True,'small_std_is_descriptive_only':cfg['small_control_std'],
          'base_revision':identity['base']['revision'],'B1_checkpoint':identity['historical_B1']['checkpoint'],'new_GPU_seconds':0}
    save(protocol/'protocol_lock.json',lock)
    save(root/'stage_status.json',{
        'V3-R1':{'status':'ready','reason':'CPU identity checks passed; synthetic tests required before replay'},
        'V3-R2':{'status':'human_review_pending','reason':'Materials do not constitute real human review'},
        'V3-R3':{'status':'planned','reason':'Independent source audit'},
        'V3-G1':{'status':'not_authorized','blockers':['new_GPU_authorization','real_human_review','frozen_revised_manifest']},
        'V3-G2':{'status':'not_authorized','blockers':['new_GPU_authorization','verified_author_data_model_recipe']}})
    save(root/'budget.json',{'limit_GPU_seconds':0,'consumed_GPU_seconds':0,'training_enabled':False,'CPU_replay_seconds':0})
    print(json.dumps({'status':'ready','source_artifacts':len(ARTIFACTS),'new_GPU_seconds':0,'protocol_locked':True}))


def replay():
    root,old,cfgpath,cfg=paths();lock=read(root/'protocol/protocol_lock.json');code=Path(__file__).resolve().parents[1]
    if sha(cfgpath)!=lock['config_sha256']:raise ValueError('v3 frozen config changed')
    for name,digest in lock['source_hashes'].items():
        if sha(code/name)!=digest:raise ValueError('v3 R1 source changed after freeze')
    for name,digest in lock['artifact_hashes'].items():
        if sha(old/name)!=digest:raise ValueError('Private v2 artifact changed after freeze')
    receipt=read(root/'cpu_tests.json')
    if receipt.get('status')!='passed':raise PermissionError('Synthetic CPU tests must pass first')
    dest=root/'reward_replay';dest.mkdir(exist_ok=False);start=time.time()
    statuses=read(root/'stage_status.json');statuses['V3-R1']={'status':'running','reason':'existing-record CPU replay'};save(root/'stage_status.json',statuses)
    def deny_dataset(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            if Path(os.fsdecode(args[0])).resolve().is_relative_to(Path(os.environ['DATA_ROOT']).resolve()):raise PermissionError('R1 reads no dataset images or labels outside frozen protocol')
    sys.addaudithook(deny_dataset)
    try:
        with cpu_replay_guard(cfg):result=analyze(old,dest,cfg)
        result.update(status='completed',runtime_seconds=time.time()-start,test_images_read=0,new_GPU_seconds=0,new_model_measurements=0,training_run=False)
        save(dest/'summary.json',result)
        statuses['V3-R1']={'status':'completed','reason':'replay and algebra checked within frozen tolerances','patients':64,'groups':256,'candidates':2048,'runtime_seconds':result['runtime_seconds']}
        budget=read(root/'budget.json');budget['CPU_replay_seconds']=result['runtime_seconds'];save(root/'budget.json',budget)
        print(json.dumps({'status':'completed','counts':result['identity_counts'],'reward_cache_order':result['reward_cache_order'],'max_errors':result['replay_errors']}))
    except Exception as e:
        statuses['V3-R1']={'status':'blocked_replay','reason':str(e),'runtime_seconds':time.time()-start}
        save(dest/'failure.json',statuses['V3-R1']);raise
    finally:save(root/'stage_status.json',statuses)


def analyze(old,dest,cfg):
    diag=read(old/'P4/B1/ced_diagnostics.json');groups=read(old/'P4/B1/groups.json')
    samples=[json.loads(line) for line in (old/'P4/B1/samples.jsonl').read_text().splitlines()]
    regions=read(old/'protocol/regions.json');subsets=read(old/'protocol/subsets.json')['P4'];frame=pd.read_csv(old/'protocol/manifest.csv',keep_default_na=False,dtype={'image_id':str,'case_id':str})
    rows=frame.set_index('image_id').to_dict('index');identity=read(old/'protocol/identity.json');effective=read(old/'P4/B1/effective_generation.json')
    frozen=read(old/'P4/B1/frozen_identity.json')
    if not frozen['unchanged'] or frozen['before']!=frozen['after'] or frozen['before']['requires_grad_count']!=0:raise ValueError('Frozen model receipt inconsistent')
    by_id={r['image_id']:r for r in diag};selected={r['image_id']:r for r in subsets}
    if len(by_id)!=64 or len(diag)!=64 or set(by_id)!=set(selected) or len(groups)!=256 or len(samples)!=2048:raise ValueError('P4 coverage mismatch')
    if len({r['case_id'] for r in diag})!=64:raise ValueError('Patient/image identity is not one-to-one')
    for k,d in by_id.items():
        row=rows[k]
        if row['case_id']!=d['case_id'] or row['split']!='train' or selected[k]['stratum']!=d['stratum']:raise ValueError('Selected patient/group identity mismatch')
        if d['eligible_geometry']!=regions.get(k,{}).get('eligible',False):raise ValueError('Geometry eligibility changed')
        if set(d['labels'])!=({'no','yes'} if d['eligible_geometry'] else set()):raise ValueError('Missing label cache')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(Path(os.environ['MODEL_ROOT'])/'Qwen2.5-VL-7B-Instruct',local_files_only=True)
    if {k:tokenizer.encode(k,add_special_tokens=False) for k in ['no','yes']}!=identity['label_ids'] or tokenizer.eos_token_id!=identity['eos_id']:raise ValueError('Tokenizer identity changed')
    grouped=defaultdict(list);seen=set();text_errors=0;truncated=0;illegal=0;reward_error=[];adv_error=[];component_errors=[]
    for s in samples:
        k=s['image_id'];key=(k,s['group'],s['slot'])
        if key in seen or k not in selected or s['group'] not in range(4) or s['slot'] not in range(8):raise ValueError('Duplicate or invalid candidate identity')
        seen.add(key);d=by_id[k];row=rows[k]
        if s['case_id']!=row['case_id'] or s['stratum']!=d['stratum'] or s['model']!='B1':raise ValueError('Candidate provenance mismatch')
        if s['seed']!=selected[k]['sampling_seeds'][s['group']*8+s['slot']]:raise ValueError('Candidate seed mismatch')
        text=tokenizer.decode(s['tokens'],skip_special_tokens=True)
        if text!=s['text']:raise ValueError('Raw text differs from stored token decoding')
        label=text.strip();valid=label in ['no','yes'];c=float(label==row['pathology']) if valid else -.1
        if s['normalized_label']!=(label if valid else None) or s['invalid_format']!=(not valid) or s['correctness_reward']!=c:raise ValueError('Text/label/correctness mapping mismatch')
        eos=effective['eos_token_id'];eos=[eos] if isinstance(eos,int) else eos
        clipped=bool(len(s['tokens'])>=effective['max_new_tokens'] and (not s['tokens'] or s['tokens'][-1] not in eos))
        if clipped!=s['truncated']:raise ValueError('Truncation flag mismatch')
        truncated+=clipped;illegal+=not valid
        expected=-.1 if not valid else d['labels'][label]['reward'] if d['eligible_geometry'] else c
        legacy=expected if d['eligible_geometry'] or not valid else 0.
        reward_error.extend([abs(s['revised_CED_reward']-expected),abs(s['legacy_B4_reward']-legacy)])
        grouped[(k,s['group'])].append(s)
    if len(grouped)!=256 or any(len(v)!=8 for v in grouped.values()) or set(Counter(k for k,g in grouped).values())!={4}:raise ValueError('Incomplete group coverage')
    # All components are rebuilt for both labels, including patients with no mixed sampled group.
    decomposed=[];case_lookup={};float64_difference=[];formula_mismatches=[]
    for k,d in by_id.items():
        if not d['eligible_geometry']:continue
        target=rows[k]['pathology'];labels={};variants={}
        for label,cached in d['labels'].items():
            c=int(label==target);v=components(cached['D_E'],cached['D_N'],c,cfg);v64=components(cached['D_E'],cached['D_N'],c,cfg,torch.float64)
            for old_key,new_key in [('reward','final_reward'),('raw_M','M'),('std','std_D_N'),('m','m'),('gate','g')]:component_errors.append(errors([v[new_key]],[cached[old_key]],cfg['relative_error_denominator_floor']))
            if cached['saturated']!=v['gate_saturated']:raise ValueError('Cached gate saturation mismatch')
            labels[label]={'float32':v,'float64':v64,'cached':cached};variants[label]=reward_versions(c,v['m'],v['g'],cfg)
            reward_error.append(abs(v['final_reward']-cached['reward']));float64_difference.append(abs(v64['final_reward']-v['final_reward']))
        correct=labels[target]['float32'];wrong_label='no' if target=='yes' else 'yes';wrong=labels[wrong_label]['float32']
        gap=by_id[k]['labels'][target]['reward']-by_id[k]['labels'][wrong_label]['reward']
        inequality=cfg['tie_weight']*(wrong['m']-correct['m'])>correct['g']
        if inequality!=(gap<0):formula_mismatches.append(k)
        row={'image_id':k,'case_id':d['case_id'],'stratum':d['stratum'],'target':target,'labels':labels,'versions':variants,
             'correct_minus_wrong_reward_gap':gap,'order':order(gap),'reversal_inequality_lhs':cfg['tie_weight']*(wrong['m']-correct['m']),
             'reversal_inequality_rhs':correct['g'],'reversal_inequality_holds':inequality,
             'version_gaps':{v:variants[target][v]-variants[wrong_label][v] for v in cfg['reward_versions']}}
        if row['version_gaps']['R-G']<0 or row['version_gaps']['R-A']<.8-cfg['reward_replay_atol']:raise ValueError('Protected reward ordering bound violated')
        decomposed.append(row);case_lookup[k]=row
    save(dest/'case_decomposition.json',decomposed)
    if formula_mismatches:raise ValueError('Reversal inequality does not explain cached order')
    if max(reward_error+[e['max_absolute'] for e in component_errors])>cfg['reward_replay_atol']:raise ValueError('Reward/components replay exceeds frozen absolute tolerance')
    # Original candidate values anchor normalization checks; ablations use the same candidate labels.
    replayed=[];historical_counts=defaultdict(Counter);seen_groups=set()
    for original in groups:
        k=original['image_id'];key=(k,original['group'])
        if key in seen_groups or key not in grouped:raise ValueError('Duplicate or missing stored group')
        seen_groups.add(key);batch=sorted(grouped[key],key=lambda s:s['slot']);d=by_id[k];target=rows[k]['pathology'];wrong='no' if target=='yes' else 'yes'
        correct=[s['correctness_reward'] for s in batch];answers=[s['text'].strip() for s in batch]
        if original['case_id']!=d['case_id'] or original['stratum']!=d['stratum'] or original['answers']!=answers or original['correct_answers']!=sum(c==1 for c in correct) or original['legal_answers']!=sum(c>=0 for c in correct):raise ValueError('Stored group summary mismatch')
        versions={}
        for version in cfg['reward_versions']:
            if d['eligible_geometry']:
                cr=case_lookup[k]['versions'][target][version];wr=case_lookup[k]['versions'][wrong][version]
            else:cr,wr=1.,0.
            values=[-.1 if c<0 else cr if c==1 else wr for c in correct]
            if version=='R-H' and d['eligible_geometry']:
                cr=d['labels'][target]['reward'];wr=d['labels'][wrong]['reward'];values=[s['revised_CED_reward'] for s in batch]
            versions[version]=group_audit(correct,values,cr,wr,cfg)
        for name,sample_key in [('correctness_only','correctness_reward'),('legacy_B4','legacy_B4_reward'),('revised_CED','revised_CED_reward')]:
            stored=original[name];values=[s[sample_key] for s in batch]
            if stored['correctness_rewards']!=correct:raise ValueError('Stored correctness vector mismatch')
            reward_error.extend(np.abs(np.asarray(stored['ced_rewards'])-values).tolist())
            from src.v3 import advantage
            actual,std=advantage(values,cfg);base,_=advantage(correct,cfg)
            adv_error.append(errors(actual,stored['ced_advantage'],cfg['relative_error_denominator_floor']));adv_error.append(errors(base,stored['correctness_advantage'],cfg['relative_error_denominator_floor']))
            for stratum in ['all',d['stratum']]:
                count=historical_counts[(name,stratum)];count['groups']+=1;count['zero_advantage_groups']+=bool(np.all(actual==0));count['legal_answers']+=sum(c>=0 for c in correct)
                count['correct_answers']+=sum(c==1 for c in correct);count['all_same_answer_groups']+=len(set(answers))==1
                mixed=set(correct)=={0.,1.};gap=min(v for v,c in zip(values,correct) if c==1)-max(v for v,c in zip(values,correct) if c==0) if mixed else None
                count['ordering_reversals']+=bool(gap is not None and gap<0);usable=bool(mixed and std>=cfg['near_zero_std'] and gap>0)
                count['usable_groups']+=usable;count['delta_A_lt_1e4_groups']+=bool(usable and np.max(np.abs(actual-base))<cfg['historical_delta_A_threshold'])
            if abs(float(np.max(np.abs(actual-base)))-stored['delta_A'])>cfg['advantage_replay_atol']:raise ValueError('Stored Delta_A replay mismatch')
        replayed.append({'image_id':k,'case_id':d['case_id'],'stratum':d['stratum'],'group':original['group'],'versions':versions})
    save(dest/'group_advantage_replay.json',replayed)
    original_summary=read(old/'P4/summary.json')['models']['B1']['variants'];differences=[]
    for (name,stratum),counts in historical_counts.items():
        for field,n in counts.items():
            expected=original_summary[name][stratum][field]
            if n!=expected:differences.append({'variant':name,'stratum':stratum,'field':field,'observed':n,'expected':expected})
    if differences:save(dest/'historical_count_differences.json',differences);raise ValueError('Historical aggregate does not match replay')
    if max(reward_error)>cfg['reward_replay_atol'] or max(e['max_absolute'] for e in adv_error)>cfg['advantage_replay_atol']:raise ValueError('Reward or advantage exceeds frozen tolerance')
    if any(n['unexplained_exceeds_tolerance'] for g in replayed for v in g['versions'].values() for n in v['normalizations'].values()):raise ValueError('Unexplained binary advantage mismatch')
    geo=[g for g in replayed if g['stratum']=='geometry_only_positive'];observed={g['case_id'] for g in geo if g['versions']['R-H']['mixed'] and g['versions']['R-H']['sign_b']<0}
    potential={c['case_id'] for c in decomposed if c['order']=='reversed'}
    for c in decomposed:
        c['observed_reversal_mixed_groups']=sum(g['case_id']==c['case_id'] and g['versions']['R-H']['mixed'] and g['versions']['R-H']['sign_b']<0 for g in geo)
    save(dest/'case_decomposition.json',decomposed)
    transitions={}
    for variant in ['R-G','R-A']:
        transitions[variant]={'patients':dict(Counter(c['order']+' -> '+order(c['version_gaps'][variant]) for c in decomposed)),
                              'mixed_groups':dict(Counter(g['versions']['R-H']['order']+' -> '+g['versions'][variant]['order'] for g in geo if g['versions']['R-H']['mixed']))}
    component_distributions={}
    for condition in ['reversed','positive','tie']:
        cases=[c for c in decomposed if c['order']==condition];metrics={}
        for role in ['correct','wrong']:
            values=[c['labels'][c['target'] if role=='correct' else ('no' if c['target']=='yes' else 'yes')]['float32'] for c in cases]
            for field in ['D_E','mean_D_N','std_D_N','M','z','m','g','correctness_gate_term','tie_term','final_reward']:
                metrics[role+'/'+field]=distribution([v[field] for v in values])
        component_distributions[condition]={'patients':len(cases),'reward_gap':distribution([c['correct_minus_wrong_reward_gap'] for c in cases]),'components':metrics}
    all_parts=[p['float32'] for c in decomposed for p in c['labels'].values()]
    strata={name:{version:aggregate_groups([g for g in replayed if name=='all' or g['stratum']==name],version) for version in cfg['reward_versions']} for name in ['all','geometry_only_positive','ineligible_positive','negative']}
    return {'analysis_type':'CPU replay and attribution of the same v2 records; not an independent experiment',
            'identity_counts':{'patients':64,'groups':256,'candidates':2048,'geometry_patients':len(decomposed),'geometry_groups':len(geo),'illegal_candidates':illegal,'truncated_candidates':truncated,'duplicate_or_missing_records':0,'raw_text_token_decode_mismatch':text_errors},
            'historical_aggregate_differences':differences,'tokenizer_identity_verified':True,'frozen_identity_receipt_verified':True,
            'precision_limit':'Stored JSON numbers preserve binary float32 values; pre-cache forward intermediates and logits are not present, so their serialization precision is NA.',
            'reward_cache_order':{'observed_reversal_patients':len(observed),'potential_reversed_cache_patients':len(potential),'reversed_without_sampled_mixed_group_patients':len(potential-observed),'positive_cache_patients':sum(c['order']=='positive' for c in decomposed),'tied_cache_patients':sum(c['order']=='tie' for c in decomposed),'inequality_mismatches':0},
            'numerical_conditions':{'case_label_pairs':len(all_parts),'control_std_exact_zero':sum(p['control_std_zero'] for p in all_parts),'small_nonzero_control_std':sum(p['control_std_small_nonzero'] for p in all_parts),'small_threshold':cfg['small_control_std'],'gate_saturated':sum(p['gate_saturated'] for p in all_parts),'gate_saturation_bounds':[cfg['gate_saturation_low'],cfg['gate_saturation_high']]},
            'replay_errors':{'reward_max_absolute':max(reward_error),'component_max_absolute':max(e['max_absolute'] for e in component_errors),'component_max_relative_with_floor':max(e['max_relative_with_floor'] for e in component_errors),'advantage_max_absolute':max(e['max_absolute'] for e in adv_error),'advantage_max_relative_with_floor':max(e['max_relative_with_floor'] for e in adv_error),'relative_floor':cfg['relative_error_denominator_floor'],'float64_vs_float32_recomputed_reward':distribution(float64_difference)},
            'component_distributions':component_distributions,'strata':strata,'transitions':transitions,
            'patient_cluster_bootstrap':cluster_intervals(decomposed,geo,cfg),
            'regional_semantics':'unresolved; no human review inferred from algebra or reward signs'}


if __name__=='__main__':
    op=os.environ.get('V3_OPERATION','replay')
    if op=='prepare':prepare()
    elif op=='replay':replay()
    else:raise PermissionError('No GPU/training stage is authorized or implemented by this CPU entry')
