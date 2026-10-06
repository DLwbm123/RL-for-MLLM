"""One CPU regression check for fixed sampling and honest failure denominators."""
from collections import Counter
from src.medevidence_p2 import select_sets, loc_schedule, enrich, summarize, token_contract, token_loss_parts


def main():
    from scripts.medevidence_p2_pipeline import charged_seconds
    assert charged_seconds([{'charged_seconds':120}], [{'started':90},{'started':95},{'started':98}], 100)==137
    rows={str(i):{'case_id':f'{79-i:03d}','split':'train','pathology':'yes' if i<40 else 'no',
                  'boxes':[[10,20,30,40]] if i<40 else [],'width':100,'height':100} for i in range(80)}
    sets=select_sets(rows);assert sets==select_sets(dict(reversed(list(rows.items()))))
    assert len(set(sets['Fit32']+sets['Ref32']))==64
    plan=loc_schedule(sets['Fit32'],rows)
    assert all(len(v)==4 and sum(rows[k]['pathology']=='yes' for k in v)==2 for v in plan)
    assert Counter(k for update in plan for k in update)==Counter({k:32 for k in sets['Fit32']})
    positive=rows['0'];negative=rows['40']
    records=[enrich({'loc_text':'[]','p':.8},positive),
             enrich({'loc_text':'bad','p':.2},positive),
             enrich({'loc_text':'[[100,200,300,400],[100,200,300,400]]','p':.8},positive),
             enrich({'loc_text':'[]','p':.2},negative)]
    result=summarize(records)
    assert result['positive']['valid_empty']==1 and result['positive']['invalid']==1
    assert result['positive_region_recall']==1/3 and result['single_strict_success_count']==0
    assert result['positive_failure_categories']['all_GT_with_extras']['patients']==1
    assert result['class_loc_inconsistent_valid_count']==1
    assert result['single_IoU_unconditional']['n']==3 and result['overall_joint_supplement']==.25
    class SplitTokenizer:
        eos_token_id=999
        def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
        def decode(self,ids,skip_special_tokens=False):return ''.join(chr(i) for i in ids)
        def __call__(self,text,**kwargs):return {'input_ids':self.encode(text),'offset_mapping':[(i,i+1) for i in range(len(text))]}
    c=token_contract(SplitTokenizer(),'[[1,2,3,4]]')
    assert c['common_prefix_length']==1 and c['GT_divergence_token']==ord('[') and c['empty_divergence_token']==ord(']')
    parts=token_loss_parts([-1]*len(c['target_ids']),c)
    assert sum(v['tokens'] for v in parts.values())==len(c['target_ids'])
    assert abs(sum(v['contribution_to_sequence_mean_NLL'] for v in parts.values())-1)<1e-12
    assert token_contract(SplitTokenizer(),'[]')['common_prefix_length'] is None
    print('passed: fixed patient selection, L-only balance/exposure, invalid/empty/extra-box denominators, token divergence and disjoint loss spans')


if __name__=='__main__':main()
