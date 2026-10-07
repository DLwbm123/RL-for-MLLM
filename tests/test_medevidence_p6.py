"""Verify the all-wrong mechanism and the shared-gradient cap on synthetic data."""
import torch
from src.medevidence_p6 import training_advantages,common_coefficient
from src.medevidence_p4 import policy_terms


def main():
    for rewards in ([-1.]*4,[1.]*4,[-1.,1.,-1.,1.]):
        b=training_advantages(rewards,False,'B_GRPO');c=training_advantages(rewards,False,'C_NEGABS')
        assert torch.equal(c,torch.tensor(rewards))
        if len(set(rewards))==1:assert torch.equal(b,torch.zeros(4))
    r=[-.8,-.6,-.2,.1]
    assert torch.equal(training_advantages(r,True,'B_GRPO'),training_advantages(r,True,'C_NEGABS'))
    # A scalar log-probability surrogate: descending this loss decreases an all-wrong response's log probability.
    current=torch.tensor([-.7,-1.2],requires_grad=True);old=current.detach().clone();reference=old.clone()
    obj,kl=policy_terms(current,old,reference,-1.,torch.ones(2,dtype=torch.bool))
    (-obj).backward();assert torch.all(current.grad>0) and float(kl)==0
    current.grad=None;obj,_=policy_terms(current,old,reference,0.,torch.ones(2,dtype=torch.bool));(-obj).backward()
    assert torch.equal(current.grad,torch.zeros_like(current))
    lam=common_coefficient(2.,3.,5.);assert lam==.1 and max(lam*3/2,lam*5/2)<=.25
    try:training_advantages([-.5]*4,False,'C_NEGABS')
    except ValueError:pass
    else:raise AssertionError('Nonbinary negative reward accepted')
    # Native PEFT regression: secondary adapters otherwise round FP32 weights through BF16.
    import tempfile
    from transformers import LlamaConfig,LlamaForCausalLM
    from peft import get_peft_model,LoraConfig
    from src.model import load_frozen_adapter
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)
    model=get_peft_model(LlamaForCausalLM(config).to(torch.bfloat16),LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    for n,p in model.named_parameters():
        if 'lora_' in n:p.data.normal_()
    expected={n:p.detach().clone() for n,p in model.named_parameters() if 'lora_' in n}
    with tempfile.TemporaryDirectory() as path:
        model.save_pretrained(path);load_frozen_adapter(model,path,'reference')
        assert all(torch.equal(p,expected[n.replace('.reference.','.default.')]) for n,p in model.named_parameters() if '.reference.' in n)
    from types import SimpleNamespace
    from src.model import Model
    wrapper=Model.__new__(Model);wrapper.model=model;wrapper.visual=torch.nn.Identity()
    wrapper.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**kw:str(ids)))
    wrapper.generation_calls=wrapper.generated_tokens=0;model.generation_config.repetition_penalty=1.05
    resolved=[];base=model.get_base_model();prepare=base._prepare_generation_config
    def capture(*args,**kwargs):
        result=prepare(*args,**kwargs);resolved.append(result[0].to_dict());return result
    base._prepare_generation_config=capture
    wrapper.generate({'input_ids':torch.tensor([[1,3]]),'attention_mask':torch.ones(1,2,dtype=torch.long)},None,max_tokens=2,sampling_config={
        'temperature':1.,'top_p':1.,'top_k':0,'max_new_tokens':2,'repetition_penalty':1.})
    assert wrapper.last_generation_settings['repetition_penalty']==resolved[-1]['repetition_penalty']==1.
    base._prepare_generation_config=prepare
    assert not torch.cuda.is_initialized()
    print('P6 synthetic objective checks passed')


if __name__=='__main__':main()
