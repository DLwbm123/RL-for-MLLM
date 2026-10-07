"""Single-image Qwen2.5-VL adapter with post-window-restore feature replacement."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import re
import os
import subprocess
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from src.data import QUESTIONS, LABELS, smart_size
from src.objectives import token_logps

REVISION='cc594898137f460bfe9f0759e9844b3ce807cfb5'


def load_frozen_adapter(model,path,name):
    """Create FP32 LoRA storage before copying FP32 checkpoint values into a BF16 base."""
    if name in model.peft_config:raise ValueError('Adapter already exists')
    config=PeftConfig.from_pretrained(path);config.inference_mode=True
    model.add_adapter(name,config)
    for n,p in model.named_parameters():
        if 'lora_' in n and f'.{name}.' in n:p.data=p.data.float()
    return model.load_adapter(path,adapter_name=name,is_trainable=False,torch_device=str(next(model.parameters()).device))


class Model:
    def __init__(self,path,adapter=None,trainable=False,min_tokens=576,max_tokens=1024):
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,use_fast=False,
            min_pixels=min_tokens*784,max_pixels=max_tokens*784)
        self.processor.tokenizer.padding_side='left'
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(path,local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={'':'cuda'},attn_implementation='sdpa')
        self.visual=self.model.visual
        self.min_tokens,self.max_tokens=min_tokens,max_tokens
        self.forward_count=0;self.vision_count=0;self.score_count=0;self.generated_tokens=0;self.generation_calls=0;self.generating=False
        if adapter:
            self.model=PeftModel.from_pretrained(self.model,adapter,is_trainable=trainable)
        elif trainable:
            targets=[n for n,m in self.model.named_modules() if re.fullmatch(r'model\.layers\.\d+\.self_attn\.(q_proj|v_proj)',n)]
            if len(targets)!=2*self.model.config.num_hidden_layers:raise RuntimeError('Unexpected LoRA module names')
            self.model=get_peft_model(self.model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,target_modules=targets,
                bias='none',task_type='CAUSAL_LM'))
        for n,p in self.model.named_parameters():
            if p.requires_grad and trainable and ('lora_' not in n or '.visual.' in n):
                raise RuntimeError('Unexpected trainable parameter '+n)
        if not trainable:
            self.model.requires_grad_(False)
        else:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.model.enable_input_require_grads()
        for m in self.model.modules():
            if isinstance(m,torch.nn.Dropout):m.p=0.
        self.model.config.use_cache=False
        self.set_training(trainable)
        base=self.model.get_base_model() if isinstance(self.model,PeftModel) else self.model
        def count_generation(module,args,out):
            if self.generating:self.forward_count+=1
        self.counter_hook=base.register_forward_hook(count_generation)
        argv=subprocess.check_output(['ps','-p',str(os.getpid()),'-o','args='],text=True).strip()
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        own=[line for line in processes.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
        forbidden=os.environ.get('FORBIDDEN_ARGV_TERMS','busbra,evidence,qwen,sft,grpo,mllm').split(',')
        if any(word in (argv+' '.join(own)).lower() for word in forbidden):raise RuntimeError('Non-neutral process command line')
        print(json.dumps({'runtime_pid':os.getpid(),'argv':argv,'gpu_process':own,'model_revision':REVISION}),flush=True)

    def set_training(self,value):
        self.model.train(value)
        self.visual.eval()

    @contextmanager
    def supplied_features(self,features):
        original=self.visual.forward
        def forward(hidden_states,grid_thw):
            if int(grid_thw.prod(-1).sum())//4!=len(features):
                raise ValueError('Visual feature/grid mismatch')
            return features
        self.visual.forward=forward
        try:yield
        finally:self.visual.forward=original

    def prompt(self,image,task='A'):
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':QUESTIONS[task]}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[image.convert('RGB')],return_tensors='pt').to('cuda')
        grid=inputs['image_grid_thw'][0].tolist()
        if grid[0]!=1:raise ValueError('Only static single images supported')
        gh,gw=grid[1]//2,grid[2]//2
        if (grid[1]*14,grid[2]*14)!=smart_size(image.height,image.width,self.min_tokens,self.max_tokens):
            raise ValueError('Stored/processor resize mismatch')
        count=int((inputs['input_ids']==self.model.config.image_token_id).sum())
        if count!=gh*gw:raise ValueError('Image placeholder/grid mismatch')
        return inputs,(gh,gw)

    def encode(self,inputs):
        with torch.no_grad():
            out=self.visual(inputs['pixel_values'].to(self.visual.dtype),grid_thw=inputs['image_grid_thw'])
        self.vision_count+=1
        return out.detach()

    def score(self,inputs,features,answer=None,ids=None,include_eos=False):
        if ids is None:
            ids=self.processor.tokenizer.encode(answer,add_special_tokens=False)
            if include_eos:ids=ids+[self.processor.tokenizer.eos_token_id]
        if not len(ids):raise ValueError('Empty answer')
        ids=torch.as_tensor(ids,device='cuda',dtype=torch.long).reshape(1,-1)
        n=inputs['input_ids'].shape[1]
        batch={k:v for k,v in inputs.items()}
        batch['input_ids']=torch.cat([inputs['input_ids'],ids],dim=1)
        batch['attention_mask']=torch.ones_like(batch['input_ids'])
        mask=torch.zeros_like(batch['input_ids'],dtype=torch.bool);mask[:,n:]=True
        with self.supplied_features(features):
            output=self.model(**batch,use_cache=False)
        self.score_count+=1
        self.forward_count+=1
        # Slice first: FP32 log-softmax over answer positions only avoids full-vocabulary prompt activations.
        logits=output.logits[:,n-1:n+ids.shape[1]-1,:]
        values=torch.log_softmax(logits.float(),dim=-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)[0]
        return {'mean':values.mean(),'sum':values.sum(),'length':len(values),'tokens':values,'ids':ids[0],
                'prompt_length':n,'eos_included':include_eos}

    def classes(self,inputs,features,labels=LABELS):
        scores=[self.score(inputs,features,answer=x) for x in labels]
        return torch.stack([x['mean'] for x in scores]),scores

    @torch.no_grad()
    def generate(self,inputs,features,max_tokens=16,sample=True,diagnostic=False,sampling_config=None):
        before=self.model.training;self.set_training(False)
        self.generating=True
        try:
            with self.supplied_features(features):
                kwargs={'do_sample':sample,'max_new_tokens':max_tokens,'use_cache':True}
                if sample:
                    cfg=sampling_config or {'temperature':1.,'top_p':.95,'top_k':0,'max_new_tokens':max_tokens}
                    kwargs.update({k:cfg[k] for k in ['temperature','top_p','top_k','max_new_tokens']})
                    if 'repetition_penalty' in cfg:kwargs['repetition_penalty']=cfg['repetition_penalty']
                if diagnostic:kwargs.update(return_dict_in_generate=True,output_scores=True)
                effective=deepcopy(self.model.generation_config)
                unused=effective.update(**kwargs)
                if unused:raise ValueError('Unrecognized generation settings: '+str(unused))
                self.last_generation_settings=effective.to_dict()
                output=self.model.generate(**inputs,generation_config=effective,use_model_defaults=False)
        finally:
            self.generating=False
            self.set_training(before)
        sequence=output.sequences if diagnostic else output
        tokens=sequence[0,inputs['input_ids'].shape[1]:].tolist()
        if diagnostic:
            self.last_generation_logps=[float(torch.log_softmax(score[0].float(),-1)[token]) for score,token in zip(output.scores,tokens)]
        self.generation_calls+=1;self.generated_tokens+=len(tokens)
        return self.processor.tokenizer.decode(tokens,skip_special_tokens=True),tokens

    def save(self,path):
        Path(path).mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
