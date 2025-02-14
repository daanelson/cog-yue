import os
os.environ['HF_HUB_CACHE'] = './inference/models'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList


stage1_model = "m-a-p/YuE-s1-7B-anneal-en-cot"
stage2_model = "m-a-p/YuE-s2-1B-general"

print('loading stage 2')
model_stage2 = AutoModelForCausalLM.from_pretrained(
    stage2_model,
    torch_dtype=torch.bfloat16,
    cache_dir="./models",
    # device_map="auto",
    )

print('saving stage 2')
model_stage2.save_pretrained('./better_models/yue-s2-1b-general', safe_serialization=True)

print('loading stage 1')
model_stage1 = AutoModelForCausalLM.from_pretrained(
    stage1_model,
    torch_dtype=torch.bfloat16,
    cache_dir="./models",
    # device_map="auto",
    )
print('saving stage 1')
model_stage1.save_pretrained('./better_models/yue-s1-7b-anneal-en-cot', safe_serialization=True)