from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",type=str,default="model/Qwen3-0.6B-Base",help="模型路径")
parser.add_argument("--tokenizer_name",type=str,default=None,help="模型路径")
parser.add_argument("--prompt",type=str,default="你好",help="提示词")

args = parser.parse_args()
model_name = args.model_name
tokenizer_name = args.tokenizer_name
prompt = args.prompt
print('输入的prompt为：',prompt[:30])
print("模型路径为：",model_name)

tokenizer_name = tokenizer_name or model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,device_map="auto")
inputs = tokenizer.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True,return_tensors="pt",max_length=2048,enable_thinking=True)
print("得到的inputs为：",inputs)
input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")

model = AutoModelForCausalLM.from_pretrained(model_name,dtype = torch.float16,device_map="auto")
model.eval()
output_ids = model.generate(inputs=input_ids,attention_mask=attention_mask,max_new_tokens=5000,eos_token_id=[151643,151645])[0][len(input_ids[0]):].tolist()
print('当前的output_ids位：',output_ids)
try:
    # 151668为</think>的token_id
    index = len(output_ids) - output_ids[::-1].index(151668)
except Exception:
    # 没有thinking
    index = 0
thinking_content = tokenizer.decode(output_ids[:index],skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:],skip_special_tokens=True).strip("\n")
print("thinking content为：",thinking_content)
print("content为：",content)
