"""
将基座模型和适配器进行合并，并保存
"""
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
import argparse
parser = argparse.ArgumentParser(description="Merge Lora Model")
parser.add_argument("--base_model",type=str)
parser.add_argument("--peft_model",type=str)
parser.add_argument("--merge_model_path",type=str)
args = parser.parse_args()


base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
tokenizer = AutoTokenizer.from_pretrained(args.peft_model)
peft_model = PeftModel.from_pretrained(base_model,model_id=args.peft_model)

merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(args.merge_model_path)
tokenizer.save_pretrained(args.merge_model_path)