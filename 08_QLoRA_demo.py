from transformers import BitsAndBytesConfig

import torch
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # 使用NF4量化
    bnb_4bit_quant_type="nf4",
    # 是否使用双重量化
    bnb_4bit_use_double_quant=False,
    # 反量化之后的数据类型，会使用该类型参与到前向传播的计算
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained("model/Qwen3-8B", quantization_config =quantization_config ).to("cuda")


from peft import get_peft_model,prepare_model_for_kbit_training
prepared_model = prepare_model_for_kbit_training(model)

from peft import LoraConfig
lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","v_proj"],
    task_type="CAUSAL_LM"
)
quantized_peft_model = get_peft_model(prepared_model,lora_config)

from datasets import load_dataset

# 1、处理数据，处理成type为language modeling ，format为对话格式的数据
psychology_data = load_dataset("json",data_files={"train":r"./data/psychology_data.jsonl"})
psychology_data = psychology_data.train_test_split(test_size=0.1)

# 将数据，转换成带messages，每个message是role 和content的形式
from typing import Dict,List
def data_convert(examples:Dict[str,List]):
    """
    将数据，转换成带messages，每个message是role 和content的形式
    """
    conversation_example_list = examples["conversation"]
    examples_message_list = []
    for example in conversation_example_list:
        message_list = []
        conversation = example[0]
        message_list.append({"role":"user","content":conversation["human"]})
        message_list.append({"role":"assistant","content":conversation["assistant"]})
        examples_message_list.append(message_list)

    return {"messages":examples_message_list}
         

mapped_psychology_data = psychology_data.map(data_convert,batched=True,remove_columns=psychology_data["train"].column_names)

# 2、构造SFTConfig实例
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/08_QLoRA_DEMO"
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-8B/")
training_args = SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=1000,
    num_train_epochs=1,
    logging_strategy="steps",
    logging_steps=100,
    report_to="tensorboard",
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=100,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    output_dir="finetuned/08_QLoRA_DEMO",
    bf16=True,
    max_length=710,
    assistant_only_loss=True,
    chat_template_path="./chat_template.jinja"
)

# 3、构造trainer

from trl.trainer.sft_trainer import SFTTrainer

trainer = SFTTrainer(
    args=training_args,
    model=quantized_peft_model,
    train_dataset=mapped_psychology_data["train"],
    eval_dataset=mapped_psychology_data["test"],
    processing_class=tokenizer
)


trainer.train()
trainer.save_model("finetuned/08_QLoRA_DEMO")