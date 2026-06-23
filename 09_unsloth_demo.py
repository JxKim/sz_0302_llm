# 优化一：将unsloth的import放在最前面，Unsloth能够为多个库打补丁：trl, transformers, peft库
from unsloth import FastLanguageModel
from trl.trainer.sft_trainer import SFTTrainer

# 优化二：当我们传入model_name，并且不传use_exact_model_name和local_files_only参数时，Unsloth，会将我们传入的model_name改写，直接从它的huggingface拉取unsloth已经量化好的模型
# 我们传入 ./model/Qwen3-8B,并且加上use_exact_model_name和local_files_only，才使得Unsloth不会从huggingface加载模型
# 1、加载模型，并进行量化的过程
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name="./model/Qwen3-8B",
    load_in_4bit=True,
    use_exact_model_name=True,
    local_files_only=True
)

quantized_peft_model = FastLanguageModel.get_peft_model(
    model=model,
    # 差异点：target_modules：不能传入all-linear，如果需要对所有的线性层插入LoRA模块，需要传入["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    target_modules=["q_proj","v_proj",],
    bias="none",
    r=8,
    lora_alpha=8,
    lora_dropout=0.05
)

from datasets import load_dataset

# 1、处理数据，处理成type为language modeling ，format为对话格式的数据
psychology_data = load_dataset("json",data_files={"train":r"./data/psychology_data.jsonl"})

psychology_data = psychology_data["train"].train_test_split(test_size=0.1)

# 将数据，转换成带messages，每个message是role 和content的形式
from typing import Dict,List
def data_convert(examples:Dict[str,List]):
    """
    将数据，先转换成messages，然后再通过tokenizer的apply_chat_template，将其格式化，放到text这个key下面
    """
    conversation_example_list = examples["conversation"]
    message_text_list = []
    for example in conversation_example_list:
        message_list = []
        conversation = example[0]
        message_list.append({"role":"user","content":conversation["human"]})
        message_list.append({"role":"assistant","content":conversation["assistant"]})
        # 后面给到Unsloth的，是调用完chat_template之后的结果
        # 优化点三：通过Unlsoth，可以自己去调用chat_template，可以使用qwen3原生的聊天模板，后续需要基于assistant回答部分计算损失，不需要依赖chat template当中 %genearation%
        message_text = tokenizer.apply_chat_template(message_list,tokenize=False,add_generation_prompt=False)
        message_text_list.append(message_text)

    return {"text":message_text_list}
         

mapped_psychology_data = psychology_data.map(data_convert,batched=True,remove_columns=psychology_data["train"].column_names)


# 2、构造SFTConfig实例
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/09_Unsloth_DEMO"
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
    output_dir="finetuned/09_Unsloth_DEMO",
    bf16=True,
    max_length=710,
    # assistant_only_loss=True,
    # chat_template_path="./chat_template.jinja"
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

from unsloth.chat_templates import train_on_responses_only

# 优化四：调用train_on_reponses_only，指导Unsloth，labels当中，哪一部分token需要计算损失，从response_part最后开始，到instruction_part开头，这一部分，会计算损失
trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n"
)

trainer.train()

# save_pretrained_merged，原生的transformers的model是没有，这个方式是Unsloth的FastLanguageModel所带的方法
# 优化点五：unsloth的model提供了save_pretrained_merged方法，通过调用该方法，就能够对模型的权重进行合并，将适配器权重和基座模型的权重合并到一起去
quantized_peft_model.save_pretrained_merged("./finetuned/Qwen3-8B-SFT-unsloth-merged", tokenizer, save_method="merged_16bit")