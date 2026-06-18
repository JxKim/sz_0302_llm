"""
SFT Demo 脚本
"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B-Base/")



from datasets import load_dataset
from typing import Dict
def get_train_data(sft_config):
    """
    加载训练数据
    """
    dataset = load_dataset("data/ultrachat_200k")
    train_dataset = dataset["train_sft"]
    # 遍历其中的数据，对每条数据，使用tokenizer.apply_chat_template进行编码
    result_list = []
    train_data_size = sft_config.train_data_size

    for i in range(train_data_size):
        message_list:list[Dict] = train_dataset[i]["messages"]
        message_list.insert(0,{"role":"system","content":"You are a helpful assistant."})
        # result_dict: input_ids和attention_mask
        result_dict:dict = tokenizer.apply_chat_template(message_list,)
        result_list.append(result_dict)

    return result_list

from transformers import PreTrainedTokenizerFast
import torch
from typing import List

def create_answer_mask(input_ids,tokenizer:PreTrainedTokenizerFast):
    """
    创建answer mask，从input_ids当中找出assistant回答的部分，然后输出一个与input_ids相同shape的mask，
    后续将其与pad_mask进行逻辑与操作，得到最终的mask，用以计算损失
    """

    
    # 构建answer mask，输入的input_ids为批量 tokenize之后的数据，对于每一条数据，查找当中assistant回答的部分，将其设置为1

    # 1. 构造一个和input_ids相同shape的全0矩阵
    answer_mask = torch.zeros_like(input_ids)

    # 2. 遍历input_ids中的每一条数据，查找assistant回答的部分，将其设置为1
    eos_token_id = tokenizer.encode('<|im_end|>')[0]
    for idx,ids in enumerate(input_ids):
        # 获取到所有的eos_position
        eos_position:List = torch.where(ids == eos_token_id)[0].tolist()

        # 排除第一个eos_position: 第一个对应的是system prompt
        eos_position = eos_position[1:]
        # 解析获得user_ends和assistant_ends
        user_ends,assistant_ends = _parse_conversation_turns(eos_position)
        # 设置answer mask
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)   
    
    # 结果返回:
    return answer_mask

def _parse_conversation_turns(eos_positions:List[int]):
    """
    输入eos_positions，输出user所对应的end位置和assistant所对应的end位置。

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么输入的eos_token_id为：[10,15,20,25]
    user_turns为从第一个开始取（具体索引位置需要加一，因为eos_token_id后面还有一个\n换行符），每隔一个取一次，assistant_turns为从第二个开始取，每隔一个取一次。

    输出结果为：
        user_turns:[11,21]
        assistant_ends:[16,26]
    """

    use_ends = [pos+1 for pos in eos_positions[::2]]
    assistant_ends = [pos+1 for pos in eos_positions[1::2]]

    return use_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    """
    将mask当中，assistant回答的部分，设置为1（原地修改，不返回新的mask），其余部分保持为0

    以下面的对话为例：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    什么是习惯？<|im_end|>
    <|im_start|>assistant
    习惯是指在一定时间内重复执行的行为。<|im_end|>
    <|im_start|>user
    如何培养一个习惯<|im_end|>
    <|im_start|>assistant
    21天培养法，每天坚持xxx<|im_end|>

    假设第一个eos_token_id index为5，第二个为10，第三个为15，第四个为20，第五个为25，
    那么user_turns:[11,21]，assistant_ends:[16,26]

    user_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    assistant_ends当中的索引指向的是<|im_end|>之后的\n的索引，
    要想获取到assistant的回答的起始位置，就需要再跳过\n,<|im_start|>,assistant 这三个token，所以需要加3.
    要想获取到assistant的回答的结束位置，就需要往前跳一个<|im_end|>，所以需要减1.
    """
    num_user_turns = len(user_ends)
    num_assistant_turns = len(assistant_ends)
    if num_user_turns == num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1

    elif num_user_turns == num_assistant_turns + 1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start = user_end + 3
            answer_end = assistant_end - 1
            mask[answer_start:answer_end] = 1
        
        # 处理最后一轮被截断的助手回答
        last_user_end = user_ends[-1] 
        last_answer_start = last_user_end + 3
        mask[last_answer_start:] = 1

# SFT,损失计算函数，具体逻辑：
def compute_loss(output_logits, labels, assistant_answer_mask):
    """
    output_logits: 模型前向传播之后，输出的结果 , shape batch_size, seq_len, vocab_size
    labels: 真实的标签 batch_size, seq_len
    assistant_answer_mask: 模型回答掩码
    """

    # 1、需要对output_logits进行log softmax，得到对数概率
    # log_probs: batch_size, seq_len, vocab_size
    log_probs = torch.log_softmax(output_logits, dim=-1)

    # 2、找到模型输出答案所对应的token的概率是多少
    result = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    )

    result = result.squeeze(-1) 

    # negative_log_probs.shape [batch_size,seq_len]
    negative_log_probs = result * (-1)

    # 哈达玛积，对应位置相乘，需要算损失的token，乘以1，不需要算损失的token，乘以0，
    masked_negative_log_probs = negative_log_probs * assistant_answer_mask

    # 将序列当中所有token的对数概率加起来，除以总的token数量
    average_loss  = masked_negative_log_probs.sum() / assistant_answer_mask.sum()

    return average_loss

import numpy as np
def cosine_scheduler_with_warmup(total_batch, warmup_ratio,lr,current_batch):
    """
    带预热的余弦衰减调度器
    """
    warmup_batch = total_batch * warmup_ratio
    if current_batch < warmup_batch:
        return current_batch * lr / warmup_batch
    else:
        progress = (current_batch - warmup_batch) / (total_batch - warmup_batch)
        # decay: 0.5 * (1 + cos(π*progress))  decay会从1，衰减成0
        decay =  0.5 * (1 + np.cos(np.pi*progress)) 
        current_lr = lr * decay
        return current_lr
    


from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
@dataclass
class SFTConfig:

    batch_size:int =4 
    lr: float = 2e-5
    warmup_ratio:float = 0.1
    log_dir:str = "logs/02_SFT_TRAIN"
    # 打印日志的间隔
    log_batch:int = 100
    save_dir:str = "finetuned/02_SFT_TRAIN"
    train_data_size:int = 2000


def train(sft_confg:SFTConfig):
    # 1、获取模型，将模型置为train模式，放到cuda
    model = AutoModelForCausalLM.from_pretrained("model/Qwen3-0.6B-Base/")
    model.to("cuda")
    model.train()

    # 2、获取训练数据
    train_data_list:list = get_train_data(sft_confg) 
    # 13条，4,13+3=16 // 4 4 
    total_batch = (len(train_data_list) + sft_confg.batch_size -1) // sft_confg.batch_size
    
    # 3、构造优化器
    optimizer = AdamW(model.parameters(),lr=sft_confg.lr)

    # 5、添加日志
    # 5.1 使用tensorboard
    writer = SummaryWriter(log_dir=sft_confg.log_dir)
    # 5.2 使用tqdm
    progress_bar = tqdm(total=total_batch)

    losses_list = []

    for batch in range(total_batch):

        # 1、张量准备
        # 1.1 获取当前批次数据

        batch_train_data:list[dict] = train_data_list[batch * sft_confg.batch_size : (batch+1) * sft_confg.batch_size]
        # 1.2 对数据进行padding
        batch_data_max_len = max([len(seq["input_ids"]) for seq in batch_train_data])
        padded_seq_result = []
        for single_seq in batch_train_data:
            padding_lenth = batch_data_max_len - len(single_seq["input_ids"])
            # 在input_ids的最后，填充padded
            single_seq["input_ids"].extend([tokenizer.pad_token_id] *padding_lenth )
            padded_seq_result.append(single_seq["input_ids"])

        # 1.3 构造张量
        data_tensor = torch.tensor(padded_seq_result, dtype=torch.long).to("cuda")
        input_ids = data_tensor[:,:-1]
        labels = data_tensor[:,1:]
        assistant_mask = create_answer_mask(input_ids=input_ids,tokenizer=tokenizer)
        # 2、模型前向传播
        output_logits = model(input_ids).logits
        # 3、损失计算
        loss = compute_loss(output_logits=output_logits,labels=labels,assistant_answer_mask=assistant_mask)
        # 将loss进行记录，后续用以打印 每 n 步的平均损失
        losses_list.append(loss.item())
        # 更新tqdm的进度条
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"current_loss:{loss.item():.4f}")


        # 4、反向传播
        loss.backward()
        # 5、参数更新
        # 5.1 更新一下学习率
        optimizer.param_groups[0]["lr"] = cosine_scheduler_with_warmup(total_batch,sft_confg.warmup_ratio,sft_confg.lr,batch)
        # 5.2 更新参数
        optimizer.step()
        # 5.3 梯度清空
        optimizer.zero_grad()

        should_log = batch % sft_confg.log_batch == 0 or batch == (total_batch-1)
        if should_log:
            # 计算前面log_batch步的平均损失，然后写在tensorboard当中
            log_batch_loss = losses_list[-sft_confg.log_batch:]
            average_loss = sum(log_batch_loss) / len(log_batch_loss)
            writer.add_scalar("train_loss",scalar_value=average_loss, global_step=batch)
    
    return model,tokenizer


def save_model_tokenizer(model,tokenizer,sft_config:SFTConfig):
    """
    将训练之后的模型和tokenzier进行保存
    """
    model.save_pretrained(sft_config.save_dir)
    tokenizer.save_pretrained(sft_config.save_dir)
    print("当前模型和tokenizer保存完毕")


if __name__ == "__main__":
    sft_config = SFTConfig()
    model, tokenizer = train(sft_confg=sft_config)
    save_model_tokenizer(model,tokenizer,sft_config)
