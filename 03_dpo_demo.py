"""
SFT Demo 脚本
"""
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("finetuned/02_SFT_TRAIN/")



from datasets import load_dataset
from typing import Dict
def get_train_data(dpo_config):
    """
    加载训练数据
    """
    dataset = load_dataset("data/ultrafeedback_binarized")
    train_dataset = dataset["train_sft"]
    # 遍历其中的数据，对每条数据，使用tokenizer.apply_chat_template进行编码
    chosen_result_list = []
    rejected_result_list = []
    train_data_size = dpo_config.train_data_size

    for i in range(train_data_size):
        chosen_message_list:list[Dict] = train_dataset[i]["chosen"]
        chosen_message_list.insert(0,{"role":"system","content":"You are a helpful assistant."})
        # result_dict: input_ids和attention_mask
        result_dict:dict = tokenizer.apply_chat_template(chosen_message_list,)
        chosen_result_list.append(result_dict)

        rejected_message_list:list[Dict] = train_dataset[i]["rejected"]
        rejected_message_list.insert(0,{"role":"system","content":"You are a helpful assistant."})
        # result_dict: input_ids和attention_mask
        result_dict:dict = tokenizer.apply_chat_template(rejected_message_list,)
        rejected_result_list.append(result_dict)



    return chosen_result_list,rejected_result_list

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

# DPO: 计算log_probs
def _compute_log_probs(output_logits, labels, assistant_mask):
    """
    通过logtis和labels计算，输出labels当中回答的log_probs
    output_logits: shape:[batch_size, seq_len, vocab_size]
    labels: shape:[batch_size, seq_len]
    assistant_mask: shape:[batch_size, seq_len]
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


    # 哈达玛积，对应位置相乘，需要算损失的token，乘以1，不需要算损失的token，乘以0，
    masked_log_probs = result * assistant_mask

    # 将所有的对数概率，沿着seq_len的维度，相加，
    final_log_probs = masked_log_probs.sum(dim =-1) / assistant_mask.sum(dim = -1)

    return final_log_probs


# DPO,损失计算函数，具体逻辑：
def compute_loss(chosen_log_probs,rejected_log_probs, reference_chosen_log_probs, reference_rejected_log_probs,beta):
    """
    chosen_log_probs: 当前正在训练的模型，输出喜欢回答的对数概率, shape:[batch_size] chose_log_probs[0]；第0个样本当中，模型输出喜好回答的概率
    rejected_log_probs: 当前正在训练的模型，输出拒绝回答的对数概率，shape:[batch_size]

    reference_chosen_log_probs: 参考模型，输出喜欢回答的对数概率，shape:[batch_size]
    reference_rejected_log_probs： 参考模型，输出拒绝回答的对数概率，shape:[batch_size]
    """

    # 1、计算margin和单条样本的loss
    # margin:[batch_size]
    margin =  (chosen_log_probs - rejected_log_probs) - (reference_chosen_log_probs - reference_rejected_log_probs)
    # loss: [batch_size]
    loss =  -torch.nn.functional.logsigmoid(beta * margin) 

    # 2、计算当前batch的平均损失
    # 求多条样本的loss加和，再除以，当前batch的batch_size
    average_loss = loss.mean()

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
class DPOConfig:

    batch_size:int =4 
    lr: float = 1e-6
    warmup_ratio:float = 0.1
    log_dir:str = "logs/03_DPO_TRAIN"
    # 打印日志的间隔
    log_batch:int = 100
    save_dir:str = "finetuned/03_DPO_TRAIN"
    train_data_size:int = 2000

    beta:float = 0.2


def train(dpo_config:DPOConfig):
    # 1、获取模型，获取两个模型，一个是训练模型，将模型置为train模式，放到cuda，第二个是参考模型
    model = AutoModelForCausalLM.from_pretrained("finetuned/02_SFT_TRAIN/")
    model.to("cuda")
    model.train()
    reference_model = AutoModelForCausalLM.from_pretrained("finetuned/02_SFT_TRAIN/")
    reference_model.to("cuda")
    reference_model.eval()

    # 2、获取训练数据
    chosen_train_data,rejected_train_data = get_train_data(dpo_config) 
    # 13条，4,13+3=16 // 4 4 
    total_batch = (len(chosen_train_data) + dpo_config.batch_size -1) // dpo_config.batch_size
    
    # 3、构造优化器
    optimizer = AdamW(model.parameters(),lr=dpo_config.lr)

    # 5、添加日志
    # 5.1 使用tensorboard
    writer = SummaryWriter(log_dir=dpo_config.log_dir)
    # 5.2 使用tqdm
    progress_bar = tqdm(total=total_batch)

    losses_list = []

    for batch in range(total_batch):

        # 1、张量准备
        # 1.1 获取当前批次数据

        chosen_batch_train_data:list[dict] = chosen_train_data[batch * dpo_config.batch_size : (batch+1) * dpo_config.batch_size]
        rejected_batch_train_data:list[dict] = rejected_train_data[batch * dpo_config.batch_size : (batch+1) * dpo_config.batch_size]
        # 1.2 对数据进行padding

        chosen_batch_data_max_len = max([len(seq["input_ids"]) for seq in chosen_batch_train_data])
        chosen_padded_seq_result = []
        for single_seq in chosen_batch_train_data:
            padding_lenth = chosen_batch_data_max_len - len(single_seq["input_ids"])
            # 在input_ids的最后，填充padded
            single_seq["input_ids"].extend([tokenizer.pad_token_id] *padding_lenth )
            chosen_padded_seq_result.append(single_seq["input_ids"])

        rejected_batch_data_max_len = max([len(seq["input_ids"]) for seq in rejected_batch_train_data])
        rejected_padded_seq_result = []
        for single_seq in rejected_batch_train_data:
            padding_lenth = rejected_batch_data_max_len - len(single_seq["input_ids"])
            # 在input_ids的最后，填充padded
            single_seq["input_ids"].extend([tokenizer.pad_token_id] *padding_lenth )
            rejected_padded_seq_result.append(single_seq["input_ids"])
        

        # 1.3 构造张量
        chosen_data_tensor = torch.tensor(chosen_padded_seq_result, dtype=torch.long).to("cuda")
        rejected_data_tensor = torch.tensor(rejected_padded_seq_result, dtype=torch.long).to("cuda")

        chosen_input_ids = chosen_data_tensor[:,:-1]
        chosen_labels = chosen_data_tensor[:,1:]

        rejected_input_ids = rejected_data_tensor[:,:-1]
        rejected_labels = rejected_data_tensor[:,1:]
        chosen_assistant_mask = create_answer_mask(input_ids=chosen_input_ids,tokenizer=tokenizer)
        rejected_assistant_mask = create_answer_mask(input_ids=rejected_input_ids,tokenizer=tokenizer)
        # 2、模型前向传播
        # 2.1 训练模型两次
        chosen_output_logits = model(chosen_input_ids).logits
        rejected_output_logits = model(rejected_input_ids).logits

        with torch.no_grad():
            reference_chosen_output_logits = reference_model(chosen_input_ids).logits
            reference_rejected_output_logits = reference_model(rejected_input_ids).logits
        

            
        # 3、损失计算
        chosen_log_probs = _compute_log_probs(chosen_output_logits,chosen_labels,chosen_assistant_mask)
        rejected_log_probs = _compute_log_probs(rejected_output_logits,rejected_labels,rejected_assistant_mask)

        reference_chosen_log_probs = _compute_log_probs(reference_chosen_output_logits,chosen_labels,chosen_assistant_mask)
        reference_rejected_log_probs = _compute_log_probs(reference_rejected_output_logits,rejected_labels,rejected_assistant_mask)
        loss = compute_loss(chosen_log_probs=chosen_log_probs,
                            rejected_log_probs=rejected_log_probs,
                            reference_chosen_log_probs=reference_chosen_log_probs, 
                            reference_rejected_log_probs=reference_rejected_log_probs,beta=dpo_config.beta)
        # 将loss进行记录，后续用以打印 每 n 步的平均损失
        losses_list.append(loss.item())
        # 更新tqdm的进度条
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"current_loss:{loss.item():.4f}")


        # 4、反向传播
        loss.backward()
        # 5、参数更新
        # 5.1 更新一下学习率
        optimizer.param_groups[0]["lr"] = cosine_scheduler_with_warmup(total_batch,dpo_config.warmup_ratio,dpo_config.lr,batch)
        # 5.2 更新参数
        optimizer.step()
        # 5.3 梯度清空
        optimizer.zero_grad()

        should_log = batch % dpo_config.log_batch == 0 or batch == (total_batch-1)
        if should_log:
            # 计算前面log_batch步的平均损失，然后写在tensorboard当中
            log_batch_loss = losses_list[-dpo_config.log_batch:]
            average_loss = sum(log_batch_loss) / len(log_batch_loss)
            writer.add_scalar("train_loss",scalar_value=average_loss, global_step=batch)
    
    return model,tokenizer


def save_model_tokenizer(model,tokenizer,dpo_config:DPOConfig):
    """
    将训练之后的模型和tokenzier进行保存
    """
    model.save_pretrained(dpo_config.save_dir)
    tokenizer.save_pretrained(dpo_config.save_dir)
    print("当前模型和tokenizer保存完毕")


if __name__ == "__main__":
    dpo_config = DPOConfig()
    model, tokenizer = train(dpo_config=dpo_config)
    save_model_tokenizer(model,tokenizer,dpo_config)
