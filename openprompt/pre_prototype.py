import json
from transformers import BertTokenizer
import torch


# todo 先从jsonl文件里面读出来dict，然后使用tokenizer将其转化为token，之后传给trainer，在train_epoch开始的时候，使用plm encode。
## encode的时候，需要考虑使用整句的隐藏向量的平均还是cls的隐藏向量，又或者是也设计个mask token。
def read_single_dict_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error parsing file: {e}")
            print(f"Invalid JSON: {content}")
            data = None
    return data


def encode_text_with_bert(text):
    # 加载预训练的 BERT tokenizer
    # path = '/Work18/2020/weixiao/code/model/bert/bert_base_uncased'
    path = 'bert-base-uncased'
    # path = "/Work18/2020/weixiao/code/model/roberta/roberta_base"

    tokenizer = BertTokenizer.from_pretrained(path)


    # 使用 tokenizer 对文本进行编码
    encoded_input = tokenizer.batch_encode_plus(text, add_special_tokens=True,  # 添加 [CLS] 和 [SEP] 标记
                                          max_length=128,  # 设置最大长度
                                          padding='longest',  # 填充到最大长度
                                          truncation=True,  # 如果超过最大长度，则进行截断
                                          return_attention_mask=True,  # 返回 attention mask
                                          return_token_type_ids=False,  # 不返回 token type ids
                                          return_tensors='pt'  # 返回 PyTorch tensors
                                          )

    return encoded_input


def get_prototype_token(verb_path):
    # 使用示例
    # print("verbalizer path",verb_path[-9:])
    # todo 这里目前的设置是verbalizer跟prototype设置的文本保持一致，后观察实验可以改成不一致的
    # verb_path = 'scripts/TextClassification/GID/GID_SD_verbalizer_eg1.jsonl'
    # verb_path = 'scripts/TextClassification/GID/GID_SD_verbalizer_eg2.jsonl'
    # verb_path = 'scripts/TextClassification/GID/GID_SD_verbalizer_eg3.jsonl'
    # todo 目前测试效果，对于sd—40来说，verb设置5个eg，proto设置4个eg效果最好。
    # verb_path = 'scripts/TextClassification/GID/GID_SD_40_verbalizer_4.jsonl'
    # verb_path = 'scripts/TextClassification/GID/GID_SD_verbalizer_eg5.jsonl'
    # print("prototype path",verb_path[-9:])
    # verb_path = 'scripts/TextClassification/GID/GID_SD_verbalizer_paraphrase.jsonl'
    prototype_dict = read_single_dict_file(verb_path)
    text = []
    for k, [v] in prototype_dict.items():
        text.append(v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prototype_token = encode_text_with_bert(text).to(device)
    return prototype_token
    # print(prototype_token['input_ids'].shape)
# exit()
