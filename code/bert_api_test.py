import torch
from transformers import BertTokenizer, BertModel
texts = [
    '我想咨询一下这个是否犯法',
    '根据中华人民共和国民法典']


# 加载bert分词器 (tokenizer)
# 这里可以选择不同的bert预训练模型
bert_tokenizer = BertTokenizer.from_pretrained('./model/bert-base-chinese')

# 获得bert的输入，input_ids和att_mask
# input_ids存放id形式的文本，att_mask非pad部分为1，否则为0
batch_input_ids, batch_att_mask = [], []
for text in texts:
    encoded_dict = bert_tokenizer.encode_plus(text, max_length=512, padding='max_length', return_length='pt', truncation=True)
    batch_input_ids.append(encoded_dict['input_ids'])
    batch_att_mask.append(encoded_dict['attention_mask'])


batch_input_ids = torch.tensor(batch_input_ids)
batch_att_mask = torch.tensor(batch_att_mask)

# 加载bert模型
bert_model = BertModel.from_pretrained('./model/bert-base-chinese')

# 推理, 查看bert输出
with (torch.no_grad()):
    last_hidden_state = bert_model(input_ids=batch_input_ids, attention_mask = batch_att_mask)[0]
    pooled_output = bert_model(input_ids=batch_input_ids, attention_mask = batch_att_mask)[1]
    print('last_hidden_state', last_hidden_state, last_hidden_state.shape) # 可以接标注
    print('\n')
    print('pooled_output', pooled_output, pooled_output.shape)   # 可以接分类
