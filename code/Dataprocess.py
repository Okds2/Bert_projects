import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import wget
import os
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
# 读取json文件：
import json
import jsonpath
from sklearn.utils import shuffle
# 问号？会不会产生很大影响？

# 数据类型一：涉及的问答都是与法律法规相关的
# 数据文件1，这里面全部涉及的都是与法律有关的问题。训练数据_带法律依据_92k.json。利用ChatGPT联想生成具体的情景问答，从而使得生成的数据集有具体的法律依据。
obj1 = json.load(open('./data_json/训练数据_带法律依据_92k.json', 'r', encoding='utf-8'))  # 注意，这里是文件的形式，不能直接放一个文件名的字符串
len(obj1)
question_list1 = []
for i in range(len(obj1)):
    cur_sentence = obj1[i]['question'].replace('\n', '')
    question_list1.append(cur_sentence)
label_list1 = [1 for i in range(len(question_list1))]
dic_1 = {"label": label_list1, "sentence": question_list1}
data_1 = pd.DataFrame(dic_1)
# 查看了最大长度。
max(data_1['sentence'].apply(len))

# 数据文件2：legal_counsel_multi_turn_with_article_v2.json，这里面确确实实会涉及到法律。ChatGPT基于法条生成的多轮法律咨询对话
obj2 = json.load(open('./data_json/legal_counsel_multi_turn_with_article_v2.json', 'r', encoding='utf-8'))  # 注意，这里是文件的形式，不能直接放一个文件名的字符串
len(obj2)
question_list2 = []
for i in range(len(obj2)):
    cur_sentence = obj2[i]['query'].replace('\n', '')
    question_list2.append(cur_sentence)
label_list2 = [1 for i in range(len(question_list2))]
dic_2 = {"label": label_list2, "sentence": question_list2}
data_2 = pd.DataFrame(dic_2)
# 查看了最大长度。
max(data_2['sentence'].apply(len))

# 数据文件3：legal_counsel_with_article_v2.json。 ChatGPT基于法条生成的法律咨询回复
obj3 = json.load(open('./data_json/legal_counsel_with_article_v2.json', 'r', encoding='utf-8'))  # 注意，这里是文件的形式，不能直接放一个文件名的字符串
len(obj3)
question_list3 = []
for i in range(len(obj3)):
    cur_sentence = obj3[i]['query'].replace('\n', '')
    question_list3.append(cur_sentence)
label_list3 = [1 for i in range(len(question_list3))]
dic_3 = {"label": label_list3, "sentence": question_list3}
data_3 = pd.DataFrame(dic_3)
# 查看了最大长度。
max(data_3['sentence'].apply(len))

law_data = pd.concat([data_1, data_2, data_3], axis = 0)
law_data.index = range(len(law_data))  # 重新更改数据的index
# 将law_data进行保存
law_data.to_excel("law_data.xlsx")


law_data['sentence'].apply(len).max()
law_data[law_data['sentence'].apply(len) == 436]

sentences = law_data.sentence.values
labels = law_data.label.values
tokenizer = BertTokenizer.from_pretrained('./model/bert-base-chinese', do_lower_case=True)  # 这里的分词器采用的是基于中文模型的

# 输出原始句子
print(' Original: ', sentences[36177])

# 将分词后的内容输出
print('Tokenized: ', tokenizer.tokenize(sentences[36177]))

# 将每个词映射到词典下标
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[36177])))

# 数据类型二：涉及的问答都是与法律法规无关
# 这一部分通过调用科大讯飞大模型的API进行生成
# 可以进行BERT模型预训练的数据，各种法律法规数据


Pretraining_data = []
obj4 = json.load(open('./data_json/训练数据_带法律依据_92k.json', 'r', encoding='utf-8'))
len(obj4[0]['reference'])

import re
for i in range(len(obj4)):
    for j in range(len(obj4[i]['reference'])):
        result = re.findall(".*\"(.*)\".*", obj4[i]['reference'][j])
        for x in result:
            cur_law = x.replace('\\n', '')
            cur_law = cur_law.replace('\n', '')
            cur_law = cur_law.replace('\\n\ ', '')
            cur_law = cur_law.replace('\ ', '')
            Pretraining_data.append(cur_law)
Pretraining_data = pd.DataFrame(Pretraining_data)
# 转成excel表格，已转：
Pretraining_data.to_excel("Pretraining_data.xlsx")

# 非法律法规数据的预处理——科大讯飞星火大模型：
file1 = open('nolegal_data.txt', 'r', encoding='utf-8')  # 打开要去掉空行的文件
file2 = open('data2.txt', 'w', encoding='utf-8')  # 生成没有空行的文件

for line in file1.readlines():
    if line == '\n':
        line = line.strip('\n')
    file2.write(line)

file1.close()
file2.close()
datas = pd.read_csv('C:/Users/12249/PycharmProjects/pythonProject1/data2.txt',header=None)
datas.columns = ['sentence']
for i in range(len(datas)):
    if '：' in datas.iloc[i][0]:
        index1 = datas.iloc[i][0].find("：")
        index2 = datas.iloc[i][0].find("？")
        datas.iloc[i][0] = datas.iloc[i][0][index1 + 1: index2 + 1]
    else:
        index1 = datas.iloc[i][0].find(".")
        index2 = datas.iloc[i][0].find("？")
        datas.iloc[i][0] = datas.iloc[i][0][index1 + 1: index2 + 1]

# 删除空行
datas.drop(datas[datas.loc[ : , 'sentence'] == ''].index, inplace=True)
# 删除重复行
datas.drop(datas[datas['sentence'].duplicated() == True].index, inplace=True)

# 非法律法规数据的预处理——科大讯飞星火大模型：
datass = pd.read_csv('C:/Users/12249/PycharmProjects/pythonProject1/nolegal_data1.txt',header=None) # 读取txt
for i in range(len(datass)):
    index = datass.iloc[i][0].find(".")
    if index == -1:
        datass.iloc[i][0] = ''
    else:
        datass.iloc[i][0] = datass.iloc[i][0][index + 1: ]
    if '#### 关闭会话' in datass.iloc[i][0]:
        datass.iloc[i][0] = datass.iloc[i][0].strip('#### 关闭会话')

datass.columns = ['sentence']
# 删除空行
datass.drop(datass[datass.loc[ : , 'sentence'] == ''].index, inplace=True)
# 删除重复行
datass.drop(datass[datass['sentence'].duplicated() == True].index, inplace=True)
datass.index = range(len(datass))

# 将数据合并
nolegal_data = pd.concat([datas, datass], axis=0)
# 删除重复行
nolegal_data.drop(nolegal_data[nolegal_data['sentence'].duplicated() == True].index, inplace=True)
nolegal_data.index = range(len(nolegal_data))
nolegal_data['label'] = [0 for i in range(len(nolegal_data))]
nolegal_data['label'], nolegal_data['sentence'] = nolegal_data['sentence'], nolegal_data['label']
nolegal_data.columns = ['label', 'sentence']
nolegal_data['sentence'] = nolegal_data['sentence'].str.strip()
nolegal_data['label'] = [0 for i in range(len(nolegal_data))]
nolegal_data.to_excel('./nolegal_data.xlsx', index = False)

legal_data = pd.read_excel('./legal_data.xlsx')
nolegal_data = pd.read_excel('./nolegal_data.xlsx')
legal_data['label'] = [1 for i in range(len(legal_data))]
legal_data.to_excel('./legal_data.xlsx', index = False)


from sklearn.model_selection import train_test_split
nolegal_sentence_train, nolegal_sentence_test, nolegal_label_train, nolegal_label_test = train_test_split(nolegal_data['sentence'], nolegal_data['label'], test_size=0.25)
legal_data = law_data.sample(50000)
legal_sentence_train, legal_sentence_test, legal_label_train, legal_label_test = train_test_split(legal_data['sentence'], legal_data['label'], test_size=0.25)

train_data_sentence = pd.concat([legal_sentence_train, nolegal_sentence_train], axis=0)
train_data_label = pd.concat([legal_label_train, nolegal_label_train], axis=0)

test_data_sentence = pd.concat([legal_sentence_test, nolegal_sentence_test], axis=0)
test_data_label = pd.concat([legal_label_test, nolegal_label_test], axis=0)

train_data = pd.concat([train_data_sentence, train_data_label], axis=1)
test_data = pd.concat([test_data_sentence, test_data_label], axis=1)

train_data.to_excel('./train_data.xlsx', index = False)
test_data.to_excel('./test_data.xlsx', index = False)

# 最终的数据
Data = pd.concat([law_data.sample(30000), nolegal_data], axis=0)
Data['sentence'] = Data['sentence'].str.strip()
# 将数据打乱
Data = shuffle(Data)
# 重新编写index
Data.index = range(len(Data))

sentences = Data.sentence.values
labels = Data.label.values



# 具有隐藏意图的数据处理
nohidden_data = pd.concat([legal_data, nolegal_data], axis=0)
nohidden_data.index = range(len(nohidden_data ))
nohidden_data['label'] = [0 for i in range(len(nohidden_data))]
nohidden_data.to_excel('./nohidden_data.xlsx', index = False)

hidden_data = pd.read_csv('C:/Users/12249/PycharmProjects/pythonProject1/hidden_meandata.txt', header=None)
for i in range(len(hidden_data)):
    if '隐藏意图' in hidden_data.iloc[i][0]:
        hidden_data.iloc[i][0] = ''
    else:
        index = hidden_data.iloc[i][0].find(".")
        if index == -1:
            hidden_data.iloc[i][0] = ''
        else:
            hidden_data.iloc[i][0] = hidden_data.iloc[i][0][index + 1:]
    # if '#### 关闭会话' in hidden_data.iloc[i][0]:
    #     hidden_data.iloc[i][0] = hidden_data.iloc[i][0].strip('#### 关闭会话')
hidden_data.columns = ['sentence']
hidden_data.drop(hidden_data[hidden_data.loc[ : , 'sentence'] == ''].index, inplace=True)
# 删除重复行
hidden_data.drop(hidden_data[hidden_data['sentence'].duplicated() == True].index, inplace=True)
hidden_data.index = range(len(hidden_data))
hidden_data['label'] = [1 for i in range(len(hidden_data))]
hidden_data.to_excel('./hidden_data.xlsx', index = False)

hidden_data = pd.read_excel('./hidden_Data.xlsx')
hidden_data['label'] = [1 for i in range(len(hidden_data))]
hidden_data['label'], hidden_data['sentence'] = hidden_data['sentence'], hidden_data['label']
hidden_data.columns = ['label', 'sentence']
nohidden_data = pd.read_excel('./nohidden_data.xlsx')
from sklearn.model_selection import train_test_split
# 按照6:2:2的结构划分
hidden_sentence_train, hidden_sentence_test, hidden_label_train, hidden_label_test = train_test_split(hidden_data['sentence'], hidden_data['label'], test_size=0.2)
nohidden_data = nohidden_data.sample(50000)
nohidden_sentence_train, nohidden_sentence_test, nohidden_label_train, nohidden_label_test = train_test_split(nohidden_data['sentence'], nohidden_data['label'], test_size=0.2)

train1_data_sentence = pd.concat([hidden_sentence_train, nohidden_sentence_train], axis=0)
train1_data_label = pd.concat([hidden_label_train, nohidden_label_train], axis=0)

test1_data_sentence = pd.concat([hidden_sentence_test, nohidden_sentence_test], axis=0)
test1_data_label = pd.concat([hidden_label_test, nohidden_label_test], axis=0)

train1_data = pd.concat([train1_data_sentence, train1_data_label], axis=1)
test1_data = pd.concat([test1_data_sentence, test1_data_label], axis=1)

train1_data.to_excel('./hidden_traindata.xlsx', index = False)
test1_data.to_excel('./hidden_testdata.xlsx', index = False)