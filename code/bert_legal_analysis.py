import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import wget
import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
# 读取json文件：
import json
import jsonpath
from sklearn.utils import shuffle
import time
import datetime
import random

train_data = pd.read_excel("./train_data.xlsx")
train_data = shuffle(train_data) # 将顺序打乱
train_data.index = range(len(train_data))
# 输入数据格式：一列为label，一列为sentence。
# 分词器
def Tokenizer(data, bert_tokenizer_path, max_length):  # 这里的data为上述所示的格式
    # 输入数据说明：data为数据集，bert_tokenizer_path为分词模型路径，max_length为最大截断或填充长度。
    sentences = data.sentence.values
    labels = data.label.values
    tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path, do_lower_case=True)
    max_len = 0
    # 添加[CLS]和[SEP]符号
    for sent in sentences:
        # 将文本分词，并添加 `[CLS]` 和 `[SEP]` 符号
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))
    # print('Max sentence length: ', max_len)
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # 输入文本
            add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
            max_length=max_length,  # 填充 & 截断长度
            pad_to_max_length=True,
            return_attention_mask=True,  # 返回 attn. masks.
            return_tensors='pt',  # 返回 pytorch tensors 格式的数据
        )
        # 将编码后的文本加入到列表
        input_ids.append(encoded_dict['input_ids'])
        # 将文本的 attention mask 也加入到 attention_masks 列表
        attention_masks.append(encoded_dict['attention_mask'])
    # 将列表转换为 tensor
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

# max_length = max(train_data['sentence'].apply(len)) + 2
max_length = 100
input_ids, attention_masks, labels = Tokenizer(train_data, './model/bert-base-chinese', max_length)

# 将输入数据合并为 TensorDataset 对象，可以进行batch_size的
def Dataloader(input_ids, attention_masks, labels, proportion, batch_size):
    # proportion为训练集占比
    dataset = TensorDataset(input_ids, attention_masks, labels)
    # 计算训练集和验证集大小
    train_size = int(proportion * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    #   为训练和验证集创建 Dataloader，对训练样本随机洗牌
    train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            sampler = RandomSampler(train_dataset), # 随机小批量
            batch_size = batch_size # 以小批量进行训练
        )
    # 验证集不需要随机化，这里顺序读取就好
    validation_dataloader = DataLoader(
            val_dataset, # 验证样本
            sampler = SequentialSampler(val_dataset), # 顺序选取小批量
            batch_size = batch_size
        )
    return train_dataloader, validation_dataloader

batch_size = 32
proportion = 0.8
train_dataloader, validation_dataloader = Dataloader(input_ids, attention_masks, labels, proportion, batch_size)

# 模型:
def model(model_path):
    model = BertForSequenceClassification.from_pretrained(
        model_path, # 小写的 12 层预训练模型
        num_labels = 2, # 分类数 --2 表示二分类
                        # 你可以改变这个数字，用于多分类任务
        output_attentions = False, # 模型是否返回 attentions weights.
        output_hidden_states = False,
        return_dict=False# 模型是否返回所有隐层状态.
    )
    # 在 gpu 中运行该模型
    model.cuda()
    return model  # 这里要注意是不是return后还要将model cuda()

model_path = './model/bert-base-chinese'
model = model(model_path)

# 优化器
def optimizer(model, l_r, ep_s):
    optimizer = AdamW(model.parameters(),
                      lr = l_r, # args.learning_rate - default is 5e-5
                      eps = ep_s # args.adam_epsilon  - default is 1e-8
                    )
    return optimizer
lr = 2e-5
eps = 1e-8
optimizer = optimizer(model, lr, eps)

# 准确率
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 计时器
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))

    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 训练函数：
def train(seed_val, epochs, model, train_dataloader, validation_dataloader, optimizer):
    # 设定随机种子值，以确保输出是确定的
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # 存储训练和评估的 loss、准确率、训练时长等统计指标,
    training_stats = []
    # 统计整个训练时长
    total_t0 = time.time()
    # 创建学习率调度器
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    for epoch_i in range(0, epochs):
        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # 统计单次 epoch 的训练时间
        t0 = time.time()
        # 重置每次 epoch 的训练总 loss
        total_train_loss = 0
        # 将模型设置为训练模式。这里并不是调用训练接口的意思
        # dropout、batchnorm 层在训练和测试模式下的表现是不同的 (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        # 训练集小批量迭代
        for step, batch in enumerate(train_dataloader):
            # 每经过40次迭代，就输出进度信息
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # 准备输入数据，并将其拷贝到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # 每次计算梯度前，都需要将梯度清 0，因为 pytorch 的梯度是累加的
            model.zero_grad()
            # 前向传播
            # 该函数会根据不同的参数，会返回不同的值。 本例中, 会返回 loss 和 logits -- 模型的预测结果
            loss, logits = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # 累加 loss
            total_train_loss += loss.item()
            # 反向传播
            loss.backward()
            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
        # 平均训练误差
        avg_train_loss = total_train_loss / len(train_dataloader)
        # 单次 epoch 的训练时长
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))
        # ========================================
        #               Validation
        # ========================================
        # 完成一次 epoch 训练后，就对该模型的性能进行验证
        print("")
        print("Running Validation...")
        t0 = time.time()
        # 设置模型为评估模式
        model.eval()
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # 将输入数据加载到 gpu 中
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # 评估的时候不需要更新参数、计算梯度
            with torch.no_grad():
                (loss, logits) = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
            # 累加 loss
            total_eval_loss += loss.item()
            # 将预测结果和 labels 加载到 cpu 中计算
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # 计算准确率
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        # 打印本次 epoch 的准确率
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        # 统计本次 epoch 的 loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        # 统计本次评估的时长
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))
        # 记录本次 epoch 的所有统计信息
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )
    return training_stats

seed_val = 42
epochs = 4
training_stats = train(seed_val, epochs, model, train_dataloader, validation_dataloader, optimizer)
pd.set_option('precision', 2)

# 加载训练统计到 DataFrame 中
df_stats = pd.DataFrame(data=training_stats)

# 使用 epoch 值作为每行的索引
df_stats = df_stats.set_index('epoch')
# 展示表格数据
df_stats

import matplotlib.pyplot as plt
import seaborn as sns

# 绘图风格设置
sns.set(style='darkgrid')
# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)
# 绘制学习曲线
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])
plt.show()
# 5. 在测试集上测试性能
# 下面我们加载测试集，并使用 Matthew相关系数来评估模型性能，
# 因为这是一种在 NLP 社区中被广泛使用的衡量 CoLA 任务性能的方法。使用这种测量方法，+1 为最高分，-1 为最低分。
# 于是，我们就可以在特定任务上，横向和最好的模型进行性能对比了。
# 加载数据集
test_data = pd.read_excel("./test_data.xlsx")
test_data = shuffle(test_data)
test_data.index = range(len(test_data))
# 打印数据集大小
print('Number of test sentences: {:,}\n'.format(test_data.shape[0]))
# 将数据集转换为列表
def validate(data, tokenizer, batch_size):
    test_sentences = data.sentence.values
    test_labels = data.label.values
    # 分词、填充或截断
    test_inputids = []
    test_attentionmasks = []
    for sent in test_sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            max_length = 100,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )
        test_inputids.append(encoded_dict['input_ids'])
        test_attentionmasks.append(encoded_dict['attention_mask'])
    test_inputids = torch.cat(test_inputids, dim=0)
    test_attentionmasks = torch.cat(test_attentionmasks, dim=0)
    test_labels = torch.tensor(test_labels)
    # 准备好数据集
    prediction_data = TensorDataset(test_inputids, test_attentionmasks, test_labels)
    prediction_sampler = SequentialSampler(prediction_data)  # 按照顺序采样
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    # 评估测试集性能
    print('Predicting labels for {:,} test sentences...'.format(len(test_inputids)))
    # 依然是评估模式
    model.eval()
    # Tracking variables
    predictions, true_labels = [], []
    # 预测
    for batch in prediction_dataloader:
        # 将数据加载到 gpu 中
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # 不需要计算梯度
        with torch.no_grad():
            # 前向传播，获取预测结果
            outputs = model(b_input_ids, token_type_ids = None,
                            attention_mask = b_input_mask)
        logits = outputs[0]
        # 将结果加载到 cpu 中
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # 存储预测结果和 labels
        predictions.append(logits)
        true_labels.append(label_ids)
    print('    DONE.')
    # 使用 Mathews 相关性系数（MCC）来评估测试集性能，原因在于类别的分布是不均匀的：
    print('Positive samples: %d of %d (%.2f%%)' % (data.label.sum(), len(data.label), (data.label.sum() / len(data.label) * 100.0)))
    # 最终评测结果会基于全量的测试数据，不过我们可以统计每个小批量各自的分数，以查看批量之间的变化。
    return predictions, true_labels

bert_tokenizer_path = './model/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_tokenizer_path, do_lower_case=True)
predictions, true_labels = validate(test_data, tokenizer, batch_size)
from sklearn.metrics import matthews_corrcoef
matthews_set = []
# 计算每个 batch 的 MCC
print('Calculating Matthews Corr. Coef. for each batch...')
# For each input batch...
for i in range(len(true_labels)):
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    # 计算该 batch 的 MCC
    matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
    matthews_set.append(matthews)
# 创建柱状图来显示每个 batch 的 MCC 分数
ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)
plt.title('MCC Score per Batch')
plt.ylabel('MCC Score (-1 to +1)')
plt.xlabel('Batch #')
plt.show()
# 合并所有 batch 的预测结果
flat_predictions = np.concatenate(predictions, axis=0)
# 取每个样本的最大值作为预测值
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
# 合并所有的 labels
flat_true_labels = np.concatenate(true_labels, axis=0)
# 计算 MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
print('Total MCC: %.3f' % mcc)


# 线下测试
a = pd.DataFrame([[1, '某保险公司需要将其闲置资金进行投资，除银行存款外，还可以使用哪些资金运用形式？'],
                  [0, '在教育体系中，如何平衡标准化考试与学生个性化需求？'], [0, '量子计算的发展将如何颠覆现有的数据加密和信息安全领域？'], [0, '加密货币的兴起对传统金融体系构成哪些挑战和机遇？'],
                  [1, '某罪犯在执行刑罚期间认为自己被错误定罪，可以向哪个机关提出申诉？？'], [1, '某医疗器械注册人在生产过程中发现了存在的医疗器械质量安全风险，但没有采取有效措施消除，药品监督管理部门可以采取什么措施？'], [0, '这本书好看吗？'], [0, '你预测的这么准确吗？'],
                  [1, '这应该不是贿赂吧？']], columns=['label', 'sentence'])
predictions, true_labels = predictions, true_labels = validate(a, tokenizer, 9)
np.argmax(predictions[0], axis=1).flatten()

# 这应该不是贿赂吧，这句话就有点模棱两可了

import torch.onnx

with torch.no_grad():
    torch.onnx.export(
        model,
        a,
        "srcnn.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])