import numpy as np
import torch
from torch import nn, optim
import os
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------
#TEXT.vocab.itos
def get_dataset_vocab(fn): #获取文本数据集，和词汇表
    tmp_lines = []
    with open(fn) as f:
        lines = list(f)
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            tmp_lines.append(line)
    f.close()
    vocab_dict = dict()
    examples = [] # 其中的元素格式为：(['we','are','student'],1)
    for i in range(0,len(tmp_lines),2):
        label = tmp_lines[i]
        text = tmp_lines[i+1]

        s_pos, e_pos = label.find('<Polarity>'), label.find('</Polarity>')
        label = label[s_pos + 10:e_pos]

        s_pos, e_pos = text.find('<text>'), text.find('</text>')
        text = text[s_pos + 6:e_pos].lower()
        text = text.replace(',', '')  # 去掉逗号
        text = text.replace('.', '')  # 去掉点号
        text = text.replace('!', '')  # 去掉！
        text = text.replace('(', ' ') # 去掉(
        text = text.replace(')', ' ') # 去掉)
        text = text.split()           #按照空格切分单词,text为list类型，长度不一
        t = (text,int(label))
        examples.append(t)

        for word in text:
            vocab_dict[word] = vocab_dict.get(word,0)+1  #统计词频
    sorted_vocab_dict = sorted(vocab_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    #如果需要，可以在此处去掉部分低频词
    vocab_word2index =  {'<unk>':0, '<pad>':1} #'<unk>'、'<pad>'分别表示未知单词和填充的单词
    for word,_ in sorted_vocab_dict:  #构建词汇的整数编号，从0，1开始
        if not word in vocab_word2index:
            vocab_word2index[word] = len(vocab_word2index) #构建单词的编码
    return examples, vocab_word2index  #形成由词汇序列和文本标记构成的样本的数据集，以及词汇的字典

#对文本打包，以当前包中最长的词序列长度为当前包的长度，不够的，填充1，0表示未知单词
def text_loader(examples,vocab_word2index, bs): #bs表示批的大小
    batchs, labels = [], []
    for i in range(0,len(examples),bs):
        max_len = 0
        for k in range(i, i + bs):
            if k==len(examples):
                break
            text, _ = examples[k]
            max_len = len(text) if max_len < len(text) else max_len

        cur_batchs, cur_labels = [], []
        for k in range(i,i+bs):
            if k==len(examples):
                break
            text,label = examples[k]
            #en_text = [vocab_word2index[word] for word in text]
            en_text = [vocab_word2index.get(word,0) for word in text]
            en_text = en_text + [1]*(max_len-len(en_text))  #填补1，使得当前包中各个向量的长度均为最大长度
            cur_batchs.append(en_text)
            cur_labels.append(label)
        cur_batchs = torch.LongTensor(cur_batchs)
        cur_labels = torch.LongTensor(cur_labels)

        batchs.append(cur_batchs)
        labels.append(cur_labels)



    return batchs,labels
#================================

class Lstm_model(nn.Module):
    #                    2524          100          117
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(Lstm_model, self).__init__()

        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P208；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        self.fc = nn.Linear(hidden_dim,1)

    def forward(self, x):  #torch.Size([128, 42])
        o = x
        o = self.embedding(o)  #torch.Size([128, 42, 100])
        o, _ = self.lstm(o)  #torch.Size([128, 42, 117])
        o = o.sum(1)    #torch.Size([128, 117])
        o = self.fc(o)  #torch.Size([128, 1])
        return o
#--------------------------------------------------
path = r'./data/corpus'
name = r'trains.txt'
fn = path+'//'+name
train_examples,train_vocab_word2index = get_dataset_vocab(fn)
train_batchs, train_labels = text_loader(train_examples,train_vocab_word2index,128) #打包
name = r'tests.txt'
fn = path+'//'+name
test_examples,_ = get_dataset_vocab(fn)
test_batchs, test_labels = text_loader(test_examples,train_vocab_word2index,128) #打包，一般用训练集的词汇表对测试集进行编码
#------------------------------
vocab_size = len(train_vocab_word2index)  #2524
embedding_dim = 100
hidden_dim = 128
#def __init__(self,vocab_size,embedding_dim,hidden_dim):
#                        2524          100             117
lstm_model = Lstm_model(vocab_size,   embedding_dim,   hidden_dim).to(device)
optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
#--------------------------------------------------


for ep in range(100):
    for text_batch, label_batch in zip(train_batchs, train_labels):
        text_batch, label_batch = text_batch.to(device), label_batch.to(device)
        pre_y = lstm_model(text_batch)  #orch.Size([128, 1])
        loss = nn.BCEWithLogitsLoss()(pre_y.squeeze(), label_batch.float())
        #print( pre_y.squeeze().shape, label_batch.shape ) # torch.Size([128]) torch.Size([128])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(ep, round(loss.item(),4))

torch.save(lstm_model,'lstm_model')
'''
'''


lstm_model = torch.load('lstm_model')
lstm_model.eval()
acc = 0
for text_batch, label_batch in zip(test_batchs, test_labels):
    text_batch, label_batch = text_batch.to(device), label_batch.to(device)
    pre_y = lstm_model(text_batch)
    pre_y = torch.sigmoid(pre_y)
    pre_y = torch.round(pre_y)  # 四舍五入，变成了0或1
    pre_y = pre_y.squeeze().long()
    correct = torch.eq(pre_y, label_batch).long()
    acc += correct.sum()
print('在测试集上的准确率：{:.1f}%'.format(100.*acc/len(test_examples)))

'''
在测试集上的准确率：80.0%
'''

print('----------------------00')
exit(0)







