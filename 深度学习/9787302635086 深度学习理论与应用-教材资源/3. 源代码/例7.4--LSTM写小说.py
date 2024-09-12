import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import jieba
jieba.setLogLevel(jieba.logging.INFO) #屏蔽jieba分词时出现的提示信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#----------------------------------------------
torch.manual_seed(123)
#random.seed(123)
np.random.seed(10)

#形成由分词后的词汇构成的新数据集，并获得词汇字典（用于词的整数编码）
def get_texts_vocab(fn):
    max_len = 0
    sentence_words = []
    vocab_dict = dict()
    with open(fn, encoding='UTF-8') as f:
        lines = list(f)
        for line in lines:
            line = line.strip()
            if line == '' or '---' in line:
                continue
            words = list(jieba.cut(line))
            words = ['<s>'] + words + ['<e>']
            if max_len < len(words):
                max_len = len(words)
            sentence_words.append(words)
            for word in words:
                vocab_dict[word] = vocab_dict.get(word, 0) + 1  # 统计词频
    f.close()
    sorted_vocab_dict = sorted(vocab_dict.items(), key=lambda kv: (kv[1], kv[0]), reverse=True) #按词频降序排列
    sorted_vocab_dict = sorted_vocab_dict[:-10]  #减掉10个低频词
    vocab_word2index = {'<unk>': 0, '<pad>': 1, '<s>':2, '<e>':3}
    for word, _ in sorted_vocab_dict:  # 构建词汇的整数编号，从0，1开始
        if not word in vocab_word2index:
            vocab_word2index[word] = len(vocab_word2index)
    return sentence_words, vocab_word2index

#对给定一个由编码（整数）构成的序列，构建若干个长度为10的序列及其后面的输出词汇编号，即形成等长的输入-输出对
#如果给定序列的长度为n，则形成的输入-输出对的数量为n+1
def enOneTxt(en_ws):
    ln = len(en_ws)
    texts,labels = [],[]
    for pre_k in range(1,ln):#输入句子的长度为10
        ps = pre_k - 10
        ps = 0 if ps<0 else ps
        pe = pre_k-1
        txt = en_ws[ps:pe+1]
        txt = txt + [1]*(10-len(txt))
        label = en_ws[pre_k]
        texts.append(txt)
        labels.append(label)
    return texts, labels



#all_sen_words存放所有的文本行，对其中的每一行进行整数编码（利用字典vocab_word2index），
# 然后基于每一个文本行（编码后）生成一系列的输入-输出对，其中输入为长度为10是的整数序列
#以所有这样的输入-输出对构成训练数据
def enAllTxts(all_sen_words, vocab_w2i):
    texts, labels = [], []
    for i, words in enumerate(all_sen_words):
        en_words = [vocab_w2i.get(word, 0) for word in words]
        txts, lbs = enOneTxt(en_words)
        texts = texts + txts
        labels = labels + lbs
    texts, labels = torch.LongTensor(texts), torch.LongTensor(labels)
    return texts, labels

#定义自动生成文本的类
class Novel_model(nn.Module):
    def __init__(self,vocab_size):
        super(Novel_model, self).__init__()
        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P214；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        self.fc2 = nn.Linear(512, vocab_size)

    def forward(self, x):  #torch.Size([128, 10])
        o = x
        o = self.embedding(o)
        o, _ = self.lstm(o) #torch.Size([128, 10, 512])
        o = torch.sum(o, dim=1) #torch.Size([128, 512])  torch.Size([128, 1024])

        o = self.fc1(o)
        o = torch.relu(o)
        #o = nn.Dropout(p=0.5)(o)
        o = self.fc2(o)  #torch.Size([128, 1524])
        return o


#---------------------------------------------------------
path = r'./data'
name = r'金庸小说节选.txt'
fn = path+'//'+name
sentence_words, vocab_word2index = get_texts_vocab(fn)  #构建数据集和编码字典
texts, labels = enAllTxts(sentence_words, vocab_word2index) #生成训练数据
#torch.Size([3866, 10]) torch.Size([3866]) 1524
#print(texts.shape, labels.shape,len(vocab_word2index))
vocab_index2word = { index:word for word,index in vocab_word2index.items()} #用于解码



dataset = TensorDataset(texts, labels)
#dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=128)

#解决类不平衡问题
class_dict = dict()
for label in labels:
    lb = label.item()
    class_dict[lb] = class_dict.get(lb, 0) + 1  # 统计各类别词汇出现的频次
weights = []   #跟dataloader.dataset中的数据行要一一对应
for label in labels:
    lb = label.item()
    weights.append(class_dict[lb])
weights = 1./torch.FloatTensor(weights)
sampler = WeightedRandomSampler(weights=weights, replacement=True,num_samples=len(labels)*1000) #解决类不平衡问题
#产生num_samples个下标值，下标取值从0到|weights|-1，个数跟weights中对应的分量值成正比
#print(list(sampler)) 每次调用sampler ，结果都不一样

dataloader = DataLoader(dataset=dataset, batch_size=128, sampler=sampler, shuffle=False)

novel_model = Novel_model(vocab_size=len(vocab_word2index)).to(device)
optimizer = torch.optim.Adam(novel_model.parameters(), lr=0.01)


for ep in range(5):
    for i, (batch_texts,batch_labels) in enumerate(dataloader):
        batch_texts, batch_labels = batch_texts.to(device),batch_labels.to(device)
        batch_out = novel_model(batch_texts) #torch.Size([128, 10]) ---> torch.Size([128, 1524])

        #print(batch_texts.shape,batch_out.shape)
        #exit(0)
        #torch.Size([128, 10]) torch.Size([128])
        loss = nn.CrossEntropyLoss()(batch_out, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%500==0:
            print(ep, round(loss.item(),8))


torch.save(novel_model,'novel_model')
torch.save(vocab_word2index,'vocab_word2index')
torch.save(vocab_index2word,'vocab_index2word') 
'''
'''

novel_model = torch.load('novel_model')
vocab_word2index = torch.load('vocab_word2index')
vocab_index2word = torch.load('vocab_index2word')
novel_model.eval()

def getNextWord(s): #给定一个词序列，生成它的下一个词
    words = list(jieba.cut(s))
    words = ['<s>'] + words #  + ['<e>']
    en_words = [vocab_word2index.get(word,0) for word in words]
    en_words = en_words[len(en_words)-10:len(en_words)]
    en_words = en_words + [1]*(10-len(en_words))
    batch_texts = torch.LongTensor(en_words).unsqueeze(0).to(device)
    batch_out = novel_model(batch_texts)
    batch_out = torch.softmax(batch_out, dim=1)
    pre_index = torch.argmax(batch_out, dim=1)
    word = vocab_index2word[pre_index.item()]
    return word


#'杨过' #'郭靖'  #s = '郭靖和黄蓉'   忽必烈
seq = '黄蓉'
while True:  #生成小说文本
    w = getNextWord(seq)
    if w=='<e>':
        break
    seq = seq+w
print('生成的小说文本：', seq)






