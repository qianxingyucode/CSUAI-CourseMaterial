import torch
import torch.nn as nn
import numpy as np
from pytorch_transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.nn.functional as F
from pytorch_transformers import BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getTexts_Labels(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = list(f)
    texts,labels = [],[]
    for line in lines:
        line = line.strip().replace('\n','')
        line = line.split('\t')
        if len(line)!=2:
            continue
        text,label = line[0],line[1]
        texts.append(text)
        labels.append(int(label))
    return texts,labels  #texts[]中，一句话一个字符串，一个字符串一个元素，构成texts；labels为对应的类别整数

#对texts中的每一个句子按字切分，然后进行索引编码，并等长化（用0填补）
def equal_len_coding(texts):
    train_tokenized_text = [tokenizer.tokenize(sentence) for sentence in texts]  # 按字切分文本
    input_ids = [tokenizer.convert_tokens_to_ids(char) for char in train_tokenized_text]  # 按索引编码
    #每一个句子一个列表，train_input_ids是由多个列表组成的列表（多个句子编码的列表）
    for i in range(len(input_ids)):  # 将样本数据填充至长度为 MAX_LEN
        tmp = input_ids[i]
        input_ids[i] = tmp[:MAX_LEN]
        input_ids[i].extend([0] * (MAX_LEN - len(input_ids[i]))) #0
    input_ids = torch.LongTensor(input_ids)
    return input_ids
#----------------------------------------------
MAX_LEN = 50  #33
batch_size = 32

#path = r'E:\教学资料\教材编写\深度学习教材编写\a. 教材撰写\学生的材料\潘秋宇\预训练模型\bert_pre_classification\bert_pre_classification\INPUT'
#model_name = 'bert-base-chinese'  # 指定需下载的预训练模型参数
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

path = r'./data/THUCNews'
name = r'train.txt'
fn = path + '\\' + name
#存放文本    存放文本的类别
train_texts, train_labels = getTexts_Labels(fn)  #180000

#train_texts, train_labels = train_texts[:2000], train_labels[:2000]  #作为示例，为节省时间，取2000条数据

name = r'test.txt'
fn = path + '\\' + name
#存放文本    存放文本的类别
test_texts, test_labels = getTexts_Labels(fn) #10000 10000

train_input_ids = equal_len_coding(train_texts) #torch.Size([2000, 50]) torch.Size([2000])
train_labels = torch.LongTensor(train_labels)

test_input_ids = equal_len_coding(test_texts)
test_labels = torch.LongTensor(test_labels)

print(train_input_ids.shape,train_labels.shape) #torch.Size([2000, 50]) torch.Size([2000])
print(test_input_ids.shape,test_labels.shape) #torch.Size([10000, 50]) torch.Size([10000])

# 构建数据集和迭代器
train_set = TensorDataset(train_input_ids,train_labels)
train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
test_set = TensorDataset(test_input_ids,test_labels)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True)

print(len(train_loader.dataset),len(test_loader.dataset))  #180000 10000

torch.save(train_loader,'train_loader')
torch.save(test_loader,'test_loader')

train_loader = torch.load('train_loader')
test_loader = torch.load('test_loader')
print(len(train_loader.dataset),len(test_loader.dataset))



class Bert_Model(nn.Module):
    def __init__(self):
        super(Bert_Model, self).__init__()
        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P245；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

    def forward(self, x, attention_mask=None): # torch.Size([32, 50])
        # 返回<class 'tuple'>，len=2  --torch.Size([32, 50, 768]) torch.Size([32, 768])
        #outputs = self.model(x, attention_mask=attention_mask)

        outputs = self.model(input_ids=x,
                             attention_mask=attention_mask,
                             token_type_ids=None #只有一句话，故不需设置该参数
                             )


        o = outputs[1]  # 取池化后的结果 batch * 768 == torch.Size([32, 768])
 
        o = self.dropout(o)
        o = self.fc(o)  #  torch.Size([32, 10])
        return o
#============================================================

bert_Model = Bert_Model().to(device)
optimizer = optim.Adam(bert_Model.parameters(), lr=1e-5)
for ep in range(5):
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #构造注意力掩码矩阵
        mask =  data.data.eq(0) #0为[PAD]的索引
        mask = mask.logical_not().byte() #转化为0,1矩阵
        output = bert_Model(data,mask)  #调用BERT模型
        #pred = torch.argmax(output, 1)
        loss = nn.CrossEntropyLoss()(output, target) #计算损失函数值
        if i%10==0:
            print(ep+1,i,len(train_loader.dataset),loss.item())
        optimizer.zero_grad()
        loss.backward()  # 会积累梯度
        optimizer.step()



torch.save(bert_Model,'bert_Model2')

'''
'''

bert_Model = torch.load('bert_Model2') #''bert_Model8864')  bert_Model1---89
bert_Model.eval()
correct = 0
for i, (data, target) in enumerate(test_loader):
    data,target = data.to(device),target.long().to(device)

    mask = data.data.eq(0)
    mask = mask.logical_not().byte()

    output = bert_Model(data,mask)  # torch.Size([32, 10])

    pred = torch.argmax(output, 1)  # torch.Size([32]) torch.Size([32])
    correct += (pred == target).sum().item()


print('正确分类的样本数：{}，样本总数：{}，准确率：{:.2f}%'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))



