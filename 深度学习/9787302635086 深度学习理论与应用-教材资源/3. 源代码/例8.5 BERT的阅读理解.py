import torch
import torch.nn as nn
from torch import optim
from pytorch_transformers import BertTokenizer,BertModel
from transformers import BertTokenizerFast
from torch.utils.data import Dataset,DataLoader
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 512

#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese',
                                                add_special_tokens=False,  # 不添加CLS,SEP
                                                do_lower_case=False)  # 不区分大小写字母，

torch.save(tokenizer,'tokenizer')
'''
'''
tokenizer = torch.load('tokenizer')

#从json文件中读取问题文本、文本段、答案的起始位置，一条样本放在一个四元组中，返回多个四元组的list
def get_query_passage_answer(fn):
    with open(fn, 'r', encoding='utf-8') as reader:
        data = json.load(reader)['data']
    examples = []
    data = data[0] #len(data)=1
    for paragraph in data['paragraphs']: #data['paragraphs']是长度为100的list
        paragraph_text = paragraph['context']  #篇章的内容
        qa = paragraph['qas'][0] #paragraph['qas']是长度为1的list，里面有一个字典
        query = qa['question'] #问题文本
        id = qa['id']
        if len(query+paragraph_text)+3>MAX_LEN:
            continue
        answer = qa['answers'][0]['text']  #答案文本
        answer_start = qa['answers'][0]['answer_start'] #起始位置索引
        answer_end = answer_start+len(answer)           #终止位置索引
        item = (query,paragraph_text,answer_start,answer_end)
        examples.append(item)
    return examples

class MyDataSet(Dataset):
    def __init__(self, query_passage_answer):
        super(MyDataSet, self).__init__()
        self.query_passage_answer = query_passage_answer
    def __len__(self):
        return len(self.query_passage_answer)
    def __getitem__(self, idx):
        query_passage_answer = self.query_passage_answer[idx]
        query = query_passage_answer[0]  #获得问题文本
        passage = query_passage_answer[1] #获得篇章文本
        answers_start = query_passage_answer[2] #获得答案的起始位置索引
        answers_end = query_passage_answer[3]  #获得答案的终止位置索引
        #answers = passage[answers_start:answers_end]
        # 对篇章文本进行索引编码，同时返回编码前后位置索引之间的关系信息
        tokenizing_result = tokenizer.encode_plus(passage,
                            return_offsets_mapping=True,
                            add_special_tokens=False)
        #对问题文本进行分词和索引编码
        query_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
        passage_ids = tokenizing_result['input_ids'] #获取篇章文本的索引编码
        token_span = tokenizing_result['offset_mapping'] #获得编码前后位置索引之间的关系信息
        query_ids = [101]+query_ids+[102] #在问题文本前后分别加上[CLS]和[SEP]的索引
        passage_ids = passage_ids+[102] #在篇章文本编码后面加上[SEP]的索引
        sen1_len = len(query_ids) #输入BERT的句子长度
        sen2_len = len(passage_ids) #填充长度
        sen_len = sen1_len + sen2_len

        input_ids = query_ids + passage_ids + [0] * (MAX_LEN - sen_len) #填充[PAD]，0为[PAD]的索引
        #(1)构造问题+篇章的索引编码
        input_ids = torch.tensor(input_ids)

        token_type_ids = [0] * sen1_len + [1] * (MAX_LEN - sen1_len)#(3)构造句子掩码向量 torch.Size([512]
        attention_mask = [1] * sen_len + [0] * (MAX_LEN - sen_len)#(2)构造注意力掩码向量 torch.Size([512]
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        # passage----->
        # token_span----->
        #建立分词前后之间字符位置的索引映射关系
        CharInd_TokenInd = [[] for _ in range(len(passage)+1)]
        CharInd_TokenInd[len(passage)] = [len(passage)]
        for token_ind, char_sp in enumerate(token_span):
            for text_ind in range(char_sp[0], char_sp[1]):
                CharInd_TokenInd[text_ind] += [token_ind]
        for k in range(len(CharInd_TokenInd) - 2, -1, -1):  # 填补空格
            if CharInd_TokenInd[k] == []:
                CharInd_TokenInd[k] = CharInd_TokenInd[k + 1]

        answers_start_ids = sen1_len + CharInd_TokenInd[answers_start][0]
        answers_end_ids = sen1_len + CharInd_TokenInd[answers_end][0]
        labels = [answers_start_ids,answers_end_ids]  #(4)构造答案的起始位置索引张量（标签）
        labels = torch.tensor(labels)
        return  input_ids, attention_mask, token_type_ids, labels
        #返回用于BERT的文本索引编码（问题+篇章文本的索引编码）、注意力掩码向量、句子掩码向量和答案的起始位置索引张量（标签）

class BertForReading(nn.Module):
    def __init__(self): #
        super(BertForReading, self).__init__()
        # 加载预训练模型
        self.model = BertModel.from_pretrained('bert-base-chinese',
                     cache_dir="./Bert_model").to(device)#加载模型
        self.qa_outputs = nn.Linear(768, 2)  # 有两个预测任务
    def forward(self, b_input_ids, b_attention_mask, b_token_type_ids):  #
        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P250；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        logits = self.qa_outputs(sequence_output) #torch.Size([8, 512, 2])
        return logits

#-----------------------------------------------
path = r'.\data\dureader_robust-data'
name = r'train.json'
fn = path +'\\'+name
query_passage_answer = get_query_passage_answer(fn)  #样本数量为12649

mydataset = MyDataSet(query_passage_answer)
data_loader = DataLoader(mydataset, batch_size=8, shuffle=True) #打包数据集
print('数据集大小：',len(data_loader.dataset))
reading_model = BertForReading().to(device) #实例化
optimizer = optim.Adam(reading_model.parameters(), lr=1e-5)

#开始训练
for ep in range(10):
    for k, (b_input_ids, b_attention_mask, b_token_type_ids, b_labels) in enumerate(data_loader):
        b_input_ids, b_attention_mask = b_input_ids.to(device), b_attention_mask.to(device)
        b_token_type_ids, b_labels = b_token_type_ids.to(device), b_labels.to(device)

        logits = reading_model(b_input_ids, b_attention_mask, b_token_type_ids)
        #logits的形状：torch.Size([8, 512, 2])
        start_logits = logits[:, :, 0] #torch.Size([8, 512])
        end_logits = logits[:, :, 1]
        #ignore_index = start_logits.size(1) #序列长度  512
        #loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss_fun = nn.CrossEntropyLoss()

        start_label = b_labels[:, 0]  #torch.Size([8])
        end_label = b_labels[:, 1]



        start_loss = loss_fun(start_logits, start_label)
        end_loss = loss_fun(end_logits, end_label)
        loss = (start_loss + end_loss) / 2
        if k%10==0:
            print(ep,k,len(data_loader),loss.item())
        optimizer.zero_grad()
        loss.backward()  # 会积累梯度
        optimizer.step()
torch.save(reading_model,'reading_model2')
'''
exit(0)
'''

reading_model = torch.load('reading_model2')
reading_model.eval()
correct = 0
for k, (b_input_ids, b_attention_mask, b_token_type_ids, b_labels) in enumerate(data_loader):
    b_input_ids, b_attention_mask = b_input_ids.to(device), b_attention_mask.to(device)
    b_token_type_ids, b_labels = b_token_type_ids.to(device), b_labels.to(device)
    # torch.Size([8, 512]) torch.Size([8, 512]) torch.Size([8, 512]) torch.Size([8, 2])
    logits = reading_model(b_input_ids, b_attention_mask, b_token_type_ids)
    #torch.Size([8, 512, 2])
    start_logits, end_logits = logits.split(1, dim=-1) #torch.Size([8, 512, 1]) torch.Size([8, 512, 1])
    start_logits = start_logits.squeeze(-1)  #   torch.Size([8, 512])
    end_logits = end_logits.squeeze(-1)  # torch.Size([8, 512])

    pre_start_pos = torch.argmax(start_logits, dim=1).long()  # torch.Size([8])
    pre_end_pos = torch.argmax(end_logits, dim=1).long()

    start_positions = b_labels[:, 0]  # torch.Size([8])
    end_positions = b_labels[:, 1]


    t1 = (pre_start_pos == start_positions).byte()
    t2 = (pre_end_pos == end_positions).byte()
    t = (t1*t2).sum()
    correct += t
    if k%5 == 0:
        print('预测完成率：',round(1.*k/len(data_loader),4))

    #break
correct = 1.*correct/len(data_loader.dataset)
print('准确率：',round(correct.item(),3))






