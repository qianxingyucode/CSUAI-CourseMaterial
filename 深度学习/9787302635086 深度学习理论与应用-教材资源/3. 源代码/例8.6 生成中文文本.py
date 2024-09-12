
import torch
from transformers import BertTokenizer
from pytorch_transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader,TensorDataset
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

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  #利用BERT的分词器
path = r'./data/THUCNews'
name = r'train.txt'
fn = path + '\\' + name
train_texts, train_labels = getTexts_Labels(fn)  #该函数代码见例??
input_ids = []
for text,label in zip(train_texts, train_labels):
    if label != 3:
        continue
    text = text + '[SEP]'
    text_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    input_ids += text_ids


seq_len = 512 #序列的长度设置为512（序列越长，对内存的要求越高）
# 使得input_ids的长度为sample_num*seq_len并运用所有的训练文本
sample_num = len(input_ids)//seq_len
if len(input_ids)%seq_len>0:
    input_ids = input_ids[:sample_num * seq_len] + input_ids[-seq_len:]
    sample_num = sample_num + 1
else:
    input_ids = input_ids[:sample_num * seq_len]
input_ids = torch.LongTensor(input_ids)


input_ids = input_ids.reshape(-1,seq_len)  #torch.Size([657, 512])
train_loader = DataLoader(input_ids,batch_size=3, shuffle=False)
print('数据集大小：',len(train_loader.dataset))
......  # 【此处为本例部分核心代码，已省略，完整代码见教材P258；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

text_model.train()
optimizer = torch.optim.Adam(text_model.parameters(), lr=1e-5)
acc_steps = 4
for ep in range(30):
    for k,b_input_ids in enumerate(train_loader):
        b_input_ids = b_input_ids.to(device)
        #输入GPT2模型，对其进行训练
        outputs = text_model.forward(input_ids=b_input_ids, labels=b_input_ids)
        loss, logits = outputs[:2]
        loss = loss / acc_steps
        loss.backward() #梯度累加
        if (k+1)%acc_steps ==0:  #采用梯度累计方法对模型进行训练
            print(ep, loss.item())
            optimizer.step()
            optimizer.zero_grad()

torch.save(text_model,'text_model')
text_model = torch.load('text_model')
text_model.eval()
text = '高考'
seq_len = 512
tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
tokens_tensor = torch.LongTensor(tokens).to(device)
tokens_tensor = tokens_tensor.unsqueeze(0)  # torch.Size([1, 2])
generated_tonken_tensor = tokens_tensor
with torch.no_grad():
    for _ in range(100):
        outputs = text_model(generated_tonken_tensor)
        next_token_logits = outputs[0][0, -1, :]  #torch.Size([13317])
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        top6 = torch.topk(next_token_logits, 6)[0] #torch.Size([6])
        top6 = top6[-1] #取最小的权值
        #将低于这6个权值的分量值都设置为负无穷小（-float('Inf') ）
        next_token_logits[next_token_logits < top6] = -float('Inf')
        #按归一化后的权重概率选择下一个词
        next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1),
                   num_samples=1)
        #将选中的词加入到generated_tonken_tensor当中，以便用于产生下一个词
        generated_tonken_tensor = torch.cat((generated_tonken_tensor,
next_token.unsqueeze(0)), dim=1)
generated_tonkens = tokenizer.convert_ids_to_tokens(
generated_tonken_tensor[0].tolist())
generated_text = ''.join(generated_tonkens).replace('[SEP]','。')
print('产生的中文文本：',generated_text)
