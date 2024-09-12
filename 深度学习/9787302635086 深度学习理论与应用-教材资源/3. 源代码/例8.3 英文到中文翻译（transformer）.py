import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math
import jieba
jieba.setLogLevel(jieba.logging.INFO)  # 屏蔽jieba分词时出现的提示信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 30  # 句子的最大长度
# 加载数据集
path = r'.\data\translate'
fg = open(path + '\\' + "en_zh_data.txt", encoding='utf-8')
lines = list(fg)
fg.close()
pairs = []
for line in lines:
    line = line.replace('\n', '')
    pair = line.split('--->')  # 英中文句子以字符串'--->'隔开
    if len(pair) != 2:
        continue
    en_sen = pair[0]  # 英文句子
    zh_sen = pair[1]  # 中文句子
    pairs.append([en_sen, zh_sen])
pairs = pairs[:50]  # 为了节省调试时间，只用50对英中文句子来训练

def getData(pairs):
    temp = []
    for pair in pairs:
        split_eng = pair[0].split(' ')  # 切分英文单词
        split_chi = [word for word in jieba.cut(pair[1]) if word != ' ']  # 中文分词
        if len(split_eng) < MAX_LENGTH and len(split_chi) < MAX_LENGTH:
            temp.append(pair)  # 保留长度小于MAX_LENGTH的句子对
    pairs = temp
    eng_lang = Word_Dict('eng')  # 初始化中文字典
    chi_lang = Word_Dict('chi')  # 初始化英文字典
    for pair in pairs:  # 对每个句子对构造字典
        eng_lang.addOneSentence(pair[0])  # 建立英文单词索引字典
        chi_lang.addOneSentence(pair[1])  # 建立中文词索引字典
    return eng_lang, chi_lang, pairs  # 返回构造好的英文字典和中文字典，以及符合长度的句子对

# 语言文本词汇的字典类Word_Dict
class Word_Dict:  # 构建词典，为每个词确定一个唯一的索引号
    def __init__(self, name):
        self.name = name  # 指处理中文还是英文
        self.word2index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.index2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
    def addOneSentence(self, sentence):
        if self.name == 'eng':
            for word in sentence.split(' '):  # 英文的话 用split(' ')分词
                self.addOneWord(word)  #
        elif self.name == 'chi':  # 中文的话 用jieba分词
            split_chi = [char for char in jieba.cut(sentence) if char != ' ']
            for word in split_chi:
                self.addOneWord(word)
    def addOneWord(self, word):  # 将词加入到字典中
        if word not in self.word2index:
            index = len(self.index2word)
            self.word2index[word] = index
            self.index2word[index] = word

def sentence2tensor(lang, sentence,flag):
    indexes = []
    if flag=='encoder_in': #编码器的输入（英文句子）
        words = [word for word in sentence.split(' ') if word.strip() != '']  # 分词
        words = words[0:MAX_LENGTH]
        words = words + ['<PAD>']*(MAX_LENGTH-len(words)) #等长化
        indexes = [lang.word2index.get(word, 1) for word in words] # 1为未知单词'<UNK>'的索引
    elif flag == 'decoder_in':  # 解码器的输入（中文句子）
        words = [word for word in jieba.cut(sentence) if word.strip() != '']  # 分词
        words = ['<SOS>'] + words
        words = words[0:MAX_LENGTH]
        words = words + ['<PAD>'] * (MAX_LENGTH - len(words))  # 等长化
        indexes = [lang.word2index.get(word, 1) for word in words]
    elif flag == 'decoder_out':  # 解码器的期望输出（中文句子）
        words = [word for word in jieba.cut(sentence) if word.strip() != '']  # 分词
        words = words[0:MAX_LENGTH-1] #保证下面添加的结束符'<EOS>'不被截出
        words = words + ['<EOS>']
        words = words + ['<PAD>'] * (MAX_LENGTH - len(words))  # 等长化
        indexes = [lang.word2index.get(word, 1) for word in words]
    else:
        pass
    return torch.LongTensor(indexes).to(device)


class MyDataSet(Dataset):
    def __init__(self, pairs):
        super(MyDataSet, self).__init__()
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        en_sentence = pair[0] #英文句子
        zh_sentence = pair[1] #中文句子
        # 传入英文字典和英文句子，返回输入编码器的张量
        en_input = sentence2tensor(eng_lang, en_sentence, flag='encoder_in')
        # 传入中文字典和中文句子，返回输入解码器的张量
        de_input = sentence2tensor(chi_lang, zh_sentence, flag='decoder_in')
        # 传入中文字典和中文句子，返回输出解码器的张量（期望输出，即输入的标记）
        de_output = sentence2tensor(chi_lang, zh_sentence, flag='decoder_out')
        return en_input,de_input,de_output


# eng_lang=构造好的英文字典   chi_lang=构造好的中文字典   pairs=符合长度的句子对
eng_lang, chi_lang, pairs = getData(pairs)

mydataset = MyDataSet(pairs)
loader = DataLoader(mydataset, batch_size=9, shuffle=True)
#torch.Size([9, 30])




# transformers位置编码
class PosEncoding(nn.Module):
    def __init__(self, d_model,  max_len=MAX_LENGTH): #
        super(PosEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model #d_model需为偶数
    def forward(self, x):  # torch.Size([9, 30])
        p = torch.arange(0, self.max_len).float().unsqueeze(1)  #
        p_2i = torch.arange(0, self.d_model, 2)
        p_2i = 1./np.power(10000.0, (p_2i.float() / self.d_model))
        pos_code = torch.zeros(self.max_len, self.d_model)  # torch.Size([30, 256])
        pos_code[:, 0::2] = torch.sin(p * p_2i)  # <----torch.Size([30, 128])
        pos_code[:, 1::2] = torch.cos(p * p_2i)  # <----torch.Size([30, 128])
        pos_code = pos_code.to(device)  #  torch.Size([30, 256])  每个位置有一个位置向量了
        o = pos_code[:x.size(1)]  #x.size(1)为句子的长度
        o = o.unsqueeze(0)  #增加第一个维度，大小为1，表示有一个句子，这是o包含了一个句子中每个位置的位置编码（向量）  torch.Size([1, 30, 256])
        o = o.repeat(x.size(0), 1, 1)  # 每个句子中，相同位置的元素，它们的位置向量是相同的，因此复制即可
        o = o.permute([1, 0, 2])  #修改形状，改为(seq_len, batch_size, d_model) torch.Size([30, 9, 256])
        return o
# transformers位置编码，用于初始化嵌入层
def PosEncoding_for_Embedding(d_model,  max_len=MAX_LENGTH): #
    p = torch.arange(0, max_len).float().unsqueeze(1)  #
    p_2i = torch.arange(0, d_model, 2)
    p_2i = 1./np.power(10000.0, (p_2i.float() / d_model))
    pos_code = torch.zeros(max_len, d_model)  # torch.Size([30, 256])
    pos_code[:, 0::2] = torch.sin(p * p_2i)  # <----torch.Size([30, 128])
    pos_code[:, 1::2] = torch.cos(p * p_2i)  # <----torch.Size([30, 128])
    pos_code = pos_code.to(device)  #  torch.Size([30, 256])  每个位置有一个位置向量了
    return pos_code

def pos_code(x): #torch.Size([9, 30]
    one_sen_poses = [pos for pos in range(x.size(1))]
    all_sen_poses = torch.LongTensor(one_sen_poses).unsqueeze(0).to(device)
    all_sen_poses = all_sen_poses.repeat(x.size(0),1) #torch.Size([9, 30])
    return all_sen_poses


class MyTransformer(nn.Module):
    #                    256       4        2       512       310            334
    def __init__(self, d_model, nhead, layer_num, dim_ff, src_vocab_size, tgt_vocab_size):
        super(MyTransformer, self).__init__()
        #利用调用nn.Transformer()来实例化类的对象，构建Transformer模型
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=layer_num,
                                          num_decoder_layers=layer_num,
                                          dim_feedforward=dim_ff)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model) #定义面向英文单词的嵌入层

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model) #定义面向中文词的嵌入层
        self.pos_encoding = PosEncoding(d_model, max_len=MAX_LENGTH)
        #在本例中源句子和目标句子的最大长度设置为一样长，故可共享编码函数PosEncoding_for_Embedding
        #self.pos_embedding = nn.Embedding.from_pretrained(\
        #    PosEncoding_for_Embedding(d_model, MAX_LENGTH), freeze=True)

        self.fc = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, en_input, de_input):  # torch.Size([9, 30]) torch.Size([9, 30])
        cur_len = de_input.shape[1] #获取目标句子的固定长度
        #产生一个三角掩码矩阵
        tgt_mask = self.transformer.generate_square_subsequent_mask(cur_len).to(device)
        src_key_padding_mask = en_input.data.eq(0).to(device)  #产生编码器输入的布尔掩码矩阵
        tgt_key_padding_mask = de_input.data.eq(0).to(device)  #产生解码器输入的布尔掩码矩阵

        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P236；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        tgt_emb = self.tgt_embedding(de_input).permute([1, 0, 2]) #对解码器输入进行嵌入表示 #torch.Size([30, 9, 256])
        tgt_pos_code = self.pos_encoding(de_input) #对解码器输入进行位置编码
        #tgt_pos_emb = self.pos_embedding(pos_code(de_input)).permute([1, 0, 2])



        de_inputs = tgt_emb + tgt_pos_code #嵌入向量加上位置向量，构成解码器的输入向量 torch.Size([30, 9, 256])
        #de_inputs = tgt_emb + tgt_pos_emb


        # 送入Transformer，dec_outputs和de_input的形状相同
        dec_outputs = self.transformer(src=en_inputs, tgt=de_inputs,
                                       tgt_mask = tgt_mask,
                                       src_key_padding_mask = src_key_padding_mask,
                                       tgt_key_padding_mask = tgt_key_padding_mask)

        #self.transformer(src=en_inputs, tgt=de_inputs)




        tmp = self.fc(dec_outputs.transpose(0, 1))  #对Transformer的输出进行调整，使输出尺寸为目标语言的词汇数torch.Size([9, 30, 305])
        de_pre_y = tmp.view(-1, tmp.size(-1))  # torch.Size([270, 305])
        return de_pre_y


#-----------------------------------------------
# Transformer 参数
d_model = 256  # 嵌入向量的长度
nhead = 4      # 多头注意力的头个数
layer_num = 2   # 编码器和解码器的层数
dim_ff = 512  # FeedForward 的维度  隐含层神经元个数??




src_vocab_size = len(eng_lang.word2index)  # 输入字典的单词个数 英文字典  构造embdding层
tgt_vocab_size = len(chi_lang.word2index)  # 输出字典的单词个数  中文字典    构造embdding层

#                      Embedding维度     注意力的头个数     编码器和解码器的层数
transformer_model = MyTransformer(d_model=d_model, nhead=nhead, layer_num=layer_num,
                                  # FeedForward 的维度          源词汇数                      目标词汇数
                                  dim_ff=dim_ff, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size).to(device)
# criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(transformer_model.parameters(), lr=1e-3, momentum=0.99)

for ep in range(150):
    total_loss = 0
    for en_input,de_input,de_label in loader:
        en_input, de_input, de_label = en_input.to(device),de_input.to(device),de_label.to(device)
        #torch.Size([9, 30]) torch.Size([9, 30]) torch.Size([9, 30])
        de_pre_y = transformer_model(en_input, de_input) #torch.Size([270, 305])


        loss = nn.CrossEntropyLoss(ignore_index=0)(de_pre_y, de_label.view(-1)) #torch.Size([270, 305]) torch.Size([270])
        total_loss += loss  # 累加所有句子对的损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #break
    print('Epoch:', '%04d' % (ep + 1), 'loss =', '{:.6f}'.format(total_loss / len(loader.dataset)))
    #break


torch.save(transformer_model, 'transformer_model')


transformer_model = torch.load('transformer_model')
transformer_model.eval()
# 翻译一个英文句子
mysentence = mydataset.pairs[3][0]
#mysentence = "but before this new order appears the world may be faced with spreading disorder if not outright chaos"

mysentence = mysentence.lower()  # 小写
print(mysentence)
en_input = sentence2tensor(eng_lang, mysentence, flag='encoder_in') #torch.Size([30])
#--------------------------------------------------------------
en_input = en_input.unsqueeze(0).to(device)  # torch.Size([1, 30])

start_index = chi_lang.word2index["<SOS>"]  # 2  获取目标语言字典的开始标志位
de_input = torch.LongTensor([[]]).to(device)
next_index = start_index

while True:
    # 解码器输入最开始为 标志位SOS 2  逐个预测 拼接 直到结束位EOS 得到最后的dec_input
    de_input = torch.cat([de_input.detach(), torch.tensor([[next_index]]).to(device)], -1)
    de_pre_y = transformer_model(en_input, de_input) #torch.Size([1, 305])
    prob = de_pre_y.max(dim=-1, keepdim=False)[1] #torch.Size([1])

    next_index = prob.data[-1] #prob.item()

    if next_index == chi_lang.word2index["<EOS>"]:
        break

word_indexes = de_input.squeeze().cpu()
out_words = [chi_lang.index2word[index.item()] for index in word_indexes]
out_sentence = ' '.join(out_words[1:])
print(out_sentence)


