import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
import jieba
MAX_LENGTH = 30
jieba.setLogLevel(jieba.logging.INFO)  # 屏蔽jieba分词时出现的提示信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载数据集
path = r'.\data\translate'
fg = open(path + '\\' + "en_zh_data.txt", encoding='utf-8')
lines = list(fg)  # 把整个文件放到列表中去   encoding=‘utf-8’  UTF-8
fg.close()
pairs = []
for line in lines:
    line = line.replace('\n', '')
    pair = line.split('--->')  # 英中文句子以字符串'--->'隔开
    if len(pair) != 2:
        continue
    en_sen = pair[0]  # 英文句子
    zh_sen = pair[1]  # 英文句子
    pairs.append([en_sen, zh_sen])
pairs = pairs[:50]  # 为了节省调试时间，只用50对英中文句子来训练


# 语言文本词汇的字典类 words_dict
class Lang_Dict:  # 构建词典，为每个词确定一个唯一的编号
    def __init__(self, name):
        self.name = name  # 指处理中文还是英文
        self.word2index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.index2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_num = 4  # 字典现有单词数  PAD  UNK  SOS  EOS  填充 未知  开始  结束

    def addSentence(self, sentence):
        if self.name == 'en':
            for word in sentence.split(' '):  # 英文的话 用split(' ')分词
                self.addWord(word)  #
        elif self.name == 'zh':  # 中文的话 用jieba分词
            split_zh = [char for char in jieba.cut(sentence) if char != ' ']
            for word in split_zh:
                self.addWord(word)

    def addWord(self, word):  # 将词加入到字典中
        if word not in self.word2index:
            self.word2index[word] = self.word_num
            self.index2word[self.word_num] = word
            self.word_num += 1


def prepareData(pairs):
    temp = []
    for pair in pairs:
        split_en = pair[0].split(' ')  # 分词统计长度
        split_zh = [word for word in jieba.cut(pair[1]) if word != ' ']  # 分词统计长度
        if len(split_en) < MAX_LENGTH and len(split_zh) < MAX_LENGTH:
            temp.append(pair)  # 保留小于定义的MAX_LENGTH的句子对
    pairs = temp

    en_lang = Lang_Dict('en')  # 初始化输入语言的字典
    zh_lang = Lang_Dict('zh')  # 初始化输出语言的字典
    for pair in pairs:  # 对每个句子对构造字典
        en_lang.addSentence(pair[0])  # 输入英文句子  构造英文字典
        zh_lang.addSentence(pair[1])  # 输入中文句子 构造中文字典
    return en_lang, zh_lang, pairs  # 返回构造好的英文字典和中文字典，以及符合长度的句子对


# en_lang=构造好的英文字典   zh_lang=构造好的中文字典   pairs=符合长度的句子对
en_lang, zh_lang, pairs = prepareData(pairs)


# len(pairs)=32,格式没有彼变

# 将句子转为张量  sentence=输入句子，lang=构造好的字典（英文或者中文），flag表示句子作为解码端输入还是输出处理
def sentence2tensor(lang, sentence):
    indexes = []  # 保存句子张量
    if lang.name == 'en':  # 如果是英文
        wordlist = sentence.split(' ')  # 分离
    elif lang.name == 'zh':  # 如果是中文
        wordlist = [word for word in jieba.cut(sentence) if word != ' ']  # 分词
    for word in wordlist:
        indexes.append(lang.word2index.get(word, 1))  # 1为未知单词'<UNK>'的编号
    indexes.append(lang.word2index['<EOS>'])  # 最后需要append结束标志位 EOS
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    # 一个句子一个张量??


class MyDataSet(Dataset):
    def __init__(self, pairs):
        super(MyDataSet, self).__init__()
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # 返回编码器输入张量  传入英文字典input_lang   句子对中的英文
        en_tensor = sentence2tensor(en_lang, pairs[idx][0])
        # 返回解码器目标张量  传入中文字典zh_lang   句子对中的中文
        zh_tensor = sentence2tensor(zh_lang, pairs[idx][1])
        return en_tensor, zh_tensor


mydataset = MyDataSet(pairs=pairs)  # 一个英文句子一个张量，一个中文句子一个张量


# len(mydataset) = 32

# 定义编码器
class Encoder(nn.Module):
    #en_vocab_num表示英文单词数，hidden_size表示词向量长度
    def __init__(self, en_vocab_num, hidden_size):
        super(Encoder, self).__init__()
        #self.hidden_size = hidden_size
        self.embedding = nn.Embedding(en_vocab_num, hidden_size)  #定义嵌入层
        #使用GRU作为循环神经网络，它是LSTM的变体
        self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self, x, h):
        x = self.embedding(x)
        x = x.reshape(1, 1, -1)  # 改变形状，以符合GRU的输入格式
        o, h = self.gru(x, h)
        return o, h  # 返回新的输出和隐层向量



# 定义解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size) 	#定义嵌入层
#第1次拼接用到的全连接层
        self.attention = nn.Linear(hidden_size * 2, MAX_LENGTH)
        #第2次拼接用到的全连接层
        self.attention_com = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.gru = nn.GRU(hidden_size, hidden_size)   #运用GRU作为循环神经网络
        #再做线性变换，以调整输出尺寸跟目标词汇数一样
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, de_input, de_hidden, en_outputs):
        #这三个参数分别是目标词输入（张量）、上一计算单元的隐层输出张量和
#解码器的输出张量
        embedded = self.embedding(de_input).view(1, 1, -1)  #词嵌入向量
        embedded = self.dropout(embedded)
#相当于将Q和K拼接（第一次）
        cat_Q_K = torch.cat((embedded[0], de_hidden[0]), 1)
        score_Q_K = self.attention(cat_Q_K)  	#做一次线性变换
        #softmax归一化（概率归一化），形成权重向量
        ......  # 【此处为本例部分核心代码，已省略，完整代码见教材P224；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

        #将激活后的结果和上一计算单元的隐层输出张量一起输入GRU进行处理
        output, de_hidden = self.gru(gru_input, de_hidden)
        #再做线性变换，以调整输出尺寸跟目标词汇数一样
        output = F.log_softmax(self.out(output[0]), dim=1)  #概率归一化
        return output, de_hidden





def init_hidden():  # 用于初始化隐含层
    return torch.zeros(1, 1, hidden_size).to(device)


hidden_size = 256  # 隐含层
encoder = Encoder(en_lang.word_num, hidden_size).to(device)
#decoder = Decoder(hidden_size, zh_lang.word_num).to(device)
decoder = AttnDecoderRNN(hidden_size, zh_lang.word_num).to(device)

en_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
de_optimizer = optim.SGD(decoder.parameters(), lr=0.01)


for ep in range(150):
    # 每次处理一对句子
    total_loss = 0
    for iter, (en_input, label) in enumerate(mydataset):  # 每次一个英文句子和中文句子对的两个张量 torch.Size([26, 1]) torch.Size([28, 1])
        en_hidden = init_hidden()  # 初始化编码器隐含层 张量[1,1,hidden_size]
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        en_input_len = en_input.size(0)  # 输入英文分词长度 torch.Size([30, 1]) torch.Size([29, 1])
        label_len = label.size(0)  # 输入中文分词长度

        en_outputs = torch.zeros(MAX_LENGTH, hidden_size).to(device)  # torch.Size([30, 256])
        for ei in range(en_input_len):
            en_output, en_hidden = encoder(en_input[ei], en_hidden)  # 编码器RNN网络输入 第ei个分词在字典里的序号的张量 例如5   以及刚才初始化的编码器隐含层
            # torch.Size([1]) torch.Size([1, 1, 256])--> torch.Size([1, 1, 256]) torch.Size([1, 1, 256])
            en_outputs[ei] = en_output[0, 0]  # 保存词x[ei]的向量

        de_hidden = torch.mean(en_outputs, dim=0).reshape(1, 1, -1) #用各计算单元输出的平均值作为语义向量
        de_input = torch.tensor(2).to(device)  #  2为起始标识符<SOS>的编号
        # 解码器的隐含层初始化为 编码器最后一个计算单元的隐含层输出（1，1，256）
        #de_hidden = en_hidden  # torch.Size([1, 1, 256])
        loss = 0
        for di in range(label_len):  # 目标中文的词语 一个一个来
            # torch.Size([1, 1]) torch.Size([1, 1, 256]) ---> torch.Size([1, 334]) torch.Size([1, 1, 256])
            # 334为中文的不同词汇数量
            #                torch.Size([]) torch.Size([1, 1, 256]) torch.Size([5, 256])
            #de_output, de_hidden = decoder(de_input, de_hidden) #, en_outputs
            de_output, de_hidden = decoder(de_input, de_hidden, en_outputs)


            _, max_i = de_output.topk(1)  # 取最高概率的词语的字典索引张量torch.Size([1, 1]) torch.Size([1, 1])
            de_input = label[di].squeeze().detach()
            loss += nn.CrossEntropyLoss()(de_output, label[di])

            if de_input.item() == 3:  # 3为终止标识符<EOS>的编号
                break
        loss.backward()
        en_optimizer.step()  # 更新编码器参数
        de_optimizer.step()  # 更新解码器参数

        loss = loss / label_len  # 一个句子的损失函数
        total_loss += loss

        #
    t = total_loss / len(mydataset)
    print('Epoch:', '%d' % (ep + 1), 'loss =', '{:.6f}'.format(t))
    # break #???????????

torch.save(encoder, 'encoder2')
torch.save(decoder, 'decoder2')
'''
# 训练完毕
exit(0)
'''

encoder = torch.load('encoder2')
decoder = torch.load('decoder2')

en_sentence = mydataset.pairs[20][0]
en_sentence = "But before this new order appears, the world may be faced with spreading disorder if not outright chaos."
en_sentence = en_sentence.lower()
en_tensor = sentence2tensor(en_lang, en_sentence)
encoder.eval()
decoder.eval()
with torch.no_grad():
    input_len = en_tensor.size(0)  # 单词长度
    en_hidden = init_hidden()  # 初始化隐含层
    max_len = MAX_LENGTH
    en_outputs = torch.zeros(MAX_LENGTH, hidden_size).to(device)  # 预设编码器输出[max_length,hidden_size]
    for ei in range(input_len):
        en_output, en_hidden = encoder(en_tensor[ei], en_hidden)  ##编码器RNN网络输入 第ei个分词在字典里的序号的张量
        en_outputs[ei] += en_output[0, 0]  # torch.Size([1, 1, 256])-->torch.Size([256])
    de_x = torch.tensor(2).to(device)  #2为<SOS>的编号
    de_hidden = en_hidden
    zh_words = []
    for di in range(max_len):
        # torch.Size([1, 334]) torch.Size([1, 1, 256])
        #de_output, de_hidden = decoder(de_x, de_hidden)
        de_output, de_hidden = decoder(de_x, de_hidden, en_outputs)
        #de_output, de_hidden = decoder(de_input, de_hidden, en_outputs) MAX_LENGTH
        _, max_i = de_output.data.topk(1)  # 最大概率值 以及对应的字典索引号

        if max_i.item() == 3: #3为<EOS>的编号
            # zh_words.append('<EOS>')
            break
        zh_words.append(zh_lang.index2word[max_i.item()])
        de_x = max_i.squeeze().detach()
zh_sentence = ' '.join(zh_words)
print(en_sentence)
print(zh_sentence)

print('------------------000')
exit(0)

print(output_sentence)
