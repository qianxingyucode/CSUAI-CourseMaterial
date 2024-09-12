import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import  AutoTokenizer  #BertTokenizer,
import torch.optim as optim
#from pytorch_transformers import BertModel
from transformers import AlbertModel #AlbertTokenizer,
from torchvision import transforms #, models  datasets,
from efficientnet_pytorch import EfficientNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readTxtFile(fn): #读指定文件的内容
    fg = open(fn, 'r', encoding='gb18030')  #读中文文本
    text = list(fg)
    assert len(text)==1
    text = text[0]
    text = text.replace('\n','')
    return text

def get_txt_img_lb(path,txtname): #获取由“文本-图像路径-类别”构成的数据集
    fn = path+'\\'+txtname
    fg = open(fn, encoding='utf-8')
    samples = []
    for line in fg:
        line = line.strip()
        if 'ID' in line:
            continue
        file_id, label = line.split(',')
        text_path = path + '\\data\\' + file_id + '.txt'
        img_path = path + '\\data\\' + file_id + '.jpg'
        text = readTxtFile(text_path)
        item = (text,img_path,label)
        samples.append(item )
    return samples

tokenizer = AutoTokenizer.from_pretrained('albert-base-v2') #执行该语句要比较久的时间albert-base-v2
tsf =  transforms.Compose([transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class MyDataSet(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, img_path, label = self.samples[idx]
        text_list = [text]
        #索引编码：
        txtdata = tokenizer.batch_encode_plus(batch_text_or_text_pairs=text_list,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=128,  # 2022.7.29 由于sg数据训练报错 超出索引范围 怀疑max过长导致超出
                                       return_tensors='pt',
                                       return_length=True)
        input_ids = txtdata['input_ids']
        token_type_ids = txtdata['token_type_ids']
        attention_mask = txtdata['attention_mask']

        img = Image.open(img_path)
        if img.mode != 'RGB':
            print('不是RGB图像！')
            exit(0)
        img = tsf(img)  #改变形状为torch.Size([3, 224, 224])
        label = int(label)

        #                    torch.Size([128])        torch.Size([3, 224, 224])         int
        return input_ids[0],token_type_ids[0],attention_mask[0],        img,          label


bert_model = AlbertModel.from_pretrained('albert-base-v2', \
                    cache_dir="./AlBert_model").to(device)
......  # 【此处为本例部分核心代码，已省略，完整代码见教材P293；建议读者手工输入核心代码并进行调试，这样方能领会其含义】

effi_model._fc = nn.Linear(2560, 768) #修改预训练模型

class Multi_Model(nn.Module):  #定义深度神经网络模型类
    def __init__(self):
        super().__init__()
        self.bert_model = bert_model
        self.effi_model = effi_model
        self.fc = nn.Linear(768 + 768, 3)
    def forward(self,data):
        input_ids, token_type_ids, attention_mask, img, _ = data
        input_ids, token_type_ids, attention_mask, img = input_ids.to(device), \
                                 token_type_ids.to(device), attention_mask.to(device), img.to(device)


        #torch.Size([8, 128]) torch.Size([8, 128]) torch.Size([8, 128]) torch.Size([8, 3, 224, 224])

        outputs = self.bert_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)
        text_feature = outputs[1]  #文本的特征，形状为torch.Size([8, 768])
        effi_outputs = self.effi_model(img)  #图像的特征，形状为torch.Size([8, 768])
        cat_feature = torch.cat([text_feature, effi_outputs], -1)  # 采用拼接融合方式，cat_feature的形状为torch.Size([8, 1536])
        out = self.fc(cat_feature)  # torch.Size([16, 3])
        return out

def train(model:Multi_Model, data_loader):  #对模型进行训练
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6, \
amsgrad=False)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10,\
                          T_mult=1, eta_min=1e-6, last_epoch=-1)
    criterion = nn.CrossEntropyLoss()
    lr = scheduler.get_last_lr()[0]
    print('epochs :0   lr:{}'.format(lr))
    print('训练中..........')
    epochs = 31
    for ep in range(epochs):
        for k,data in enumerate(data_loader):
            input_ids, token_type_ids, attention_mask, img, label = data
            label = label.to(device)
            pre_y = model(data)
            loss = criterion(pre_y, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        if not ep + 1 == epochs:
            print('epochs :{}   lr:{:.6f}'.format(ep + 1, lr))
        if ep % 5 == 0:  #每5轮循环保存一次模型参数
            torch.save({'model_state_dict': model.state_dict()}, f'multi_model_new.pt')
            check_point = torch.load(f'multi_model_new.pt')
            model.load_state_dict(check_point['model_state_dict'])
    torch.save({'model_state_dict': model.state_dict()}, f'multi_model_new.pt')
    print('训练完毕！')
    return None




def getAccOnadataset(model:Multi_Model, data_loader): #测试模型的准确率
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input_ids, token_type_ids, attention_mask, img, label = data
            label = label.to(device)
            pre_y = multi_model(data)
            pre_y = torch.argmax(pre_y, dim=1)
            t = (pre_y == label).long().sum()
            correct += t
        correct = 1. * correct / len(data_loader.dataset)
    model.train()
    return correct.item()



if __name__ == '__main__':
    batch_size = 8
    path = r'.\data\multimodal-cla'
    samples_train = get_txt_img_lb(path, 'ID-label-train.txt')  # 读取训练集,3609
    samples_test = get_txt_img_lb(path, 'ID-label-test.txt')  # 读取测试集,902

    # 实例化训练集和测试集
    train_dataset = MyDataSet(samples_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MyDataSet(samples_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    torch.save(train_loader, 'train_loader')
    torch.save(test_loader, 'test_loader')

    '''
    '''
    # 3609+902 = 4511
    train_loader = torch.load('train_loader')
    test_loader = torch.load('test_loader')

    print(len(train_loader.dataset))
    print(len(test_loader.dataset))

    multi_model = Multi_Model().to(device)
    train(multi_model, train_loader)  # 对模型进行训练

    print('测试中..........')
    check_point = torch.load(f'multi_model_new.pt')  # 获取已训练的最好模型的参数
    multi_model.load_state_dict(check_point['model_state_dict'])
    acc_test = getAccOnadataset(multi_model, test_loader)
    print('在测试集上的准确率：{:.1f}%'.format(acc_test*100))
