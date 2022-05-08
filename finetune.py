import tensorflow as tf
import os as os
import numpy as np
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split

from torchtext.legacy import data, datasets
import torchtext

# 设置cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1234
torch.manual_seed(SEED)  # 为cpu设置随机种子
torch.cuda.manual_seed(SEED)  # 为gpu设置随机种子
torch.backends.cudnn.deterministic = True  # 提升一点训练速度，没有额外开销
modelpath = r""

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
sys.stdout = Logger('EmbeddingBag_IMDB.log', sys.stdout)


Batch_Size = 64
N_epochs = 10
Dropout = 0.2
LR = 0.1
Emb_Dim = 100
Hidden_Dim = 768
Output_dim = 1

def load_data():
    """
    加载IMDB数据，
    :return: 返回 data_iter, valid_iter, test_iter格式的数据。
    """
    # 使用en_core_web_sm作为tokenizer的语言，可以使用en_core_web_trf或者large来提升token序列化，从而提升模型acc。
    # 按照“常理”，将batch放在第一个纬度。
    TEXT = data.Field(tokenize=tokenizer, tokenizer_language='en_core_web_sm',
                      batch_first=True)
    LABEL = data.LabelField(dtype=torch.float)
		# load和split数据。这个数据需要下载。
    train_data, test_data = datasets.IMDB.splits(root='../.data',
                                                 text_field=TEXT,
                                                 label_field=LABEL)
    print(train_data[1])
    print(len(train_data))
    # 再将train_data分割为train和valid的常规操作。
    train_data, valid_data = train_data.split(split_ratio=0.7,
                                              random_state=random.seed(SEED))
    print('build vocab...')
    # 定义build_vocab使用的向量，这个就相当于是别人训练好的非常精炼的词向量，直接用在我们的数据的单词，这样我们就不需要从头训练。
    # 对了，这条解释是我猜的。。。:)。
    # 这行代码的作用就是使用之前下载的glove数据，因为glove数据很大，每次下载很不方便。
    # 如果使用缓存数据，直接将下一行的 vectors='glove.6B.100d'替换为vectors=vectors。
    vectors = torchtext.vocab.Vectors(name='glove.6B.100d.txt', cache='../.vector_cache/')
    # traindata已经有数据的所有内容，所以直接用traindata构建vocabulary。这里的maxsize可选可不选。
    TEXT.build_vocab(train_data,
                     vectors='glove.6B.100d',
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
		# 创建torch需要的iterator形式。将数据在gpu上训练。
    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        datasets=(train_data, valid_data, test_data),
        batch_size=Batch_Size,
        device=device)

    return train_iter, valid_iter, test_iter, TEXT, LABEL


tokenizer = 'spacy'
train_iter, valid_iter, test_iter, TEXT, LABEL = load_data()

class Bertclassifier(nn.Module):
    """
    使用一个embedding bag层对每个batch的数据进行embedding和平均操作，
    nn.embedding bag 类似于 nn.embedding + mean()，
    最后添加两个全连接层。
    """
    
    def __init__(self, emb_dim, output_dim, path, dropout=0):
        super(EmbeddingBag, self).__init__()
        
				# 这里需要注意的一个是，padding_idx=pad_idx，因为使用的词向量不同，每个vocab的映射就有区别，对于padding的编号可能会不同，使用的哪个tokenizer，就要使用那个tokenizer的 pad_id/unk_id/cls_id/sep_id。否则映射的id错误，会直接影响到模型的拟合能力。
        self.bert = torch.load(path)
        self.linear = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
      	# x直接输出使用。使用的embeddingbag()函数，所以这里不需要对emb的数据进行squeeze(1)纬度合并操作。
        x = self.embedding(text)
        out = self.linear(x)
        return out
      
# 剩余初始化参数，将vectors glove的词向量作为初始化参数
vocab_size = len(TEXT.vocab)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = Bertclassifier(vocab_size, Emb_Dim, Output_dim, modelpath, Dropout)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

def binary_acc(predictions, label):
    # predictions是一整个batch的预测
    # 首先对prediction进行sigmoid，
    # 然后通过round()函数对sigmoid的结果进行四舍五入，得到0、1.
    # 这个函数很简单，如果不懂可以使用torch.tensor创建一个一维的数据的predictions和label模拟一下。
    predictions = torch.round(torch.sigmoid(predictions))
    corrects = (predictions == label).float()
    acc = sum(corrects) / len(corrects)

    return acc

def train(model: nn.Module,
          iterator: data.BucketIterator,
          optimizer: optim.Adam,
          criterion: nn.BCEWithLogitsLoss):
    
    epoch_loss = 0
    epoch_acc = 0
    # total_len：保存当前iterator的所有样本数，等价 len(iterator) * batch_size，
    # 但是有时候iterator的最后一个batch不等于batch_size，有的最后一个batch等于30,batch_size等于64，
    # 所以len(iterator) * batch_size不准确，但是对于loss影响很小，所以两种方法均可。
    total_len = 0
    
		# 将模型调整为model.train()模式。具体可搜索model.train()和model.eval()的区别。
    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        
        # [batch_size, output_dim] --> [batch_size]。具体可以在这里写一个print(predictions)，看一下shape，是可以压缩的，而且必须压缩，不然和label的纬度不同没办法比较。
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_acc(predictions, batch.label)

        loss.backward()
        optimizer.step()
				
        # pytorch新版本使用item()，而item()必须乘以batch才是一整个batch的loss。acc同理。
        # 这里有点绕，可以将loss.item()和loss都print()到控制台，便于理解。
        epoch_loss += loss.item() * len(batch.label)
        epoch_acc += acc.item() * len(batch.label)
        total_len += len(batch.label)

    return epoch_loss / total_len, epoch_acc / total_len



def evaluate(model: nn.Module,
             iterator: data.BucketIterator,
             criterion: nn.BCEWithLogitsLoss):

    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    model.eval()
    
    # with torch.no_grad()：我也不知道为什么写这句，大家都写，我也写。
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_acc(predictions, batch.label)

            epoch_loss += loss * len(batch.label)
            epoch_acc += acc * len(batch.label)
            total_len += len(batch.label)
 
    return epoch_loss / total_len, epoch_acc / total_len


def epoch_time(start_time, end_time):
    inter_time = end_time - start_time
    inter_mins = int(inter_time / 60)
    inter_secs = int(inter_time % 60)
    return inter_mins, inter_secs

best_valid_loss = float('inf')  # 定义源loss为无限大
for epoch in range(N_epochs):
    start_time = time.time()
    print('第', epoch, '次循环：')

    train_loss, train_acc = train(model, train_iter, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iter, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'EmbeddingBag_IMDB.pt')

    print(f'Epoch{epoch:02}, time: {epoch_mins}m {epoch_secs}s')
    print(f'\ttrain loss: {train_loss:.3f}, train acc: {train_acc * 100:.2f}%')
    print(f'\t valid loss: {valid_loss:.3f}, valid acc: {valid_acc * 100:.2f}%')

model_test = EmbeddingBag(vocab_size, Emb_Dim, Hidden_Dim, Output_dim, PAD_IDX, Dropout)
model_test.load_state_dict(torch.load('EmbeddingBag_IMDB.pt'))
model_test.eval()
test_loss, test_acc = evaluate(model_test, test_iter, criterion)
print(f'\ttest loss: {test_loss:.3f}, test acc: {test_acc * 100:.2f}%')
