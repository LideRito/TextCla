sentences = [
    "Jim Henson was a puppeteer.",
    "This here's an example of using the BERT tokenizer.",
    "Why did the chicken cross the road?"
            ]

sentences = [
    "我很喜欢猫。",
    "这是一个bert tokenizer的示例。",
    "鸡为什么要过桥？"
            ]

labels = [
    "class1",
    "class2",
    "class1"
]

def read_file(path):
    tokenizer = BertTokenizer.load_from_vocab_file(os.path.join('long-bert', 'vocab.txt'))
    lines = open(path, 'r', encoding='utf-8').read().splitlines()
    x_list = []
    y_list = []
    for line in tqdm.tqdm(lines):
        rows = line.split('\t')
        if len(rows) >= 2:
            y_list.append(rows[0])
            x_list.append(tokenizer.tokenize(str(rows[1:]))[2:-2])
        else:
            print(rows)
    return x_list, y_list

########## Load Bert Embedding ##########
import os
import tqdm
from kashgari.embeddings import TransformerEmbedding
from kashgari.tokenizers import BertTokenizer
import random as rd
import numpy as np

model_folder = 'long-bert'
checkpoint_path = os.path.join(model_folder, 'bert.ckpt')
config_path = os.path.join(model_folder, 'config.json')
vocab_path = os.path.join(model_folder, 'vocab.txt')

bert_embedding = TransformerEmbedding(vocab_path, config_path, checkpoint_path,bert_type='long_bert')

# tokenizer = BertTokenizer.load_from_vocab_file(os.path.join('long-bert', 'vocab.txt'))
sentences, labels = read_file('cnews/cnews.val.txt')
# print(sentences)
sentences_tokenized = sentences

# print(sentences_tokenized)

train_x, train_y = sentences_tokenized[:4000], labels[:4000]
validate_x, validate_y = sentences_tokenized[4000:], labels[4000:]

train_x = np.array(train_x)
train_y = np.array(train_y)

state = np.random.get_state()
np.random.shuffle(train_x)

np.random.set_state(state)
np.random.shuffle(train_y)

train_x = train_x.tolist()
train_y = train_y.tolist()

print(train_x[:2])
print(train_y[:2])

# print(validate_x)
# print(validate_y)

########## build model ##########
from kashgari.tasks.classification import BiLSTM_Model
model = BiLSTM_Model(bert_embedding)

########## /build model ##########
model.fit(
    train_x, train_y,
    validate_x, validate_y,
    epochs=4,
    batch_size=2
)
# save model
model.save('./test_model')