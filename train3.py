
########## Load Bert Embedding ##########
import os
import tqdm
from kashgari.embeddings import BertEmbedding
from kashgari.tokenizers import BertTokenizer

def read_file(path):
    tokenizer = BertTokenizer.load_from_vocab_file(os.path.join('bert-base-chinese', 'vocab.txt'))
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


bert_embedding = BertEmbedding('bert-base-chinese')

data_x, data_y = read_file('cnews/cnews.val.txt')

train_x, train_y = data_x[:2], data_y[:2]

validate_x, validate_y = data_x[2:3], data_y[2:3]

# for i in range(4000):
#     data_y[i]=str(i)
# for i in range(4000,5000):
#     validate_y[i-4000]=str(i)

print(train_x)
print(train_y)

print(validate_x)
print(validate_y)

# print(train_x,'\t',train_y)

########## build model ##########
from kashgari.tasks.classification import BiLSTM_Model
model = BiLSTM_Model(bert_embedding)

########## /build model ##########
model.fit(
    train_x, train_y,
    validate_x, validate_y,
    epochs=3,
    batch_size=128
)
# save model
model.save('./test_model')