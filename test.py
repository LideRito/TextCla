import tqdm
import jieba
from kashgari.tasks.classification import CNN_LSTM_Model
from kashgari.embeddings import BertEmbedding
from kashgari.tokenizers import BertTokenizer
import os


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

if __name__ == '__main__':
    val_x, val_y = read_file('cnews/cnews.val.txt')

    print(val_x[:2])
    print(val_y[:2])