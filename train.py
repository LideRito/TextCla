import tqdm
from kashgari.tasks.classification import CNN_Model
from kashgari.embeddings import TransformerEmbedding
from kashgari.tokenizers import BertTokenizer
import os
import numpy as np

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

def train():
    test_x, test_y = read_file('cnews/cnews.test.txt')
    train_x, train_y = read_file('cnews/cnews.train.txt')
    val_x, val_y = read_file('cnews/cnews.val.txt')

    # 初始化 BERT embedding

    model_folder = 'long-bert'
    checkpoint_path = os.path.join(model_folder, 'bert.ckpt')
    config_path = os.path.join(model_folder, 'config.json')
    vocab_path = os.path.join(model_folder, 'vocab.txt')

    embedding = TransformerEmbedding(vocab_path, config_path, checkpoint_path,bert_type='long_bert')

    # 使用 embedding 初始化模型

    model = CNN_Model(embedding)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    state = np.random.get_state()
    np.random.shuffle(train_x)

    np.random.set_state(state)
    np.random.shuffle(train_y)

    train_x = train_x.tolist()
    train_y = train_y.tolist()

    print('---------------train---------------')
    model.fit(
        train_x, train_y, 
        val_x, val_y, 
        batch_size=2, epochs=3
    )

    print('--------------evaluate-------------')
    model.save('./bert_emb_model')
    model.evaluate(test_x, test_y,batch_size=2)

    

if __name__ == "__main__":
    train()


