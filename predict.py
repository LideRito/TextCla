
from kashgari.tasks.classification import CNN_Model
from kashgari.tokenizers import BertTokenizer
from kashgari.embeddings import TransformerEmbedding

import os

file_dir = os.path.dirname(os.path.abspath(__file__))
model_path = 'bert_emb_model'
model_path_abs = os.path.join(file_dir,model_path)
print(model_path_abs)
model = CNN_Model.load_model(model_path_abs)
#loaded_model = kashgari.utils.load_model(model_path_abs)

def predict(text):
    tokenizer = BertTokenizer.load_from_vocab_file(os.path.join(model_path_abs, 'vocab.txt'))
    tmp = tokenizer.tokenize(text)
    #输出分类结果
    res = model.predict(tmp)
    #要根据res输出风险度看你怎么分哪一类风险度高

    #暂时输出分类结果
    return res

if __name__ == "__main__":
    text = input("请输入要判别的文本")
    res = predict(text)
    print(res)
