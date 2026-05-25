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
########## Load Bert Embedding ##########
import os
from kashgari.embeddings import BertEmbedding
from kashgari.tokenizers import BertTokenizer

bert_embedding = BertEmbedding('bert-base-chinese')

tokenizer = BertTokenizer.load_from_vocab_file(os.path.join('bert-base-chinese', 'vocab.txt'))
sentences_tokenized = [tokenizer.tokenize(s) for s in sentences]

print(sentences_tokenized)

"""
The sentences will become tokenized into:
[
    ['jim', 'henson', 'was', 'a', 'puppet', '##eer', '.'],
    ['this', 'here', "'", 's', 'an', 'example', 'of', 'using', 'the', 'bert', 'token', '##izer', '.'],
    ['why', 'did', 'the', 'chicken', 'cross', 'the', 'road', '?']
]
"""

train_x, train_y = sentences_tokenized[:2], labels[:2]
validate_x, validate_y = sentences_tokenized[2:], labels[2:]

print(train_x)
print(train_y)

print(validate_x)
print(validate_y)

########## build model ##########
from kashgari.tasks.classification import BiLSTM_Model
model = BiLSTM_Model(bert_embedding)

########## /build model ##########
model.fit(
    train_x, train_y,
    validate_x, validate_y,
    epochs=4,
    batch_size=32
)
# save model
model.save('./test_model')