#! -*- coding: utf-8 -*-
# BERT生成口令修改路径
# 这里只用到BERT的encoder
# 输入为口令字符序列
# 输出为口令分类标签
# 

from __future__ import print_function
import glob, re
from keras.layers.core import Lambda
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from passbert.backend import keras, K
from passbert.layers import Loss
from passbert.models import build_transformer_model
from passbert.tokenizers import Tokenizer, load_vocab, PasswordTokenizer
from passbert.optimizers import Adam
from passbert.snippets import sequence_padding, open
from passbert.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

maxlen = 32
batch_size = 1
steps_per_epoch = 10
epochs = 10
# Total label number
label_num = 3
# number of bert layers
layers_num = 2
# 操作padding字符
TRANS_PADDING_ID = 0

# bert配置
config_path = '/disk/yjt/PasswordSimilarity/model/bert/csdn-rockyou/passbert_csdnroc_model.json'
checkpoint_path = '/disk/yjt/PasswordSimilarity/model/bert/csdn-rockyou/passbert_csdnroc_model.ckpt'
trans_dict_path = '/disk/yjt/PasswordSimilarity/json/trans_dict/trans_dict_2idx.json'

pwds = [
    ("password1",2),
    ("password",2),
    ("passwor",2),
    ("passwo",2),
    ("password123",2),
    ("password2",2),
    ("qwerty123",1),
    ("qwerty",1),
    ("qwerty1",1),
    ("Qwerty",1)
]

tokenizer = PasswordTokenizer()

data = []
pbar = tqdm(desc=u'口令加载中', total=len(pwds))

for pwd in pwds:
    data.append(pwd)
    pbar.update(1)

pbar.close()
np.random.shuffle(data)

class data_generator(DataGenerator):
    """数据生成器
    """
    # 生成[token ids, segment ids], [transfermation list]
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        # The text is in [password, transformation list] format. The transformation should be encoded type, like [2,5,7,8,0]
        # Is end marked the end of the data sequence.
        for is_end, text in self.sample(random):
            pwd = text[0]
            label = text[1]
            pwd = [x for x in pwd]
            wrappered_text = [tokenizer._token_start, *pwd, tokenizer._token_end]
            segment_ids = [0] * len(wrappered_text)
            token_ids = [tokenizer.token_to_id(x) for x in wrappered_text]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            # print(pwd)
            # print(to_categorical(trans_list, trans_num).shape)
            # batch_labels.append(padding_number_list(trans_list,value=0,length=len(pwd),num_types=trans_num))
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                # batch_labels = sequence_padding(batch_labels)
                # batch_labels = to_categorical(batch_labels, num_classes=trans_num)
                # print(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


model = build_transformer_model(
    config_path,
    checkpoint_path
)

output = model.output
output = Dense(128,activation='tanh')(output)
output = Dense(label_num,activation='softmax')(output)
output = Lambda(lambda x: x[:, 0])(output)

# output = [ keras.layers.Lambda(lambda x: x[:, 0])(out) for out in output]

# output = CrossEntropy(1)(model.inputs + model.outputs)

def my_loss(y_true, y_pred):
    y_true = K.print_tensor(y_true, message='y_true = ')
    y_pred = K.print_tensor(y_pred, message='y_pred = ')
    return K.sparse_categorical_crossentropy(y_true, y_pred)

def my_sparse_loss_metric(y_true, y_pred):
    print(y_true)
    print(y_pred)
    return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


model = Model(model.input, output)
model.compile(
    loss=my_loss,
    optimizer=Adam(1e-4),
    metrics=['sparse_categorical_accuracy',my_sparse_loss_metric]
)
model.summary()

def predict_pwd(pwd):
    text = [tokenizer._token_start, *pwd, tokenizer._token_end]
    token_ids = tokenizer.tokens_to_ids(text)
    # token_ids = np.concatenate([token_ids, output_ids], 1)
    # segment_ids = np.zeros_like(token_ids)
    segment_ids = [0] *len(token_ids)
    results = model.predict([[token_ids], [segment_ids]])
    print(results)
    return results

class PathGenerator(AutoRegressiveDecoder):
    """
    Generate the path from given passwords
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids = inputs[0]
        # token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.zeros_like(token_ids)
        return self.first_token(model).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        text = [tokenizer._token_start, *text, tokenizer._token_end]
        token_ids = tokenizer.tokens_to_ids(text)
        # token_ids, _ = tokenizer.encode(text)
        results = self.predict([token_ids],[],np.zeros(1))
        print(results)
        # for p in results[0][0,:]:
        #     print(p)
        # print(results)
        # results = [np.argmax(p) for p in results[0][0,:]]
        print(results)
        return text

path_generator = PathGenerator(start_id=None, end_id=0, maxlen=maxlen)

# class StoryCompletion(AutoRegressiveDecoder):
#     """基于随机采样的故事续写
#     """
#     @AutoRegressiveDecoder.wraps(default_rtype='probas')
#     def predict(self, inputs, output_ids, states):
#         token_ids = inputs[0]
#         token_ids = np.concatenate([token_ids, output_ids], 1)
#         segment_ids = np.zeros_like(token_ids)
#         return self.last_token(model).predict([token_ids, segment_ids])

#     def generate(self, text, n=1, topp=0.95):
#         token_ids, _ = tokenizer.encode(text)
#         results = self.random_sample([token_ids[:-1]], n, topp=topp)  # 基于随机采样
#         return [text + tokenizer.decode(ids) for ids in results]


# story_completion = StoryCompletion(
#     start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
# )


def just_show():
    # ,
    test_list = ["password1","password","qwerty","qwerty123","QWERTY","QwErTy","P@ssword","drowssap"]
    for s in test_list:
        t = predict_pwd(s)
        # t = path_generator.generate(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % (t))


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save_weights('./best_model.weights')
        # 演示效果
        # just_show()


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(data, batch_size)

    just_show()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

    just_show()
