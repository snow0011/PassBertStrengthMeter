#! -*- coding: utf-8 -*-
# BERT生成口令修改路径
# 这里只用到BERT的encoder
# 输入为口令字符序列
# 输出为口令修改路径 edit path
# 

from __future__ import print_function
import glob, re
from keras.layers.core import Lambda
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import os
import argparse
import json
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
from editpath import load_inplace_trans_dict, load_password_letter, recover_inplace_edit
from losses import categorical_focal_loss_new

TRANS_PADDING_ID = 1

def load_passwords(path:str):
    pwds = []
    with open(path, "r") as f:
        for line in f:
            src,tar,label = line.strip('\n\r').split("\t")
            label = json.loads(label)
            pwds.append((src, tar, label))
    return pwds

class Config:
    def __init__(self):
        self.maxlen = 32
        self.batch_size = 1
        self.epochs = 1
        # number of bert layers
        self.layers_num = 2
        # Learning rate
        self.lr = 1e-5
        # bert配置
        self.config_path = '/disk/yjt/PasswordSimilarity/model/bert/csdn-rockyou/passbert_csdnroc_model.json'
        self.checkpoint_path = '/disk/yjt/PasswordSimilarity/model/bert/rockyou/roc_tf.ckpt'
        self.label_path = '/disk/yjt/PasswordSimilarity/data/guessing/temp1.txt'

        # Automic configuration
        self.trans_num = len(load_inplace_trans_dict())
        self.pwds = load_passwords(self.label_path)
        # self.pwds = self.pwds[15:16]
        self.steps_per_epoch = len(self.pwds) // self.batch_size
        # self.steps_per_epoch = 10000

def padding_number_list(num_list, value=0, length=32, num_types=2):
    res = [[value] for _ in range(length)]
    for i, val in enumerate(num_list):
        res[i+1][0] = val
    return [res]

class data_generator(DataGenerator):
    def __init__(self, data, batch_size, trans_num):
        super().__init__(data, batch_size=batch_size)
        self.trans_num = trans_num
    

    # 生成[token ids, segment ids], [transfermation list]
    def __iter__(self, random=False):
        tokenizer = PasswordTokenizer()
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        # The text is in [password, transformation list] format. The transformation should be encoded type, like [2,5,7,8,0]
        # Is end marked the end of the data sequence.
        for is_end, text in self.sample(random):
            pwd = text[0]
            target = text[1]
            trans_list = text[2]
            pwd = [x for x in pwd]
            padding = [tokenizer._token_pad for _ in range(3)]
            wrappered_text = [tokenizer._token_start, *pwd, *padding, tokenizer._token_end]
            segment_ids = [0] * len(wrappered_text)
            token_ids = [tokenizer.token_to_id(x) for x in wrappered_text]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            # value = 0 will keep the padding character
            label = padding_number_list(trans_list, value=TRANS_PADDING_ID, length=len(wrappered_text), num_types=self.trans_num)
            label = to_categorical(label, self.trans_num)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)

                # print(batch_token_ids, batch_segment_ids, batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

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

class ModelWrapper:
    def __init__(self, config:Config):
        self.config = config
        # Build model
        self.model = self.load_model()
        self.evaluator = Evaluator()
        self.tokenizer = PasswordTokenizer()
    
    def predict(self, text):
        text = [self.tokenizer._token_start, *text, self.tokenizer._token_end]
        token_ids = self.tokenizer.tokens_to_ids(text)
        segment_ids = [0 for _ in range(len(token_ids))]
        results = self.model.predict([[token_ids], [segment_ids]])
        print(results.shape)
        results = [np.argmax(p) for p in results[0,:,:]]
        return results

    def load_model(self):
        model = build_transformer_model(
            self.config.config_path,
            self.config.checkpoint_path
        )
        output_layer = 'Transformer-%s-FeedForward-Norm' % (self.config.layers_num-1)
        output = model.get_layer(output_layer).output
        output = Dense(128,activation='tanh')(output)
        output = Dense(self.config.trans_num,activation='softmax')(output)
        model = Model(model.input, output)
        def my_loss(y_true, y_pred):
            # Debug print
            # y_true = K.print_tensor(y_true, message='y_true = ')
            # y_pred = K.print_tensor(y_pred, message='y_pred = ')
            # sparse categorical loss
            # return K.sparse_categorical_crossentropy(y_true, y_pred)
            # alpha = [0.25 for _ in range(self.config.trans_num)]
            # alpha[0] = 0.05
            # use focal loss here
            # loss_fn = categorical_focal_loss_new()
            # return loss_fn(y_true, y_pred)
            return K.categorical_crossentropy(y_true,y_pred)
        def my_sparse_loss_metric(y_true, y_pred):
            return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        model.compile(
            loss=my_loss,
            optimizer=Adam(self.config.lr),
            metrics=['accuracy']
        )
        model.summary()
        return model

    def fit(self, train_generator):
        
        self.model.fit(
            train_generator.forfit(),
            steps_per_epoch=self.config.steps_per_epoch,
            epochs=self.config.epochs,
            callbacks=[self.evaluator]
        )

def just_show(model:ModelWrapper):
    # ,"1234","PASS","QWER"
    test_list = [
        "qwertyuiop",
        "1107VZ",
        "1q2w3e4r5t",
        "a282828"
        ]
    # test_list = [u"password",u"qwerty","password1","qwerty123"]
    for s in test_list:
        t = model.predict(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % (t))

def main():
    config = Config()
    pwds = config.pwds

    # Load data and data preprocess.
    data = []
    pbar = tqdm(desc=u'口令加载中', total=len(pwds))
    for pwd in pwds:
        data.append(pwd)
        pbar.update(1)
    pbar.close()

    np.random.shuffle(data)

    train_generator = data_generator(data, config.batch_size, config.trans_num)
    model = ModelWrapper(config)
    model.fit(train_generator)
    just_show(model)


if __name__ == '__main__':

    main()
