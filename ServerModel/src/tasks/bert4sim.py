#! -*- coding: utf-8 -*-
# BERT生成口令修改路径
# 这里只用到BERT的encoder
# 输入为口令字符序列
# 输出为口令修改路径 edit path
# 

from __future__ import print_function
import glob, re
from keras.models import load_model
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
from editpath import load_inplace_trans_dict, load_password_letter, recover_inplace_edit, load_reverse_trans_dict
from losses import seq_categorical_focal_loss
import heapq
from functools import reduce
from multiprocessing import Queue, Process, Lock
import datetime


TRANS_PADDING_ID = 0
global_mapper = load_reverse_trans_dict()

def load_passwords(path:str):
    pwds = []
    with open(path, "r") as f:
        src,tar,label = "","",[]
        for line in f:
            try:
                src,tar,label = line.strip('\n\r').split("\t")
            except Exception as e:
                # print("Line: ",line)
                # print("Items: ",line.strip('\n\r').split("\t"))
                continue
            label = json.loads(label)
            pwds.append((src, tar, label))
    return pwds

class Config:
    def __init__(self):
        self.maxlen = 32
        self.batch_size = 128
        self.epochs = 3
        # number of bert layers
        self.layers_num = 4
        # Learning rate
        self.lr = 1e-5
        self.cpu_num = 10
        # bert配置
        self.config_path = ''
        self.checkpoint_path = ''
        self.label_path = ''
        self.model_save = ''
        self.mode = "test" # train, cmd, test
        self.model_load = ''
        self.pwd_pairs = ''
        self.output_csv = ''

        # Automic configuration
        self.trans_num = len(load_inplace_trans_dict())
        if self.mode == "train":
            self.pwds = load_passwords(self.label_path)
        else:
            self.pwds = []
        # self.pwds = self.pwds[:100]
        self.steps_per_epoch = len(self.pwds) // self.batch_size
        # self.steps_per_epoch = 100

def padding_number_list(num_list, value=0, length=32, num_types=2):
    res = [[value] for _ in range(length)]
    for i, val in enumerate(num_list):
        res[i][0] = val
    return res

def get_padding(tokenizer, length=3):
    return [tokenizer._token_start for _ in range(length)]

def get_pwd_pairs(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip('\r\n')
            ss = line.split("\t")
            pwd = ss[0]
            target = ss[1]
            yield (pwd, target)

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
            padding = get_padding(tokenizer)
            wrappered_text = [tokenizer._token_start, *pwd, *padding, tokenizer._token_end]
            segment_ids = [0] * len(wrappered_text)
            token_ids = [tokenizer.token_to_id(x) for x in wrappered_text]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            # value = 0 will keep the padding character
            label = padding_number_list(trans_list, value=TRANS_PADDING_ID, length=len(wrappered_text)-2, num_types=self.trans_num)
            # label = to_categorical(label, self.trans_num)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                # print(batch_token_ids, batch_segment_ids, batch_labels)
                yield [batch_token_ids, batch_segment_ids], [batch_labels]
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, save_fn):
        self.save_fn = save_fn
        self.lowest = 2.5

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            self.save_fn()
            # model.save_weights('./best_model.weights')
            # Save model 
            self.save_fn()
        # 演示效果
        # just_show()

class PriorityQueue: 
    def __init__(self):
        self._queue = []
        self._index =0
        self.unique_set = set()

    def push(self, item, priority):
        # 传入两个参数，一个是存放元素的数组，另一个是要存储的元素，这里是一个元组。
        # 由于heap内部默认有小到大排，所以对priority取负数
        if tuple(item) not in self.unique_set:
            heapq.heappush(self._queue, (-priority, self._index, item))
            self._index += 1
            self.unique_set.add(tuple(item))

    def pop(self):
        return heapq.heappop(self._queue)
    
    def size(self):
        return len(self._queue)

def to_sorted_prob_list(prob_array):
    queue = []
    for i, prob in enumerate(prob_array):
        queue.append((i, prob))
    return sorted(queue, key=lambda x:x[1],reverse=True)

def multiply(array):
    if(len(array) == 0):
        return 0
    if(len(array) == 1):
        return array[1]
    return reduce(lambda x,y:x*y, array)

class MultiTasks:
    def __init__(self, cpu_num:int, task_fn, output_path):
        self._processors = []
        self.lock = Lock()
        self.queue = Queue(2 * cpu_num)
        self.task_fn = task_fn
        self.cpu_num = cpu_num
        self.output = open(output_path, "w")
        for _ in range(cpu_num):
            p = Process(target=self.worker, args=(self.lock, self.queue, self.task_fn))
            p.start()
            self._processors.append(p)

    def record(self, values, lock):
        with lock:
            for item in values:
                try:
                    item = [str(i) for i in item]
                    value = "\t".join(item)
                    self.output.write(f"{value}\n")
                except TypeError as e:
                    # print(item)
                    # continue
                    raise e
            self.output.flush()
        del values

    def worker(self, lock, queue, task_fn):
        print("Process start")
        while True:
            item = queue.get()
            if len(item) == 0:
                break
            # res is a array
            records = []
            for value in item:
                # 参数解包
                res = task_fn(*value)
                records.append(res)
            self.record(records, lock)
        print("Worker Done")

    def getQueue(self):
        return self.queue

    def done(self):
        for i in range(self.cpu_num):
            self.queue.put([])
        for i in range(self.cpu_num):
            self._processors[i].join()
        self.output.close()

def top_k_password(src, target, probs, top_k=1000):
    results = probs
    top_list = []
    pos_list = list(range(len(src)-2))
    for i in pos_list:
        probs = results[i]
        top_list.append(to_sorted_prob_list(probs))
    queue = PriorityQueue()
    queue.push([0 for _ in top_list], multiply([x[0][1] for x in top_list]))
    path_list = []
    for _ in range(top_k):
        if queue.size() == 0:
            break
        item = queue.pop()
        pass_item = [top_list[i][item[-1][i]][0] for i in pos_list]  #item[-1]
        path_list.append((pass_item, -item[0]))
        next_arr = [item[-1][:] for _ in pos_list]
        for i in pos_list:
            next_arr[i][i] += 1
            priority = multiply([top_list[m][n][1] for m,n in enumerate(next_arr[i])])
            queue.push(next_arr[i], priority)
    del queue
    result = (src, target, -1, -1)
    for index, item in enumerate(path_list):
        path, prob = item
        path_op = [global_mapper[x] for x in path]
        guess = recover_inplace_edit(src, path_op)
        if guess == target:
            result = (src, target, prob, index)
            break
    return result

class ModelWrapper:
    def __init__(self, config:Config):
        self.config = config
        self.build_custom_objects()
        # Build model
        if config.mode == "train":
            self.model = self.load_model()
        else:
            self.model = self.load_finetune()
        self.evaluator = Evaluator(self.save)
        self.tokenizer = PasswordTokenizer()
        self.custom_objects = {}

    def build_custom_objects(self):
        def my_loss(y_true, y_pred):
            # Debug print
            # y_true = K.print_tensor(y_true, message='y_true = ')
            # y_pred = K.print_tensor(y_pred, message='y_pred = ')
            # sparse categorical loss
            masked = K.argmax(y_true, axis=2)
            return K.sparse_categorical_crossentropy(y_true, y_pred)
        
        def my_loss_metric(y_true, y_pred):
            # y_true = K.print_tensor(y_true, message='y_true = ')
            # y_pred = K.print_tensor(y_pred, message='y_pred = ')
            return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        def my_loss_metric1(y_true, y_pred):
            masked = K.argmax(y_true, axis=2)
            temp1 = K.ones_like(masked)
            temp = K.zeros_like(masked)
            y_pred = tf.where(K.equal(masked, temp), y_true, y_pred)
            return keras.metrics.categorical_accuracy(y_true, y_pred)

        self.custom_objects = {
            'my_loss':my_loss,
            'my_loss_metric':my_loss_metric
        }
    
    def predict(self, text):
        padding = get_padding(self.tokenizer)
        text = [self.tokenizer._token_start, *text, *padding, self.tokenizer._token_end]
        token_ids = self.tokenizer.tokens_to_ids(text)
        segment_ids = [0 for _ in range(len(token_ids))]
        results = self.model.predict([[token_ids], [segment_ids]])
        # print(results.shape)
        results = [np.argmax(p) for p in results[0,:,:]]
        return results

    def query(self, text):
        token_ids, segment_ids = self.encode_text([text])
        padding = get_padding(self.tokenizer)
        text = [self.tokenizer._token_start, *text, *padding, self.tokenizer._token_end]
        results = self.model.predict([token_ids, segment_ids])
        return results[0]

    def encode_text(self, pwds):
        batch_tokens = []
        batch_segments = []
        for pwd in pwds:
            padding = get_padding(self.tokenizer)
            text = [self.tokenizer._token_start, *pwd, *padding, self.tokenizer._token_end]
            token_ids = self.tokenizer.tokens_to_ids(text)
            segment_ids = [0 for _ in range(len(token_ids))]
            batch_tokens.append(token_ids)
            batch_segments.append(segment_ids)
        return batch_tokens, batch_segments

    def batch_top_k(self, texts):
        assert len(texts) <= 1024
        batch_tokens, batch_segments = self.encode_text(texts)
        batch_tokens = sequence_padding(batch_tokens)
        batch_segments = sequence_padding(batch_segments)
        results = self.model.predict([batch_tokens, batch_segments], batch_size=len(texts))
        return results

    def top_k(self, text, top_k=10):
        token_ids, segment_ids = self.encode_text([text])
        padding = get_padding(self.tokenizer)
        text = [self.tokenizer._token_start, *text, *padding, self.tokenizer._token_end]
        results = self.model.predict([token_ids, segment_ids])
        results = results[0]
        # print(results.shape)
        # print(token_ids)
        top_list = []
        pos_list = list(range(len(text)-2))
        for i in pos_list:
            probs = results[i]
            top_list.append(to_sorted_prob_list(probs))
        queue = PriorityQueue()
        queue.push([0 for _ in top_list], multiply([x[0][1] for x in top_list]))
        path_list = []
        for _ in range(top_k):
            if queue.size() == 0:
                break
            item = queue.pop()
            pass_item = [top_list[i][item[-1][i]][0] for i in pos_list]  #item[-1]
            path_list.append((pass_item, -item[0]))
            next_arr = [item[-1][:] for _ in pos_list]
            for i in pos_list:
                next_arr[i][i] += 1
                priority = multiply([top_list[m][n][1] for m,n in enumerate(next_arr[i])])
                queue.push(next_arr[i], priority)
        del queue
        return path_list

    def load_model(self):
        model = build_transformer_model(
            self.config.config_path,
            self.config.checkpoint_path
        )


        output_layer = 'Transformer-%s-FeedForward-Norm' % (self.config.layers_num-1)
        output = model.get_layer(output_layer).output
        output = Dense(512,activation='tanh')(output)
        output = Dense(self.config.trans_num,activation='softmax')(output)
        output = Lambda(lambda x: x[:, 1:-1])(output)
        model = Model(model.input, output)
        
        model.compile(
            loss=self.custom_objects['my_loss'],
            optimizer=Adam(self.config.lr),
            metrics=[self.custom_objects['my_loss_metric']]
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

    def save(self):
        self.model.save(self.config.model_save)
        print(f"Model saved in {self.config.model_save}")
        pass

    def load_finetune(self):
        return load_model(self.config.model_load, custom_objects=self.custom_objects)

def just_show(model:ModelWrapper):
    # ,"1234","PASS","QWER"
    test_list = [
        "qwertyuiop",
        "1107VZ",
        "1107vz",
        "1q2w3e4r5t",
        "a282828"
        ]
    # test_list = [u"password",u"qwerty","password1","qwerty123"]
    for s in test_list:
        t = model.predict(s)
        print(u'输入: %s' % s)
        print(u'结果: %s\n' % (t))

def cmd_test(model:ModelWrapper):
    mapper = load_reverse_trans_dict()
    while(True):
        password = input("Password: ")
        paths = model.top_k(password, top_k=100)
        for path, prob in paths:
            path_op = [mapper[x] for x in path]
            target = recover_inplace_edit(password, path_op)
            print(target, prob, path)

def query(model:ModelWrapper, config:Config):
    pairs = get_pwd_pairs(config.pwd_pairs)
    mapper = load_reverse_trans_dict()
    result = []
    pwd_count = 0
    guessed_num = 0
    for pwd, target in pairs:
        pwd_count +=1 
        if pwd_count % 500 == 0:
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Batch {pwd_count}, Crack rate: {guessed_num/pwd_count}')
        guessed = False
        paths = model.top_k(pwd, 1000)
        for index, item in enumerate(paths):
            path, prob = item
            path_op = [mapper[x] for x in path]
            guess = recover_inplace_edit(pwd, path_op)
            if guess == target:
                guessed = True
                guessed_num += 1
                result.append((pwd, target, prob, index))
                break
        if not guessed:
            result.append((pwd, target, -1, -1))
    with open(config.output_csv, "w") as f:
        total_num = len(result)
        guessed_num = 0
        for pwd, target, prob, index in result:
            if index >= 0:
                guessed_num += 1
            f.write(f"{pwd}\t{target}\t{prob}\t{index}\n")
        print(f"Guess rate: {guessed_num/total_num}")

def multi_query(model:ModelWrapper, config:Config):
    pairs = get_pwd_pairs(config.pwd_pairs)
    batch_size = 256
    manager = MultiTasks(config.cpu_num, top_k_password, config.output_csv)
    index = 0
    tasks = []
    for pwd, target in pairs:
        probs = model.query(pwd)
        tasks.append((pwd, target, probs))
        if(len(tasks) % batch_size == batch_size-1):
            manager.getQueue().put(tasks)
            del tasks
            tasks = []
        index += 1
        if index % 1000 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": Batch ", index)
    if(len(tasks)  > 0):
        manager.getQueue().put(tasks)
    manager.done()

def batch_query(model:ModelWrapper, config:Config):
    pairs = get_pwd_pairs(config.pwd_pairs)
    batch_size = 1
    manager = MultiTasks(config.cpu_num, top_k_password, config.output_csv)
    index = 0
    tasks = []
    for pwd, target in pairs:
        if(len(tasks) % batch_size == batch_size-1):
            probs = model.batch_top_k([x[0] for x in tasks])
            items = [(tasks[i][0], tasks[i][1], probs[i]) for i in range(len(tasks))]
            manager.getQueue().put(items)
            del items
            del tasks
            del probs
            tasks = []
        tasks.append((pwd, target))
        index += 1
        if index % 10000 == 0:
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ": Batch ", index)
    if len(tasks) != 0:
        probs = model.batch_top_k([x[0] for x in tasks])
        items = [(tasks[i][0], tasks[i][1], probs[i]) for i in range(len(tasks))]
        manager.getQueue().put(items)
    manager.done()

def cmd_test_old(model:ModelWrapper):
    mapper = load_reverse_trans_dict()
    while(True):
        password = input("Password: ")
        path = model.predict(password)
        path_op = [mapper[x] for x in path]
        target = recover_inplace_edit(password, path_op)
        print(path)
        print(target)

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
    model = ModelWrapper(config)
    if config.mode == "test":
        # batch_query(model, config)
        query(model, config)
        # multi_query(model, config)
    elif config.mode == "train":
        train_generator = data_generator(data, config.batch_size, config.trans_num)
        model.fit(train_generator)
        just_show(model)
        model.save()
    else: # cmd
        cmd_test(model)


if __name__ == '__main__':

    main()
