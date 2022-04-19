"""
PPSM for passbert, referred to as PBSM
"""
import argparse
import os
import string
import sys
import numpy
import math
import heapq
from keras.models import load_model
import tensorflow as tf
import logging
from functools import reduce
from typing import List
import string
# the folder is ``src/``
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
# 自定义层
from passbert.layers import Loss
from passbert.backend import keras, K
# import tensorflowjs as tfjs

def load_password_letter():
    LETTERS = string.ascii_letters
    NUMBERS = string.digits
    SPECIALS = string.punctuation
    SPACE = " "
    return LETTERS + NUMBERS + SPECIALS + SPACE

def load_inplace_trans_dict():
    actions = {}
    # load simple operations
    actions[('k',None)] = len(actions)
    actions[('d',None)] = len(actions)
    letters = load_password_letter()
    for ch in letters:
        actions[('s',ch)] = len(actions)
    # load complex operations
    for ch in letters:
        for ch2 in letters:
            actions[('x',ch+ch2)] = len(actions)
    return actions

def load_reverse_trans_dict():
    mapper = load_inplace_trans_dict()
    ans = {}
    for i,item in mapper.items():
        ans[item] = i
    return ans

def recover_inplace_edit(pwd, path):
    mapper = load_reverse_trans_dict()
    decode_path = [mapper[x] for x in path]
    res = [ch for ch in pwd] + ['', '', '']
    for i, item in enumerate(decode_path):
        if item[0] == 'k':
            continue
        if item[0] == 's' or item[0] == 'x':
            res[i] = item[1]
        if item[0] == 'd':
            res[i] = ''
    return "".join(res).strip(' ')

def multiply(array):		
    if(len(array) == 0):
        return 0
    if(len(array) == 1):
        return array[1]
    return reduce(lambda x,y:x*y, array)

def to_sorted_prob_list(prob_array):
    queue = []
    for i, prob in enumerate(prob_array):
        queue.append((i, prob))
    return sorted(queue, key=lambda x:x[1],reverse=True)

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

class ModelSerializer():
    def __init__(self,
                 weightfile=None):
        self.weightfile = weightfile
        self.model = None
        import passbert.tokenizers as pt
        self.tokenizer = pt.PasswordTokenizer()
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

        self.custom_objects = {
            'my_loss':my_loss,
            'my_loss_metric':my_loss_metric
        }

    def load_model(self):
        logging.info('Loading model weights')
        self.sess = K.get_session()
        tf.saved_model.loader.load(self.sess, ["serve"], "/disk/yjt/BertMeter/PassBertStrengthMeter/model/TPG/savedmodel")
        self.graph = tf.get_default_graph()
        # self.model = load_model(self.weightfile, custom_objects=self.custom_objects)
        logging.info('Done loading model')
        return self.model
    
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
    
    def top_k(self, text, top_k=10):
        pwd = text
        token_ids, segment_ids = self.encode_text([text])
        print(token_ids, segment_ids)
        padding = get_padding(self.tokenizer)
        text = [self.tokenizer._token_start, *text, *padding, self.tokenizer._token_end]
        # results = self.model.predict([token_ids, segment_ids])
        x = self.sess.graph.get_tensor_by_name('Input-Token:0')
        y = self.sess.graph.get_tensor_by_name('Input-Segment:0')
        z = self.sess.graph.get_tensor_by_name('lambda_1/strided_slice:0')
        results = self.sess.run(z, feed_dict={x:token_ids, y: segment_ids})
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
        for path in path_list:
            pp = recover_inplace_edit(pwd, path[0])
            print(pp, path[0],path[1])
        return path_list

def get_padding(tokenizer, length=3):
    return [tokenizer._token_start for _ in range(length)]

def to_array(*args):
    """批量转numpy的array
    """
    results = [numpy.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results

def wrapper():
    bert = ModelSerializer("/disk/yjt/BertMeter/PassBertStrengthMeter/model/TPG/training_nokeep.h5")
    bert.load_model()
    # Test Mode
    while True:
        pwd = input("Password: ")
        res = bert.top_k(pwd, 10)
        # print(res)
    
    # tfjs.converters.save_keras_model(bert.model, "/disk/yjt/BertMeter/PassBertStrengthMeter/model/JS/TPG")
    # print(bert.model.inputs)
    # print(bert.model.outputs)
    # sess = K.get_session()
    # Save bert model as SavedModel format
    # tf.saved_model.simple_save(
    #     sess,
    #     "/disk/yjt/BertMeter/PassBertStrengthMeter/model/TPG/savedmodel",
    #     inputs={'Input-Token': bert.model.inputs[0], 'Input-Segment': bert.model.inputs[1]},
    #     outputs={'lambda_1/strided_slice:0': bert.model.outputs[0]}
    # )
    pass

if __name__ == '__main__':
    wrapper()
