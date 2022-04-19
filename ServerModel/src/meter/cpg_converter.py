"""
PPSM for passbert, referred to as PBSM
"""
import argparse
import os
import string
import sys
import numpy
import math
from keras.models import load_model

import tensorflow as tf
import logging
from typing import List
# the folder is ``src/``
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from passbert.layers import Loss
from passbert.models import build_transformer_model
from passbert.backend import keras, K

class ModelSerializer():
    def __init__(self,
                 weightfile=None, config_path=None):
        self.weightfile = weightfile
        self.config_path = config_path
        self.model = None
        import passbert.tokenizers as pt
        self.tokenizer = pt.PasswordTokenizer()

    def load_model(self):
        logging.info('Loading model weights')
        self.model = build_transformer_model(config_path=self.config_path, checkpoint_path=self.weightfile, with_mlm=True)
        # self.model = load_model(self.weightfile)
        logging.info('Done loading model')
        return self.model

def needed_items():
    letters = string.ascii_letters
    digits = string.digits
    symbols = string.punctuation
    needed = [*letters, *digits, *symbols]
    return needed

def to_array(*args):
    """批量转numpy的array
    """
    results = [numpy.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results

def eval_pwd(pwd: str, tokenizer, bert, max_len: int, needed_ids: List[int]):
    # generate masked password
    probabilities = []
    for i in range(len(pwd)):
        chr_list = list(pwd)
        chr_list[i] = '\t' #  tokenizer._token_mask
        wrappered = [tokenizer._token_start, *chr_list, tokenizer._token_end]
        # last = max_len - len(wrappered)
        # wrappered += [tokenizer._token_pad] * last
        token_ids = []
        for idx, token in enumerate(wrappered):
            if token == '\t':
                got_id = tokenizer._token_mask_id
                masked_index = idx
            else:
                got_id = tokenizer.token_to_id(token)
            token_ids.append(got_id)
        segment_ids = [0] * len(wrappered)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        probas = bert.predict([token_ids, segment_ids])[0]
        # masked_index = i + 1
        masked_item = pwd[i]
        masked_id = tokenizer.token_dict[masked_item]
        tar = probas[masked_index:masked_index + 1]
        # ps = [math.e ** x for x in tar[0]]
        ps = tar[0]
        # print(ps)
        total = sum(ps[_i] for _i in needed_ids)
        # ps = [ps[i] / total for i in needed_ids]
        # print(f"{ps[masked_id]} / {total} = {ps[masked_id] / total}")
        # print(pwd[i], all(x > 0 for x in ps))
        prob = ps[masked_id] / total
        # print(prob, ps[masked_id])
        # print(pwd, pwd[i], masked_id, prob, sum(ps), ps)
        probabilities.append(prob)
    return probabilities

def wrapper():
    bert = ModelSerializer("/disk/yjt/BertMeter/PassBertStrengthMeter/model/CPG/rockyou2021_tf_medium.ckpt", "/disk/yjt/BertMeter/PassBertStrengthMeter/config/bert_config_medium.json")
    bert.load_model()
    while True:
        pwd = input("Password: ")
        needed = needed_items()
        res = eval_pwd(pwd,  bert.tokenizer, bert.model, 16, [bert.tokenizer.token_dict[token] for token in needed])
        print(res)
    # print(bert.model.inputs)
    # print(bert.model.outputs)
    # sess = K.get_session()
    # Save bert model as SavedModel format
    # tf.saved_model.simple_save(
    #     sess,
    #     "/disk/yjt/BertMeter/PassBertStrengthMeter/model/CPG/savedmodel",
    #     inputs={'Input-Token': bert.model.inputs[0], 'Input-Segment': bert.model.inputs[1]},
    #     outputs={'lambda_1/strided_slice:0': bert.model.outputs[0]}
    # )
    pass

if __name__ == '__main__':
    wrapper()
