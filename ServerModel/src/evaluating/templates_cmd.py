"""
This file read the templates and converts them to the format of BERT input.
The template is from command.
For example, input template 'p__sword', the model should output the first 10 passwords.
"""
from collections import defaultdict
import os
from re import T
import sys
import argparse
import numpy as np
from itertools import product
from typing import Dict, Tuple, Set, List
from tqdm import tqdm
import heapq
from functools import reduce

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import passbert.tokenizers as pt
from passbert.snippets import to_array
from passbert.models import build_transformer_model

def multiply(array):
    return reduce(lambda x,y:x*y, array)

def load_bert(config_path:str, checkpoint_path:str):
    # dict_path = '/disk/cw/nlp-guessing/bert/bert_base_dir/vocab.txt'
    tokenizer = pt.PasswordTokenizer()
    bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)
    return tokenizer, bert

class PriorityQueue: 
    def __init__(self):
        self._queue = []
        self._index =0

    def push(self, item, priority):
        # 传入两个参数，一个是存放元素的数组，另一个是要存储的元素，这里是一个元组。
        # 由于heap内部默认有小到大排，所以对priority取负数
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)
    
    def size(self):
        return len(self._queue)

def to_letter_prob_list(prob_array, tokenizer=pt.PasswordTokenizer()):
    queue = []
    for i, prob in enumerate(prob_array[0]):
        queue.append((tokenizer.id_to_token(i), prob))
    return sorted(queue, key=lambda x:x[1],reverse=True)

def valid_seq(seq, max_token_id=99):
    for i in seq:
        if i >= max_token_id:
            return False
    return True

def fetch_replacements(template: Tuple, bert, tokenizer, mask_in_template: str = '\t'):
    token_ids = []
    masked_indices = []
    wrappered_template = [tokenizer._token_start, *template, tokenizer._token_end]
    for idx, token in enumerate(wrappered_template):
        if token == mask_in_template:
            got_id = tokenizer._token_mask_id
            masked_indices.append(idx)
        else:
            got_id = tokenizer.token_to_id(token)
        token_ids.append(got_id)
    segment_ids = [0] * len(wrappered_template)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    probas = bert.predict([token_ids, segment_ids])[0]
    masked_values = []
    for idx in masked_indices:
        tar = probas[idx:idx+1]
        masked_values.append(to_letter_prob_list(tar, tokenizer))
    queue = PriorityQueue()
    queue.push([0 for _ in masked_values], multiply([x[0][1] for x in masked_values]))
    for _ in range(10):
        if queue.size() == 0:
            break
        item = queue.pop()
        if valid_seq(item[-1]):
            pwd = [x for x in template]
            for i,j in enumerate(masked_indices):
                pwd[j-1] = masked_values[i][item[-1][i]][0]
            print("".join(pwd), -item[0])
            next_arr = [item[-1][:] for _ in range(len(masked_indices))]
            for  i in range(len(masked_indices)):
                next_arr[i][i] += 1
                priority = multiply([masked_values[m][n][1] for m,n in enumerate(next_arr[i])])
                queue.push(next_arr[i], priority)

def wrapper(**kwargs):
    config_path = kwargs.get("config_path", '/disk/yjt/PasswordSimilarity/model/bert/csdn-rockyou/passbert_csdnroc_model.json')
    checkpoint_path= kwargs.get("checkpoint_path", '/disk/yjt/PasswordSimilarity/model/bert/rockyou/roc-coverted.ckpt')
    tokenizer, bert = load_bert(config_path=config_path, checkpoint_path=checkpoint_path)
    mask_in_template = '\t'
    bert.summary()
    while(True):
        pwd = input("Template: ")
        pwd = [ch if ch != '_' else mask_in_template for ch in pwd]
        fetch_replacements(pwd, bert, tokenizer, mask_in_template)
    pass


def wrapper_with_cli():
    cli = argparse.ArgumentParser("Enumerating password candidates based on templates")
    cli.add_argument('--config', dest='config', type=str, default='/disk/yjt/PasswordSimilarity/model/bert/csdn-rockyou/passbert_csdnroc_model.json', help='config path of bert')
    cli.add_argument('--checkpoint', dest='checkpoint', type=str, default='/disk/yjt/PasswordSimilarity/model/bert/rockyou/roc-coverted.ckpt', help='checkpoint path of bert')
    args = cli.parse_args()
    wrapper(config_path=args.config, checkpoint_path=args.checkpoint)
    pass


if __name__ == '__main__':
    wrapper_with_cli()
    pass
