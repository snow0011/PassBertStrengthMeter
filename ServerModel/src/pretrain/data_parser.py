import argparse
import random
import sys
import os
from typing import List
import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from passbert.tokenizers import PasswordTokenizer
from passbert.snippets import parallel_apply
from passbert.backend import K

class TrainingIterator(object):
    """
    预训练数据迭代器
    """

    def __init__(self, tokenizer, sequence_length):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size
        pass

    def padding(self, sequence, padding_value=None):
        """对单个序列进行补0
        """
        if padding_value is None:
            padding_value = self.token_pad_id

        sequence = sequence[:self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        return sequence + [padding_value] * padding_length

    def sentence_process(self, text):
        """单个文本的处理函数，返回处理后的instance
        """
        raise NotImplementedError

    def paragraph_process(self, texts, starts, ends, paddings):
        """单个段落（多个文本）的处理函数
        说明：texts是单句组成的list；starts是每个instance的起始id；
              ends是每个instance的终止id；paddings是每个instance的填充id。
        做法：不断塞句子，直到长度最接近sequence_length，然后padding。
        """
        instances, instance = [], [[start] for start in starts]
        for text in texts:
            # 处理单个句子
            sub_instance = self.sentence_process(text)
            q = [i for i in sub_instance]
            sub_instance = [i[:self.sequence_length - 2] for i in sub_instance]
            for _a, _b in zip(q, sub_instance):
                for _i, _j in zip(_a, _b):
                    assert _i == _j
            new_length = len(instance[0]) + len(sub_instance[0])

            # 如果长度即将溢出
            if new_length > self.sequence_length - 1:
                # 插入终止符，并padding
                complete_instance = []
                for item, end, pad in zip(instance, ends, paddings):
                    item.append(end)
                    item = self.padding(item, pad)
                    complete_instance.append(item)
                # 存储结果，并构建新样本
                instances.append(complete_instance)
                instance = [[start] for start in starts]

            # 样本续接
            for item, sub_item in zip(instance, sub_instance):
                item.extend(sub_item)

        # 插入终止符，并padding
        complete_instance = []
        for item, end, pad in zip(instance, ends, paddings):
            item.append(end)
            item = self.padding(item, pad)
            complete_instance.append(item)

        # 存储最后的instance
        instances.append(complete_instance)

        return instances
    pass

    def process(self, corpus, workers=8, max_queue_size=2000):
        iter = []

        def write_to_iterator(instances):
            iter.extend(instances)

        def paragraph_process(texts):
            return self.paragraph_process(texts)

        parallel_apply(
            func=paragraph_process,
            iterable=corpus,
            workers=workers,
            max_queue_size=max_queue_size,
            callback=write_to_iterator,
        )
        return iter


class TrainingIteratorRoBERTa(TrainingIterator):
    def __init__(self, tokenizer, word_segment, mask_rate, sequence_length):
        super(TrainingIteratorRoBERTa, self).__init__(
            tokenizer, sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate
        pass

    def token_process(self, token_id):
        """以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def sentence_process(self, text):
        """单个文本的处理函数
        流程：分词，然后转id，按照mask_rate构建全词mask的序列
              来指定哪些token是否要被mask
        """
        words = self.word_segment(text)
        rands = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(text=word)[1:-1]
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)

            mask_ids.extend(word_mask_ids)
        return [token_ids, mask_ids]

    def paragraph_process(self, texts):
        """
        给原方法补上starts、ends、paddings
        """
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super(TrainingIteratorRoBERTa,
                     self).paragraph_process(texts, starts, ends, paddings)

    pass


def read_training(f_training: str, splitter=' ') -> List[List[str]]:
    training_list = []
    with open(f_training, 'r') as fin:
        for line in fin:
            line = line.strip('\r\n')
            chunks = line.split(splitter)
            training_list.append(chunks)
            pass
        pass
    return training_list


def shuffle_training(training_list: List[List[str]], count: int = 10):
    return random.shuffle(training_list)


def roberta_parser_wrapper(tokenizer, sequence_len, mask_rate):
    TD = TrainingIteratorRoBERTa(
        tokenizer=tokenizer, word_segment=lambda x: x, mask_rate=mask_rate, sequence_length=sequence_len
    )

    def parser(corpus, span):
        instances = TD.process(corpus=corpus)
        xs, ys = [], []
        for idx, instance in enumerate(instances):
            
            # features = {k: tf.train.Feature(int64_list=tf.train.Int64List(value=v)) for k, v in zip(['token_ids', 'mask_ids'], instance)}
            # tf_features = tf.train.Features(feature=features)
            # tf_example = tf.train.Example(features=tf_features)
            token_ids = tf.convert_to_tensor(instance[0], dtype=tf.float64)
            mask_ids = tf.convert_to_tensor(instance[1], dtype=tf.float64)
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            is_masked = K.not_equal(mask_ids, 0)
            masked_token_ids = K.switch(is_masked, mask_ids - 1, token_ids)
            x = {
                'Input-Token': masked_token_ids,
                'Input-Segment': segment_ids,
                'token_ids': token_ids,
                'is_masked': K.cast(is_masked, K.floatx()),
            }
            y = {
                'mlm_loss': K.zeros([1]),
                'mlm_acc': K.zeros([1]),
            }
            xs.append(x)
            ys.append(y)
            if len(xs) == span:
                # print(f"current index is {idx}, yield {span} items")
                yield tf.data.Dataset((xs, ys))
                yield np.array(xs), np.array(ys)
                xs, ys = [], []
            pass
        if len(xs) > 0:
            yield np.array(xs), np.array(ys)

    return parser


def default_wrapper():
    training_list = read_training(
        "/disk/cw/nlp-guessing/Prepassword/data/toytrain.txt")
    tokenizer = PasswordTokenizer()
    seq_len = 32
    roberta_parser = roberta_parser_wrapper(tokenizer=tokenizer, sequence_len=seq_len, mask_rate=0.15)
    items = roberta_parser(corpus=[training_list], span=3)
    for xs, ys in roberta_parser(corpus=[training_list], span=3):
        print(len(xs))
        pass
    pass


def wrapper():
    cli = argparse.ArgumentParser("Data parser")
    cli.add_argument('-i', '--training', dest='training', type=str, required=True,
                     help='The training file')
    cli.add_argument("-f", '--formats', dest='formats', type=str, choices=['roberta', 'gpt'], nargs='+',
                     help="The format of the preprocessed training file")
    cli.add_argument("-s", '--save', dest='save', type=str,
                     help="save preprocessed training file here")
    cli.add_argument('-d', '--dict', dest='dict',
                     type=str, help='the dictionary used')
    # args = cli.parse_args()
    training_list = read_training(
        "/disk/cw/nlp-guessing/Prepassword/data/toytrain.txt")
    tokenizer = PasswordTokenizer()
    seq_len = 32
    roberta_parser = roberta_parser_wrapper(tokenizer=tokenizer, sequence_len=seq_len, mask_rate=0.15)
    items = roberta_parser(corpus=[training_list], span=3)

    print(items)

    pass


if __name__ == '__main__':
    default_wrapper()
    pass
