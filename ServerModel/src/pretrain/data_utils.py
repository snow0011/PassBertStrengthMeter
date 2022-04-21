#! -*- coding: utf-8 -*-
# 预训练语料构建

import tensorflow as tf
import numpy as np
import os
import sys
import random
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from passbert.tokenizers import PasswordTokenizer
from passbert.backend import K
from passbert.snippets import parallel_apply


os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras


class TrainingDataset(object):
    """预训练数据集生成器
    """

    def __init__(self, tokenizer, sequence_length=512, must_concat=False):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_start_id
        self.token_sep_id = tokenizer._token_end_id
        self.token_mask_id = tokenizer._token_mask_id
        self.vocab_size = tokenizer._vocab_size
        self.must_concat = must_concat

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

        Comments by @ChuanwangWANG
        Here we append a password as long as the length of the vector is less than seq_len (512 by default)
        the format of the final vector is like:
        [cls] *pwd [cls] *pwd [cls] *pwd [sep]
        where *pwd refers to defold pwd, i.e., pwd = [1, 2, 3] and *pwd = 1, 2, 3
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
            # Note that here we set each sequence contains only one password
            # if new_length > 0:
            if new_length > self.sequence_length - 1 or not self.must_concat:
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
                if self.must_concat:
                    item.append(self.token_cls_id)

        # 插入终止符，并padding
        complete_instance = []
        for item, end, pad in zip(instance, ends, paddings):
            item.append(end)
            item = self.padding(item, pad)
            complete_instance.append(item)

        # 存储最后的instance
        instances.append(complete_instance)
        # print("instances = ", instances)
        return instances

    def tfrecord_serialize(self, instances, instance_keys):
        """转为tfrecord的字符串，等待写入到文件
        """
        def create_feature(x):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

        serialized_instances = []
        for instance in instances:
            features = {
                k: create_feature(v)
                for k, v in zip(instance_keys, instance)
            }
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            serialized_instance = tf_example.SerializeToString()
            serialized_instances.append(serialized_instance)

        return serialized_instances

    def process(self, corpus, record_name, workers=8, max_queue_size=2000):
        """处理输入语料（corpus），最终转为tfrecord格式（record_name）
        自带多进程支持，如果cpu核心数多，请加大workers和max_queue_size。
        """
        writer = tf.io.TFRecordWriter(record_name)
        globals()['count'] = 0

        def write_to_tfrecord(serialized_instances):
            globals()['count'] += len(serialized_instances)
            for serialized_instance in serialized_instances:
                writer.write(serialized_instance)

        def paragraph_process(texts):
            instances = self.paragraph_process(texts)
            serialized_instances = self.tfrecord_serialize(instances)
            return serialized_instances

        parallel_apply(
            func=paragraph_process,
            iterable=corpus,
            workers=workers,
            max_queue_size=max_queue_size,
            callback=write_to_tfrecord,
        )

        writer.close()

    @staticmethod
    def load_tfrecord(record_names, batch_size, parse_function):
        """加载处理成tfrecord格式的语料
        """
        if not isinstance(record_names, list):
            record_names = [record_names]

        dataset = tf.data.TFRecordDataset(record_names)  # 加载
        dataset = dataset.map(parse_function, num_parallel_calls=8)  # 解析
        dataset = dataset.repeat()  # 循环
        dataset = dataset.shuffle(batch_size * 1000)  # 打乱
        dataset = dataset.batch(batch_size)  # 成批
        dataset = dataset.prefetch(buffer_size=1)

        return dataset


class TrainingDatasetRoBERTa(TrainingDataset):
    """预训练数据集生成器（RoBERTa模式）
    """

    def __init__(
        self, tokenizer, word_segment, mask_rate=0.15, sequence_length=512, must_concat=False
    ):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
            word_segment是任意分词函数。
        """
        super(TrainingDatasetRoBERTa, self).__init__(
            tokenizer, sequence_length, must_concat=must_concat)
        self.word_segment = word_segment
        self.mask_rate = mask_rate

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
            # print(f"[TrainingDatasetRoberta::sentence_process]: word = {word}, word_tokens = {word_tokens}, word_token_ids = {word_token_ids}")
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)

            mask_ids.extend(word_mask_ids)
        # print(f"text = {text}, words = {words}, token_ids = {token_ids}, mask_ids = {mask_ids}")
        return [token_ids, mask_ids]

    def paragraph_process(self, texts):
        """给原方法补上starts、ends、paddings
        """
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super(TrainingDatasetRoBERTa,
                     self).paragraph_process(texts, starts, ends, paddings)

    def tfrecord_serialize(self, instances):
        """给原方法补上instance_keys
        """
        instance_keys = ['token_ids', 'mask_ids']
        return super(TrainingDatasetRoBERTa,
                     self).tfrecord_serialize(instances, instance_keys)

    @staticmethod
    def load_tfrecord(record_names, sequence_length, batch_size):
        """给原方法补上parse_function
        """
        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
                'mask_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
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
            return x, y

        return TrainingDataset.load_tfrecord(
            record_names, batch_size, parse_function
        )

class TrainingDatasetCPG(TrainingDataset):
    """预训练数据集生成器（RoBERTa模式）
    """

    def __init__(
        self, tokenizer, word_segment, mask_rate=0.50, sequence_length=512
    ):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
            word_segment是任意分词函数。
        """
        super(TrainingDatasetCPG,self).__init__(
            tokenizer, sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate
        print(f"TrainingDatasetCPG record")

    def token_process(self, token_id):
        """以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        # rand = np.random.random()
        # if rand <= 0.8:
        return self.token_mask_id
        # elif rand <= 0.9:
        #     return token_id
        # else:
        #     return np.random.randint(0, self.vocab_size)

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
            # print(f"[TrainingDatasetRoberta::sentence_process]: word = {word}, word_tokens = {word_tokens}, word_token_ids = {word_token_ids}")
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)

            mask_ids.extend(word_mask_ids)
        # print(f"text = {text}, words = {words}, token_ids = {token_ids}, mask_ids = {mask_ids}")
        return [token_ids, mask_ids]

    def paragraph_process(self, texts):
        """给原方法补上starts、ends、paddings
        """
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super(TrainingDatasetCPG,
                     self).paragraph_process(texts, starts, ends, paddings)

    def tfrecord_serialize(self, instances):
        """给原方法补上instance_keys
        """
        instance_keys = ['token_ids', 'mask_ids']
        return super(TrainingDatasetCPG,
                     self).tfrecord_serialize(instances, instance_keys)

    @staticmethod
    def load_tfrecord(record_names, sequence_length, batch_size):
        """给原方法补上parse_function
        """
        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
                'mask_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
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
            return x, y

        return TrainingDataset.load_tfrecord(
            record_names, batch_size, parse_function
        )


class TrainingDatasetGPT(TrainingDataset):
    """预训练数据集生成器（GPT模式，单向语言模型）
    """

    def sentence_process(self, text):
        """单个文本的处理函数
        流程：分词，然后转id。
        """
        tokens = self.tokenizer.tokenize(text=text)[1:-1]
        token_ids = self.tokenizer.tokens_to_ids(tokens)
        return [token_ids]

    def paragraph_process(self, texts):
        """给原方法补上starts、ends、paddings
        """
        starts = [self.token_cls_id]
        ends = [self.token_sep_id]
        paddings = [self.token_pad_id]
        return super(TrainingDatasetGPT,
                     self).paragraph_process(texts, starts, ends, paddings)

    def tfrecord_serialize(self, instances):
        """给原方法补上instance_keys
        """
        instance_keys = ['token_ids']
        return super(TrainingDatasetGPT,
                     self).tfrecord_serialize(instances, instance_keys)

    @staticmethod
    def load_tfrecord(record_names, sequence_length, batch_size):
        """给原方法补上parse_function
        """
        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            x = {
                'Input-Token': token_ids,
                'Input-Segment': segment_ids,
            }
            y = {
                'lm_loss': K.zeros([1]),
                'lm_acc': K.zeros([1]),
            }
            return x, y

        return TrainingDataset.load_tfrecord(
            record_names, batch_size, parse_function
        )


class TrainingDatasetUniLM(TrainingDatasetGPT):
    """预训练数据集生成器（UniLM模式，Seq2Seq模型）
    """
    @staticmethod
    def load_tfrecord(record_names, sequence_length, batch_size, token_sep_id):
        """给原方法补上parse_function
        """
        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            segment = K.random_uniform(
                shape=[1], minval=1, maxval=sequence_length - 1, dtype='int64'
            )[0]
            segment_ids = K.one_hot(segment + 1, sequence_length)
            segment_ids = K.cast(K.cumsum(segment_ids), 'int64')
            token_ids_1 = token_ids[:segment]
            token_ids_2 = K.zeros([1], dtype='int64') + token_sep_id
            token_ids_3 = token_ids[segment:-1]
            token_ids = K.concatenate([token_ids_1, token_ids_2, token_ids_3])
            x = {
                'Input-Token': token_ids,
                'Input-Segment': segment_ids,
            }
            y = {
                'unilm_loss': K.zeros([1]),
                'unilm_acc': K.zeros([1]),
            }
            return x, y

        return TrainingDataset.load_tfrecord(
            record_names, batch_size, parse_function
        )


def wrapper(**kwargs):

    from passbert.tokenizers import Tokenizer
    import json
    import glob
    import re
    from tqdm import tqdm

    model = kwargs['model']  # roberta
    sequence_length = kwargs['sequence_length']  # 32
    workers = kwargs['workers']  # 40
    max_queue_size = 4000
    # dict_path = kwargs['dict_path'] # '/disk/cw/nlp-guessing/bert/bert_base_dir/vocab.txt'
    # '/disk/xm/data_xm/rockyou_new.txt'
    training_filename = kwargs['training_filename']
    dup_factor = kwargs['dup_factor']  # 1
    num_samples = kwargs['num_samples']  # 1000000
    # '/disk/cw/nlp-guessing/corpora/bert_rockyou.tfrecord'
    record_filename = kwargs['record_filename']
    must_concat = kwargs['must_concat']
    # tokenizer = Tokenizer(dict_path, do_lower_case=True)
    tokenizer = PasswordTokenizer()

    def some_texts(training_file=training_filename, dupe_factors=dup_factor, n_samples=num_samples):
        luds = re.compile(
            r"^[a-zA-Z0-9\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]{4,30}$")
        with open(training_file, 'r') as fin:
            lines = []
            for line in fin:
                line = line.strip('\r\n')
                if luds.search(line):
                    lines.append(line)
        if n_samples < len(lines):
            lines = random.sample(lines, n_samples)
            print(
                f"We will only use {n_samples} passwords for pretraining.", file=sys.stderr)
            pass
        for _ in range(dupe_factors):
            random.shuffle(lines)
            count, texts = 0, []
            for line in lines:
                pwd = line
                texts.append(pwd)
                count += 1
                if count == 1000:
                    yield texts
                    count, texts = 0, []
            if texts:
                yield texts
            pass

        # filenames = glob.glob('/home/spaces_ac_cn/corpus/*/*/*')
        # np.random.shuffle(filenames)
        # count, texts = 0, []
        # for filename in filenames:
        #     with open(filename) as f:
        #         for l in f:
        #             l = json.loads(l)['text'].strip()
        #             texts.extend(re.findall(u'.*?[\n。]+', l))
        #             count += 1
        #             if count == 10:  # 10篇文章合在一起再处理
        #                 yield texts
        #                 count, texts = 0, []
        # if texts:
        #     yield texts
        pass

    assert model in ['roberta', 'gpt', 'unilm', 'cpg']  # 判断是否支持的模型类型

    if model == 'cpg':
        """
        The same as roberta
        """
        def word_segment(text):
            return list(text) 

        TD = TrainingDatasetCPG(
            tokenizer, word_segment, sequence_length=sequence_length
        )

        # for i in range(10):  # 数据重复10遍
        TD.process(
            corpus=tqdm(some_texts()),
            record_name=record_filename,
            workers=workers,
            max_queue_size=max_queue_size,
        )

    if model == 'roberta':
        """
        import jieba_fast as jieba
        jieba.initialize()
        def word_segment(text):
            return jieba.lcut(text)
        """
        def word_segment(text):
            return list(text) 

        TD = TrainingDatasetRoBERTa(
            tokenizer, word_segment, sequence_length=sequence_length, must_concat=must_concat
        )

        # for i in range(10):  # 数据重复10遍
        TD.process(
            corpus=tqdm(some_texts()),
            record_name=record_filename,
            workers=workers,
            max_queue_size=max_queue_size,
        )

    elif model == 'gpt':

        TD = TrainingDatasetGPT(tokenizer, sequence_length=sequence_length, must_concat=must_concat)

        TD.process(
            corpus=tqdm(some_texts()),
            record_name=record_filename,
            workers=workers,
            max_queue_size=max_queue_size,
        )

    elif model == 'unilm':

        TD = TrainingDatasetUniLM(tokenizer, sequence_length=sequence_length, must_concat=must_concat)

        TD.process(
            corpus=tqdm(some_texts()),
            record_name=record_filename,
            workers=workers,
            max_queue_size=max_queue_size,
        )


if __name__ == '__main__':
    cli = argparse.ArgumentParser('Data utils: generating record file')
    cli.add_argument('-i', '--input', dest='training',
                     type=str, required=True, help='training file')
    cli.add_argument('-m', '--model', dest='model', type=str, required=True, choices=['roberta', 'gpt', 'unilm', 'cpg'],
                     help='the format of the parsed training file')
    cli.add_argument('-l', '--seq-len', dest='seq_len', default=512, required=False, type=int, help='sequence length for each vector')
    cli.add_argument("-w", '--workers', type=int, dest='workers', default=40, required=False, help='number of workers')
    cli.add_argument("-d", '--dup-factor', type=int, dest='dup_factor', default=1, required=False, help='replication factor')
    cli.add_argument('-n', '--num-samples', type=int, default=1000000000, required=False,
                     help='number of samples for training file when num_samples < num_passwords in training file')
    cli.add_argument('-s', '--save-record', required=True, type=str, help='save record file')
    cli.add_argument('--must-concat', action='store_true', help='concat short passwords instead of simply padding')
    args = cli.parse_args()

    wrapper(model=args.model, sequence_length=args.seq_len, workers=args.workers, training_filename=args.training,
            dup_factor=args.dup_factor, num_samples=args.num_samples, record_filename=args.save_record, must_concat=args.must_concat)
    
    pass