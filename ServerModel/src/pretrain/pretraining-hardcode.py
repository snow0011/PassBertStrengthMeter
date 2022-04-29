#! -*- coding: utf-8 -*-
# 预训练脚本，多GPU版/TPU版本

from inspect import ismemberdescriptor
import os, re, sys

import numpy
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.ops.gen_linalg_ops import BatchSelfAdjointEig
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from data_utils import *
from passbert.models import build_transformer_model, build_mixed_transformer_model
from passbert.backend import keras, K
from passbert.optimizers import Adam
from passbert.optimizers import extend_with_weight_decay
from passbert.optimizers import extend_with_layer_adaptation
from passbert.optimizers import extend_with_piecewise_linear_lr
from passbert.optimizers import extend_with_gradient_accumulation
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import Callback, CSVLogger

model = 'gpt'

logging = tf.compat.v1.logging
logging.set_verbosity(logging.DEBUG)
# 语料路径和模型保存路径
# 如果是TPU训练，那么语料必须存放在Google Cloud Storage上面，
# 路径必须以gs://开头；如果是GPU训练，改为普通路径即可。
model_saved_path = '/disk/cw/nlp-guessing/models/passbert_model.ckpt'
bert_model_save_path = '/disk/cw/nlp-guessing/models/passbert_bert_model_rockyou.ckpt'
gpt_model_save_path = '/disk/cw/nlp-guessing/models/passbert_bert_model.ckpt'
corpus_paths = [
    # '/disk/cw/nlp-guessing/corpora/tf_examples.tfrecord'
    '/disk/cw/nlp-guessing/corpora/bert_sample.tfrecord'
]
corpus_paths4bert = [
    '/disk/cw/nlp-guessing/corpora/bert_rockyou.tfrecord'
]
corpus_paths4gpt = [
    '/disk/cw/nlp-guessing/corpora/gpt_sample.tfrecord'
]
# 其他配置
sequence_length = 32
batch_size = 512
# '/home/spaces_ac_cn/chinese_L-12_H-768_A-12/bert_config.json'
config_path = '/disk/cw/nlp-guessing/models/bert_config.json'
# '/home/spaces_ac_cn/chinese_L-12_H-768_A-12/bert_model.ckpt'  # 如果从零训练，就设为None
checkpoint_path = None
learning_rate = 0.00176
weight_decay_rate = 0.01
num_warmup_steps = 31250
num_train_steps = 125000  # 125000
steps_per_epoch = 10000  # 10000
grad_accum_steps = 1  # 大于1即表明使用梯度累积
epochs = num_train_steps * grad_accum_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm', 'bias']
exclude_from_layer_adaptation = ['Norm', 'bias']
tpu_address = None  # 'grpc://xxx.xxx.xxx.xxx:8470'  # 如果用多GPU跑，直接设为None
which_optimizer = 'lamb'  # adam 或 lamb，均自带weight decay
lr_schedule = {
    num_warmup_steps * grad_accum_steps: 1.0,
    3 * num_warmup_steps * grad_accum_steps: 0.9,
    num_train_steps * grad_accum_steps: 0.8,
}
floatx = K.floatx()

# 读取数据集，构建数据张量

if model == 'roberta':

    dataset = TrainingDatasetRoBERTa.load_tfrecord(
        record_names=corpus_paths,
        sequence_length=sequence_length,
        batch_size=batch_size // grad_accum_steps,
    )

elif model == 'gpt':

    dataset = TrainingDatasetGPT.load_tfrecord(
        record_names=corpus_paths,
        sequence_length=sequence_length,
        batch_size=batch_size // grad_accum_steps,
    )

elif model == 'unilm':

    dataset = TrainingDatasetUniLM.load_tfrecord(
        record_names=corpus_paths,
        sequence_length=sequence_length,
        batch_size=batch_size // grad_accum_steps,
        token_sep_id=3,  # 这里需要自己指定[SEP]的id
    )
dataset4bert = TrainingDatasetRoBERTa.load_tfrecord(
    record_names=corpus_paths4bert,
    sequence_length=sequence_length,
    batch_size=batch_size // grad_accum_steps,
)

dataset4gpt = TrainingDatasetGPT.load_tfrecord(
    record_names=corpus_paths,
    sequence_length=sequence_length,
    batch_size=batch_size // grad_accum_steps,
)


def build_transformer_model_with_mlm(bert=None):
    """带mlm的bert模型
    """
    mdl = bert
    if bert is None:
        bert = build_transformer_model(
            config_path, with_mlm='linear',
            application='mlm',
            return_keras_model=False
        )
        mdl = bert.model
    proba = mdl.output

    # 辅助输入
    token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
    is_masked = Input(shape=(None,), dtype=floatx, name='is_masked')  # mask标记

    # 设置 with_nsp=True 后，y_pred 将会变成含有两个值的列表
    # 这个列表可以想办法存储 LM 和 MLM 训练目标的结果，然后分别计算 loss 后相加
    def mlm_loss(inputs):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred + 1e-5, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def mlm_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        y_true = K.cast(y_true, floatx)
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc
    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])
    # 注意，token_ids 中的 name 属性用于寻找 dataset 中对应的 key，
    # models 中 embedding 过程的 Input-Token 等也是如此。
    # 如果拼写出错的话，会找不到对应的 key，从而报错。
    tmp = mdl.inputs + [token_ids, is_masked]
    train_model = Model(
        tmp, [mlm_loss, mlm_acc]
    )

    loss = {
        'mlm_loss': lambda y_true, y_pred: y_pred,
        'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
    }

    return bert, train_model, loss


def build_transformer_model_with_lm(bert=None):
    """带lm的bert模型
    """
    mdl = bert
    if bert is None:
        bert = build_transformer_model(
            config_path,
            with_mlm='linear',
            application='lm',
            return_keras_model=False
        )
        mdl = bert.model
    token_ids = mdl.inputs[0]
    proba = mdl.output

    def lm_loss(inputs, mask=None):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_true, y_pred = y_true[:, 1:], y_pred[:, :-1]

        if mask is None:
            mask = 1.0
        else:
            mask = K.cast(mask[1][:, 1:], floatx)

        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred + 1e-5, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def lm_acc(inputs, mask=None):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_true, y_pred = K.cast(y_true[:, 1:], floatx), y_pred[:, :-1]

        if mask is None:
            mask = 1.0
        else:
            mask = K.cast(mask[1][:, 1:], floatx)

        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

    lm_loss = Lambda(lm_loss, name='lm_loss')([token_ids, proba])
    lm_acc = Lambda(lm_acc, name='lm_acc')([token_ids, proba])

    train_model = Model(mdl.inputs, [lm_loss, lm_acc])

    loss = {
        'lm_loss': lambda y_true, y_pred: y_pred,
        'lm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
    }

    return bert, train_model, loss


def build_transformer_model_with_unilm():
    """带unilm的bert模型
    """
    bert = build_transformer_model(
        config_path,
        with_mlm='linear',
        application='unilm',
        return_keras_model=False
    )
    token_ids = bert.model.inputs[0]
    segment_ids = bert.model.inputs[1]
    proba = bert.model.output

    def unilm_loss(inputs, mask=None):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred, segment_ids = inputs
        y_true, y_pred = y_true[:, 1:], y_pred[:, :-1]

        if mask is None:
            mask = 1.0
        else:
            mask = K.cast(mask[1][:, 1:], floatx)

        segment_ids = K.cast(segment_ids, floatx)
        mask = mask * segment_ids[:, 1:]

        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred  + 1e-5, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def unilm_acc(inputs, mask=None):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, segment_ids = inputs
        y_true, y_pred = K.cast(y_true[:, 1:], floatx), y_pred[:, :-1]

        if mask is None:
            mask = 1.0
        else:
            mask = K.cast(mask[1][:, 1:], floatx)

        segment_ids = K.cast(segment_ids, floatx)
        mask = mask * segment_ids[:, 1:]

        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

    token_proba_segment = [token_ids, proba, segment_ids]
    unilm_loss = Lambda(unilm_loss, name='unilm_loss')(token_proba_segment)
    unilm_acc = Lambda(unilm_acc, name='unilm_acc')(token_proba_segment)

    train_model = Model(bert.model.inputs, [unilm_loss, unilm_acc])

    loss = {
        'unilm_loss': lambda y_true, y_pred: y_pred,
        'unilm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
    }

    return bert, train_model, loss


def build_transformer_model_mixed():
    mixed_model = build_mixed_transformer_model(
        config_path, 
        with_mlm='linear',
        application='mlm',
        return_keras_model=False,
    )
    bert, gpt = mixed_model.model
    my_bert, train_bert, loss_bert = build_transformer_model_with_mlm(bert)
    my_gpt, train_gpt, loss_gpt = build_transformer_model_with_lm(gpt)
    # 优化器
    optimizer = extend_with_weight_decay(Adam)
    if which_optimizer == 'lamb':
        optimizer = extend_with_layer_adaptation(optimizer)
    optimizer = extend_with_piecewise_linear_lr(optimizer)
    optimizer_params = {
        'learning_rate': learning_rate,
        'lr_schedule': lr_schedule,
        'weight_decay_rate': weight_decay_rate,
        'exclude_from_weight_decay': exclude_from_weight_decay,
        'exclude_from_layer_adaptation': exclude_from_layer_adaptation,
        'bias_correction': False,
        'clipnorm': 1,
    }
    if grad_accum_steps > 1:
        optimizer = extend_with_gradient_accumulation(optimizer)
        optimizer_params['grad_accum_steps'] = grad_accum_steps
    optimizer = optimizer(**optimizer_params)

    # 模型定型
    train_bert.compile(loss=loss_bert, optimizer=optimizer)
    train_gpt.compile(loss=loss_gpt, optimizer=optimizer)

    return train_bert, train_gpt, mixed_model
    pass


def build_transformer_model_for_pretraining():
    """构建训练模型，通用于TPU/GPU
    注意全程要用keras标准的层写法，一些比较灵活的“移花接木”式的
    写法可能会在TPU上训练失败。此外，要注意的是TPU并非支持所有
    tensorflow算子，尤其不支持动态（变长）算子，因此编写相应运算
    时要格外留意。
    """
    if model == 'roberta':
        bert, train_model, loss = build_transformer_model_with_mlm()
    elif model == 'gpt':
        bert, train_model, loss = build_transformer_model_with_lm()
    elif model == 'unilm':
        bert, train_model, loss = build_transformer_model_with_unilm()

    # 优化器
    optimizer = extend_with_weight_decay(Adam)
    if which_optimizer == 'lamb':
        optimizer = extend_with_layer_adaptation(optimizer)
    optimizer = extend_with_piecewise_linear_lr(optimizer)
    optimizer_params = {
        'learning_rate': learning_rate,
        'lr_schedule': lr_schedule,
        'weight_decay_rate': weight_decay_rate,
        'exclude_from_weight_decay': exclude_from_weight_decay,
        'exclude_from_layer_adaptation': exclude_from_layer_adaptation,
        'bias_correction': False,
    }
    if grad_accum_steps > 1:
        optimizer = extend_with_gradient_accumulation(optimizer)
        optimizer_params['grad_accum_steps'] = grad_accum_steps
    optimizer = optimizer(**optimizer_params)

    # 模型定型
    train_model.compile(loss=loss, optimizer=optimizer)

    # 如果传入权重，则加载。注：须在此处加载，才保证不报错。
    if checkpoint_path is not None:
        bert.load_weights_from_checkpoint(checkpoint_path)

    return train_model


if tpu_address is None:
    # 单机多卡模式（多机多卡也类似，但需要硬软件配合，请参考https://tf.wiki）
    strategy = tf.distribute.MirroredStrategy()
else:
    # TPU模式
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address
    )
    tf.config.experimental_connect_to_host(resolver.master())
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    train_model = build_transformer_model_for_pretraining()
    train_bert, train_gpt, mixed_model = build_transformer_model_mixed()
    train_bert.summary()
    train_gpt.summary()


class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model_path):
        self.model_path = model_path
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        自动保存最新模型
        """
        self.model.save_weights(self.model_path, overwrite=True)


# 保存模型
checkpoint = ModelCheckpoint(model_saved_path)
checkpoint_bert = ModelCheckpoint(bert_model_save_path)
checkpoint_gpt = ModelCheckpoint(gpt_model_save_path)
# 记录日志
csv_logger = keras.callbacks.CSVLogger('training.log')

# 模型训练
# train_model.fit(
#     dataset,
#     steps_per_epoch=steps_per_epoch,
#     epochs=epochs,
#     callbacks=[checkpoint, csv_logger],
# )
mini_batches4bert=5
mini_batches4gpt=5
mini_sum = mini_batches4bert + mini_batches4gpt
# sess = tf.Session()
# for _ in range(4):
#     i, j = 0, 0
#     step_i, step_j = 128, 128
#     for _ in range(steps_per_epoch):
#         train_bert.fit(dataset4bert.skip(i).take(step_i).repeat(), steps_per_epoch=128, epochs=1, callbacks=[checkpoint_bert])
#         train_gpt.fit(dataset4gpt.skip(j).take(step_j).repeat(), steps_per_epoch=128, epochs=1, callbacks=[checkpoint_gpt])
#         i, j = i + step_i, j + step_j
#     dataset4bert.shuffle(batch_size * 1000)
#     dataset4gpt.shuffle(batch_size * 1000)
#     pass
# train_bert.train_on_batch(
#     x = {
#         'Input-Token': numpy.array([0, 1, 2, 3, 4]), 'Input-Segment': numpy.array([0, 0, 0, 0, 0]), 
#         'token_ids': numpy.array([0, 1, 2, 3, 4]), 'is_masked': numpy.array([0, 0, 0, 0, 0])}, 
#     y = {'mlm_loss': numpy.array([.0]), 'mlm_acc': numpy.array([.0])})
train_bert.fit(dataset4bert, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[checkpoint_bert, csv_logger])
"""
Note that the model (checkpoint) saved in callbacks could not be used in testing and evaluating.
the following two lines will convert the format of the model and make the model avaliable.
"""
train_bert.load_weights(bert_model_save_path)
mixed_model.save_weights_as_checkpoint(filename='/disk/cw/nlp-guessing/models/passbert_bert_model_test.ckpt')