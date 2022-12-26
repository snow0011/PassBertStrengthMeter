#! -*- coding: utf-8 -*-
# script for pretraining, Multi-GPUs version and multi-TPUs version

from inspect import ismemberdescriptor
import os, re, sys
import argparse

import numpy
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.ops.gen_linalg_ops import BatchSelfAdjointEig
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
os.environ['TF_KERAS'] = '1'  # Must be tf.keras
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

cli = argparse.ArgumentParser("Pretraining")
cli.add_argument("-m", "--model", dest='model', type=str, required=True, choices=['roberta', 'gpt', 'unilm'], help='model to be used')
cli.add_argument('-s', '--save-ckpt', dest='save', type=str, help='save ckpt-format model here')
cli.add_argument('--save-h5', dest='save_h5', type=str, default=None, help='save h5-format model here')
cli.add_argument('-i', '--input', dest='training', type=str, nargs='+', required=True, help='training file')
cli.add_argument('-c', '--config', dest='config', type=str, required=True, help='config path')
cli.add_argument('-l', '--log', dest='log', type=str, required=True, help='log path')
# cli.add_argument('--keras-ckpt-path', dest='keras_ckpt_path', type=str, default=None, help='keras checkpoint path')
cli.add_argument('--lr', dest='learning_rate', type=float, default=0.002, help='learning rate')
cli.add_argument('--warmup-steps', dest='warmup_steps', type=int, required=True, help='warmup stemps during training phase')
cli.add_argument('--total-steps', dest='total_steps', type=int, required=True,
                 help='total steps during training phase, num epochs = total steps / steps per epoch')
cli.add_argument('--steps-per-epoch', dest='steps_per_epoch', type=int, required=True, help='each epoch has specified steps')
cli.add_argument("--grad-acc", dest='grad_acc', type=int, help="gradiant accumulation", default=1)
cli.add_argument('-pre', dest='pre', type=str, help='load the first pre-trained model',required=True)

args = cli.parse_args()
model = args.model

logging = tf.compat.v1.logging
logging.set_verbosity(logging.DEBUG)
# corpus path and model path 
# if training with TPUs, you should save your corpus in Google Cloud Storage
# The path should start with gs:// 
# if training with GPUs, using local path
model_saved_path = args.save  #'/disk/cw/nlp-guessing/models/passbert_model.ckpt'
# bert_model_save_path = args.save  # '/disk/cw/nlp-guessing/models/passbert_bert_model_rockyou.ckpt'
# gpt_model_save_path = '/disk/cw/nlp-guessing/models/passbert_bert_model.ckpt'
corpus_paths = args.training
# [
#     '/disk/cw/nlp-guessing/corpora/bert_rockyou.tfrecord'
# ]
# other configurations
sequence_length = 32
batch_size = 512
# '/home/spaces_ac_cn/chinese_L-12_H-768_A-12/bert_config.json'
config_path = args.config  # '/disk/cw/nlp-guessing/models/bert_config.json'
# '/home/spaces_ac_cn/chinese_L-12_H-768_A-12/bert_model.ckpt'  # 如果从零训练，就设为None
checkpoint_path = args.pre
learning_rate = args.learning_rate  # 0.00176
weight_decay_rate = 0.01
num_warmup_steps = args.warmup_steps  # 31250
# num epochs = num_train_steps / steps_per_epoch
num_train_steps = args.total_steps  # 125000  # 125000
steps_per_epoch = args.steps_per_epoch  # 10000  # 10000
grad_accum_steps = args.grad_acc  # using gradiant accumulation when it is larger than 1
epochs = num_train_steps * grad_accum_steps // steps_per_epoch
exclude_from_weight_decay = ['Norm', 'bias']
exclude_from_layer_adaptation = ['Norm', 'bias']
tpu_address = None  # 'grpc://xxx.xxx.xxx.xxx:8470'  # None if using GPUs
which_optimizer = 'lamb'  # adam or lamb， with weight decay by default
lr_schedule = {
    num_warmup_steps * grad_accum_steps: 1.0,
    3 * num_warmup_steps * grad_accum_steps: 0.9,
    num_train_steps * grad_accum_steps: 0.8,
}
floatx = K.floatx()

# reading corpus, construct tensors

if model == 'roberta':

    dataset = TrainingDatasetCPG.load_tfrecord(
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
        token_sep_id=3,  # specifying the id of [SEP]
    )


def build_transformer_model_with_mlm(bert=None):
    """ bert with mlm
    """
    mdl = bert
    print(checkpoint_path)
    if bert is None:
        bert = build_transformer_model(
            # Load pretrain model
            config_path=config_path, 
            # checkpoint_path=checkpoint_path,
            with_mlm='linear',
            application='mlm',
            return_keras_model=False
        )
        mdl = bert.model
    proba = mdl.output

    # auxiliary input
    token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # id
    is_masked = Input(shape=(None,), dtype=floatx, name='is_masked')  # mask

    # if with_nsp=True, y_pred will be a list containing two items
    def mlm_loss(inputs):
        """calculating loss, wrapped as a layer
        """
        y_true, y_pred, mask = inputs
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred + 1e-5, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def mlm_acc(inputs):
        """calculating accuracy，wrapped as a layer
        """
        y_true, y_pred, mask = inputs
        y_true = K.cast(y_true, floatx)
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc
    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])
    # The property of name in token_ids  name are used to find the corresponding key in dataset,
    # and so is Input-Token.
    # If misspelled, you will get error messages like `key not found`.
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
        """calculating los, wrapped as a layer
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
        """calculating accuracy, wrapped as a layer
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
    """bert with unilm
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
        """calculating loss, wrapped as a layer
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
        """calculating accuracy, wrapped as a layer
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

    train_bert.compile(loss=loss_bert, optimizer=optimizer)
    train_gpt.compile(loss=loss_gpt, optimizer=optimizer)

    return train_bert, train_gpt, mixed_model
    pass


def build_transformer_model_for_pretraining():
    """constructing the training model, which is appliable for TPU/GPU
    the code should follow the standard application of Keras for each layer.
    Besides, TPU may not support varibale tensors.
    """
    if model == 'roberta':
        bert, train_model, loss = build_transformer_model_with_mlm()
    elif model == 'gpt':
        bert, train_model, loss = build_transformer_model_with_lm()
    elif model == 'unilm':
        bert, train_model, loss = build_transformer_model_with_unilm()

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

    train_model.compile(loss=loss, optimizer=optimizer)

    # load weights if provided. Note that you should load weights here to escape errors
    if checkpoint_path is not None:
        print(f"Load model {checkpoint_path}")
        bert.load_weights_from_checkpoint(checkpoint_path)

    return train_model, bert


if tpu_address is None:
    # single host multi GPUs, refer to https://tf.wiki
    strategy = tf.distribute.MirroredStrategy()
else:
    # TPU mode
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address
    )
    tf.config.experimental_connect_to_host(resolver.master())
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
    train_model, built_model = build_transformer_model_for_pretraining()
    # train_bert, train_gpt, mixed_model = build_transformer_model_mixed()

    train_model.summary()
    # train_gpt.summary()


class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model_path, built_model, built_model_path):
        super().__init__()
        self.model_path = model_path
        self.built_model = built_model
        self.built_model_path = built_model_path
        pass

    def on_epoch_end(self, epoch, logs=None):
        """
        save the latest model autometically
        """
        if self.model_path is not None:
            self.model.save_weights(self.model_path, overwrite=True)
        if self.built_model_path is not None:
            self.model.save_weights(self.built_model_path)


# saving model
checkpoint = ModelCheckpoint(model_saved_path, built_model, args.save_h5)
# checkpoint_bert = ModelCheckpoint(bert_model_save_path)
# checkpoint_gpt = ModelCheckpoint(gpt_model_save_path)
# saving log
csv_logger = keras.callbacks.CSVLogger(args.log)


train_model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[checkpoint, csv_logger])


"""
Note that the model (checkpoint) saved in callbacks could not be used in testing and evaluating.
the following two lines will convert the format of the model and make the model avaliable.
"""
# train_model.load_weights(model_saved_path)
# built_model.save_weights_as_checkpoint(args.tf_ckpt_path)
# train_bert.load_weights(bert_model_save_path)
# mixed_model.save_weights_as_checkpoint(filename='/disk/cw/nlp-guessing/models/passbert_bert_model_test.ckpt')