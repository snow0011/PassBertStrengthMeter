import argparse
from msilib.schema import Error
import os
import string
import sys
import numpy
import math
from keras.models import load_model
import tensorflow as tf
import logging
from functools import reduce
from typing import List
import string
# the folder is ``src/``
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
# 自定义层,这些库用于加载h5格式的模型
from passbert.backend import keras, K


# Read bert model (.h5, .ckpt)
# Convert the model to saved model format (Next will be converted to SavedModel format(.pb))
# Reveal .pb model don't need the layer information like .h5 file. 

def readH5(model_path):
    def my_loss(y_true, y_pred):
        # sparse categorical loss
        return K.sparse_categorical_crossentropy(y_true, y_pred)
    
    def my_loss_metric(y_true, y_pred):
        return keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    custom_objects = {
        'my_loss':my_loss,
        'my_loss_metric':my_loss_metric
    }
    logging.info('Loading model weights')
    model = load_model(model_path, custom_objects=custom_objects)
    logging.info('Done loading model')
    return model

def readCKPT(model_path, config_path):
    from passbert.models import build_transformer_model
    logging.info('Loading model weights')
    model = build_transformer_model(config_path=config_path, checkpoint_path=model_path, with_mlm=True)
    logging.info('Done loading model')
    return model

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("-m","--model",dest="model",type=str,required=True,help="Path of Bert model")
    cli.add_argument("-f","--format",dest="format",required=True,choices=["h5","ckpt"],type=str,help="Model format")
    cli.add_argument("-o","--output",dest="output",required=True,type=str,type="Output SavedModel directory")
    cli.add_argument("-c","--config",dest="config",required=False,default=None,type=str,type="Bert model configuration")
    args = cli.parse_args()
    model = None
    if "h5" == args.format:
        model = readH5(args.model)
    elif "ckpt" == args.format:
        if args.config == None:
            raise Exception("Configuration is needed in ckpt format")
        model = readCKPT(args.model, args.config)
    print(model.inputs)
    print(model.outputs)
    sess = K.get_session()
    # Save bert model as SavedModel format
    tf.saved_model.simple_save(
        sess,
        args.output,
        inputs={'Input-Token': model.inputs[0], 'Input-Segment': model.inputs[1]},
        outputs={'Output': model.outputs[0]})

if __name__ == '__main__':
    main()