from __future__ import print_function
import glob
from keras.layers.core import Lambda
import numpy as np
import sys
import os
import argparse

from numpy.lib.npyio import save
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tasks.losses import binary_focal_loss
from keras.layers import Dense, Flatten
from keras.models import Model
from passbert.snippets import DataGenerator, AutoRegressiveDecoder
from passbert.snippets import sequence_padding
from passbert.optimizers import Adam
from passbert.tokenizers import load_vocab, PasswordTokenizer
from passbert.models import build_transformer_model
from passbert.backend import keras, K
from tqdm import tqdm

def read_in_training_data(root_path: str, max_len: int):
    ans = []
    def handler(line):
        ss = line.split("\t")
        return (ss[0], int(ss[1]))
    with open(root_path, "r") as f:
        for line in f:
            line = line.strip("\r\n")
            ans.append(handler(line))
    return ans

class data_generator(DataGenerator):

    def __init__(self, data, tokenizer, batch_size=32, buffer_size=None):
        super().__init__(data, batch_size=batch_size, buffer_size=buffer_size)
        self.tokenizer = tokenizer
    
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        # The text is in [password, transformation list] format. The transformation should be encoded type, like [2,5,7,8,0]
        # Is end marked the end of the data sequence.
        for is_end, text in self.sample(random):
            pwd, label = text
            wrappered_text = [self.tokenizer._token_start,
                              *pwd, self.tokenizer._token_end]
            segment_ids = [0] * len(wrappered_text)
            token_ids = [self.tokenizer.token_to_id(x) for x in wrappered_text]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            # Transfer the label list to binary format
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                # print(batch_labels)
                # print(batch_token_ids, batch_segment_ids, batch_labels)
                yield [batch_token_ids, batch_segment_ids], [batch_labels] 
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

        pass

    pass

def build_model_for_fine_tuning(config_path: str, checkpoint_path: str, num_classes: int):
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
    output = model.output
    output = Dense(512, activation='tanh')(output)
    output = Dense(num_classes, activation='sigmoid')(output)
    output = Lambda(lambda x: x[:, 0])(output)
    model = Model(model.input, output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5))
    model.summary()
    return model


def predict_pwd(pwd, tokenizer, model):
    text = [tokenizer._token_start, *pwd, tokenizer._token_end]
    token_ids = tokenizer.tokens_to_ids(text)
    segment_ids = np.zeros_like(token_ids)
    results = model.predict([[token_ids], [segment_ids]])
    return results


class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_model: str):
        super().__init__()
        self.lowest = 1e10
        self.save_model = save_model
        pass

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # update the saved model here
            if self.save_model.endswith('ckpt'):
                self.model.save_weights(f"{self.save_model}")
            elif self.save_model.endswith('h5'):
                self.model.save(f"{self.save_model}")
            else:
                self.model.save_weights(f"{self.save_model}.ckpt")
                self.model.save(f"{self.save_model}.h5")
            pass
        pass
    pass

def wrapper():
    cli = argparse.ArgumentParser('')
    cli.add_argument('--config', dest='config', required=True, type=str, help='config path for bert')
    cli.add_argument('--model', dest='model', required=True, type=str, help='the saved model to further fine-tuning')
    cli.add_argument('--save', dest='save', required=True, type=str, help='save fine-tuned model')
    cli.add_argument('--training-path', dest='training_path', required=True, type=str, help='')
    cli.add_argument('--batch-size', dest='batch_size', default=512, type=int, help='batch size')
    cli.add_argument('--steps-per-epoch', dest='steps_per_epoch', default=100, type=int, 
                     help='steps per epoch, batch_size * stpes per epoch = number of passwords trained in an epoch')
    cli.add_argument('--epochs', dest='epochs', default=4, type=int, help='number of epochs')
    cli.add_argument('--max-len', dest='max_len', default=32, type=int, help='max len of the password')
    
    # number of classes in this task
    NUM_CLASS=10
    
    args = cli.parse_args()
    evaluator = Evaluator(save_model=args.save)
    tokenizer = PasswordTokenizer()
    
    training_data = read_in_training_data(root_path=args.training_path, max_len=args.max_len)
    
    training_data_generator = data_generator(training_data, tokenizer=tokenizer, batch_size=args.batch_size)
    
    model = build_model_for_fine_tuning(config_path=args.config, checkpoint_path=args.model, num_classes=NUM_CLASS)
    
    # Fine-tuning process
    model.fit(training_data_generator.forfit(), steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, callbacks=[evaluator])
    save_model=args.save
    
    print(f"Model save at {save_model}")
    if save_model.endswith('ckpt'):
        model.save_weights(f"{save_model}")
    elif save_model.endswith('h5'):
        model.save(f"{save_model}")
    else:
        model.save_weights(f"{save_model}.ckpt")
        model.save(f"{save_model}.h5")
    pass


if __name__ == '__main__':
    wrapper()
    pass