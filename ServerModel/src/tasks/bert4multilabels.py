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

def read_in_training_data(root_path: str, max_len: int, num_rules: int):
    """
    @param root_path: the directory of the saved training data.
    @param max_len: the maximum length of the passwords.
    @param num_rules: number of rules, i.e., number of classes in the task.
    @return: password and corresponding label
    """
    def get_all(_root_path, _s): return sorted(
        glob.glob(os.path.join(_root_path, _s)), key=lambda x: int(x.split('_')[-1]))
    # char_not_found = max(char_map.values()) + 1
    data = []
    all_pwd_list_path = get_all(root_path, 'X.txt_*')
    number_ones = 0
    total_binaries = 0
    for pwd_list_path, index_path, hits_path in tqdm(zip(all_pwd_list_path, get_all(root_path, 'index_hits*'), get_all(root_path, 'hits*')), desc='Files: ', total=len(all_pwd_list_path)):
        with open(pwd_list_path, 'r') as f_pwd_list:
            pwd_list = [pwd.strip() for pwd in f_pwd_list]
        # parsed_pwd_list = np.zeros((len(pwd_list), max_len), np.uint8)
        # for idx, pwd in enumerate(pwd_list):
            # parsed_pwd = [char_map.get(c, char_not_found) for c in pwd]
        with open(index_path, 'rb') as f_index:
            total_index_bytes = 4 * len(pwd_list)
            b_indexes = f_index.read(total_index_bytes)
            indexes = np.frombuffer(b_indexes, dtype=np.uint32)
        with open(hits_path, 'rb') as f_hits:
            for pwd, index in zip(pwd_list, indexes):
                total_rule_bytes = index * 4
                b_rules = f_hits.read(total_rule_bytes)
                rule_hits = np.frombuffer(b_rules, dtype=np.uint32)
                y = np.zeros(num_rules, dtype=np.uint32)
                y[rule_hits] = 1
                number_ones += len(rule_hits)
                total_binaries += num_rules
                if len(pwd) > max_len:
                    continue
                data.append((pwd, y))
                pass
            pass
        break
    print(f"Ratio of 1: {number_ones / total_binaries}")
    return data


class data_generator(DataGenerator):

    def __init__(self, data, tokenizer, batch_size=32, buffer_size=None):
        super().__init__(data, batch_size=batch_size, buffer_size=buffer_size)
        self.tokenizer = tokenizer

    # 生成[token ids, segment ids], [transfermation list]
    def __iter__(self, random=False):
        """
        数据生成器
        """
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


def my_loss(y_true, y_pred):
    # y_true = K.print_tensor(y_true, message='y_true = ')
    # y_pred = K.print_tensor(y_pred, message='y_pred = ')

    # alpha = 0.005
    # alpha = 1 - percentage_of_one/percentage_of_zeros
    focal_loss_fn = binary_focal_loss(alpha=0.95)

    return focal_loss_fn(y_true, y_pred)

def my_binary_accuracy(y_true, y_pred, threshold=0.5):
    if threshold != 0.5:
        threshold = K.cast(threshold, y_pred.dtype)
        y_pred = K.cast(y_pred > threshold, y_pred.dtype)
    mask_y_pred = K.batch_dot(y_true, y_pred)
    return K.sum(mask_y_pred)/K.sum(y_true)

def my_sparse_loss_metric(y_true, y_pred):
    # The test threshold of the training model
    return my_binary_accuracy(y_true, y_pred, threshold=0.2)

def K_sparse_loss_metric(y_true, y_pred):
    # The test threshold of the training model
    return keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.2)

custom_objects={
    "my_loss":my_loss,
    "my_binary_accuracy":my_binary_accuracy,
    "K_sparse_loss_metric":K_sparse_loss_metric
}


def build_model_for_fine_tuning(config_path: str, checkpoint_path: str, num_classes: int):
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)
    output = model.output
    output = Dense(512, activation='tanh')(output)
    output = Dense(num_classes, activation='sigmoid')(output)
    output = Lambda(lambda x: x[:, 0])(output)

    model = Model(model.input, output)
    model.compile(loss=my_loss, optimizer=Adam(1e-5), 
                  metrics=[my_sparse_loss_metric,K_sparse_loss_metric])
    model.summary()
    return model


def predict_pwd(pwd, tokenizer, model):
    text = [tokenizer._token_start, *pwd, tokenizer._token_end]
    token_ids = tokenizer.tokens_to_ids(text)
    segment_ids = np.zeros_like(token_ids)
    results = model.predict([[token_ids], [segment_ids]])
    # print(results)
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


def read_rules(rule_path: str):
    with open(rule_path, 'r') as f_rule:
        raw_rules = [r.strip() for r in f_rule]
        rules = [r for r in raw_rules if r and r[0] != '#']
    return rules


def just4show(tokenizer, model):
    demo_test_list = ["password", "!!!!", "!!!!!", "!!!!!!", "!!!!!!!"]
    threshold = 0.5
    for pwd in demo_test_list:
        classification = predict_pwd(pwd, tokenizer, model)
        print(f"input: {pwd}, output: {classification}")
        print("max label: ", np.max(classification[0]))
        for i, v in enumerate(classification[0]):
            if v >= threshold:
                print(f"{i}\t{v}")
            pass
        pass
    pass


def wrapper_for_interactive(tokenizer, model, rules):
    threshold = 0.6
    while True:
        pwd_threshold = input(f"Type in password [followd by a SPACE and a THRESHOLD or use {threshold}]:\n")
        if ' ' in pwd_threshold:
            lst = pwd_threshold.split(' ')
            pwd, threshold = lst
            threshold = float(threshold)
        else:
            pwd = pwd_threshold
        classification = predict_pwd(pwd, tokenizer, model)
        print(classification)
        for i, v in enumerate(classification[0]):
            if v >= threshold:
                print(f"{i}\t{v}")
            pass

    pass

def wrapper():
    cli = argparse.ArgumentParser('')
    cli.add_argument('--config', dest='config', required=True, type=str, help='config path for bert')
    cli.add_argument('--model', dest='model', required=True, type=str, help='the saved model to further fine-tuning')
    cli.add_argument('--save', dest='save', required=True, type=str, help='save fine-tuned model')
    cli.add_argument('--training-path', dest='training_path', required=True, type=str, help='')
    cli.add_argument('--rules', dest='rules', required=True, type=str, help='the rule set used to generate training data')
    cli.add_argument('--batch-size', dest='batch_size', default=512, type=int, help='batch size')
    cli.add_argument('--steps-per-epoch', dest='steps_per_epoch', default=100, type=int, 
                     help='steps per epoch, batch_size * stpes per epoch = number of passwords trained in an epoch')
    cli.add_argument('--epochs', dest='epochs', default=4, type=int, help='number of epochs')
    cli.add_argument('--max-len', dest='max_len', default=32, type=int, help='max len of the password')
    cli.add_argument('--it', action='store_true', help='interactive mode')
    args = cli.parse_args()
    rules = read_rules(rule_path=args.rules)
    num_classes = len(rules)
    evaluator = Evaluator(save_model=args.save)
    tokenizer = PasswordTokenizer()
    training_data = read_in_training_data(root_path=args.training_path, max_len=args.max_len, num_rules=num_classes)
    training_data_generator = data_generator(training_data, tokenizer=tokenizer, batch_size=args.batch_size)
    model = build_model_for_fine_tuning(config_path=args.config, checkpoint_path=args.model, num_classes=num_classes)
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
    if args.it:
        wrapper_for_interactive(tokenizer, model, rules)
        pass
    else:
        just4show(tokenizer=tokenizer, model=model)
    pass


if __name__ == '__main__':
    wrapper()
    pass