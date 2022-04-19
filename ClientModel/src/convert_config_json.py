import sys
import argparse
import string
from collections import defaultdict
import csv
import json

spliter = " "
SYMBOLS = '~!@#$%^&*(),.<>/?\'"{}[]\\|-_=+;: `'
PASSWORD_END = '\n'

def read_config_file(afile):
    file_format = json.load
    answer = file_format(afile)
    return answer["config"]

class ModelDefaults():
    char_bag = (
        string.ascii_lowercase + string.ascii_uppercase + string.digits +
        SYMBOLS+PASSWORD_END)
    model_type = 'LSTM'
    sequence_model = 0
    hidden_size = 128
    layers = 1
    max_len = 40
    min_len = 4
    training_chunk = 128
    generations = 20
    chunk_print_interval = 1000
    lower_probability_threshold = 1.1 * (10**-11)
    relevel_not_matching_passwords = True
    training_accuracy_threshold = -1.0
    train_test_ratio = 10
    rare_character_optimization = False
    rare_character_optimization_guessing = False
    uppercase_character_optimization = False
    rare_character_lowest_threshold = 20
    guess_serialization_method = 'human'
    simulated_frequency_optimization = False
    intermediate_fname = " : "
    save_always = True
    save_model_versioned = False
    randomize_training_order = True
    model_optimizer = 'adam'
    guesser_intermediate_directory = 'guesser_files'
    cleanup_guesser_files = True
    early_stopping = False
    early_stopping_patience = 10000
    compute_stats = False
    password_test_fname = ""
    chunk_size_guesser = 1000
    random_walk_seed_num = 1000
    max_gpu_prediction_size = 25000
    cpu_limit = 8
    random_walk_confidence_bound_z_value = 1.96
    random_walk_confidence_percent = 5
    random_walk_upper_bound = 10
    no_end_word_cache = False
    enforced_policy = 'basic'
    pwd_list_weights = {}
    dropouts = False
    dropout_ratio = .25
    tensorboard = False
    tensorboard_dir = "."
    context_length = 10
    train_backwards = False
    dense_layers = 0
    dense_hidden_size = 128
    secondary_training = False
    secondary_train_sets = {}
    training_main_memory_chunksize = 1000000
    probability_steps = False
    freeze_feature_layers_during_secondary_training = True
    secondary_training_save_freqs = False
    guessing_secondary_training = False
    guesser_class = None
    freq_format = 'hex'
    padding_character = False
    convolutional_kernel_size = 3
    embedding_layer = False
    embedding_size = 8
    previous_probability_mapping_file = None
    probability_calculator_cache_size = 0
    tokenize_guessing = False
    tokenize_words = False
    token_dict_fname = None
    token_extension = False
    guess_mode = False

    def __init__(self, adict=None, **kwargs):

        self.adict = adict if adict is not None else dict()
        for k in kwargs:
            self.adict[k] = kwargs[k]

    def __getattribute__(self, name):
        if name != 'adict' and name in self.adict:
            return self.adict[name]

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name != 'adict' and not name.startswith("_"):
            self.adict[name] = value
        else:
            super().__setattr__(name, value)

    @staticmethod
    def fromFile(afile):
        if afile is None:
            return ModelDefaults()
        return ModelDefaults(read_config_file(afile))

def read_voc(voc, char_bag=set(string.ascii_lowercase + string.ascii_uppercase + string.digits + SYMBOLS)):
    if voc is None:
        voc  = []
    mapper = defaultdict(int)
    total = 0
    for ch in char_bag:
        mapper[ch] = 1
        total += 1
    for line in voc:
        line = line.strip("\r\n")
        ss = line.split(spliter)
        subword = ss[0]
        if subword[-1] not in char_bag:
            subword = subword[:-1]
        if len(subword)>=5:
            continue
        count = 1
        if len(ss) > 1:
            count = int(ss[1])
        mapper[subword] += count
        total += count
    vals = sorted(mapper.items(), key=lambda x:x[0])
    return zip(*vals),total

def read_csv(tsv):
    if tsv is None:
        return []
    output = []
    for row in csv.reader(tsv, quotechar=None, delimiter='\t'):
        output.append((float(row[1]), float(row[2])))
    return sorted(output, key=lambda x: x[0])[::100]

def read_config(config):
    return ModelDefaults.fromFile(config)

def main():
    cli = argparse.ArgumentParser("Convert configure to json")
    cli.add_argument("--voc",type=argparse.FileType("r"),default=None)
    cli.add_argument("--csv",type=argparse.FileType("r"),default=None)
    cli.add_argument("--config",type=argparse.FileType("r"))
    cli.add_argument("--ofile",type=argparse.FileType("w"))
    args = cli.parse_args()
    csv_list = read_csv(args.csv)
    config = read_config(args.config)
    vals = read_voc(args.voc,char_bag=config.char_bag)
    res = {}
    subwords, freq = vals[0]
    res["subwords"] = subwords
    res["freqs"] = list(map(lambda x:x/vals[1], freq))
    res["guess_table"]=csv_list
    res["context_len"]=config.context_length
    res["version"]="Sub-word neural network guessing model: version 0"
    res["embedding"] = True
    json.dump(res, args.ofile)
    print("Convert successfully")


if __name__ == '__main__':
    main()
