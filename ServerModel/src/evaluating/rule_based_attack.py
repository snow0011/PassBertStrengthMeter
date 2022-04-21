import json
import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from passbert.tokenizers import load_vocab, PasswordTokenizer
from passbert.models import build_transformer_model
from pyruleengine.PyHashcat import read_rules, read_target, read_words
from pyruleengine.PyRuleEngine import RuleEngine
from tasks.bert4multilabels import custom_objects, my_sparse_loss_metric
from keras.models import load_model


def predict_pwd(pwd, tokenizer, model, seq_len=32):
    padding = [tokenizer._token_pad] * (seq_len - len(pwd) - 2)
    text = [tokenizer._token_start, *pwd, tokenizer._token_end, *padding]
    token_ids = tokenizer.tokens_to_ids(text)
    segment_ids = np.zeros_like(token_ids)
    results = model.predict([[token_ids], [segment_ids]])
    return results


def adams(words_path, rules_path, target_path, model, tokenizer, threshold, log_rules_cnt_for_each_word: str):
    rules = read_rules(rule_path=rules_path)
    word_list = read_words(words_path=words_path)
    targets = read_target(target_path=target_path)
    rule_engine = RuleEngine(rules=rules)
    guess_number = 0
    acc_guessed = 0
    f_log = open(log_rules_cnt_for_each_word, 'w')
    metadata = {
        "rules": rules,
        "threshold": threshold,
    }
    f_log.write(f"{json.dumps(metadata)}\n")
    f_log.flush()
    used_rules_count = 0
    word_count = 0
    for w_i, word in word_list:
        if w_i % 100 == 0:
            print(f"{w_i:10} words parsed", end='\r', file=sys.stderr, flush=True)
        word_count = w_i + 1
        probabilities = predict_pwd(word, tokenizer, model)[0]
        indices = [r_i for r_i in range(len(probabilities)) if probabilities[r_i] >= threshold]
        if len(indices) == 0:
            continue
        _log = [word, indices]
        used_rules_count += len(indices)
        f_log.write(f"{json.dumps(_log)}\n")
        rule_engine.change_indices(indices)
        candidates = rule_engine.apply(word)
        for guess, rule in candidates:
            guess_number += 1
            if guess in targets:
                duplicates = targets[guess]
                acc_guessed += duplicates
                del targets[guess]
                yield word, guess, rule, guess_number, duplicates, acc_guessed
    rules_per_word = used_rules_count / word_count
    f_log.write(f"{used_rules_count} / {word_count} = {rules_per_word:5.2f} rules per word, "
                f"{rules_per_word} / {len(rules)} = {rules_per_word / len(rules):7.3%} averagely used rules\n")
    f_log.flush()
    f_log.close()


def wrapper():
    cli = argparse.ArgumentParser("")
    cli.add_argument('-w', '--words', required=True, type=str, help='words path')
    cli.add_argument('-r', '--rules', required=True, type=str, help='rules path')
    cli.add_argument('-t', '--targets', required=True, type=str, help='passwords to guess')
    cli.add_argument('-m', '--model', required=True, type=str, help='model path')
    cli.add_argument('-b', '--budget', required=True, type=float, help='budget, which has the same meaning as `budget` in ADaMs')
    cli.add_argument('-s', '--save', required=False, type=str, default='stdout', help='save the output into file')
    cli.add_argument('--log', required=True, type=str, help='rules applied for each word will be saved here')
    args = cli.parse_args()
    model = load_model(args.model, custom_objects={**custom_objects, 'my_sparse_loss_metric': my_sparse_loss_metric})
    tokenizer = PasswordTokenizer()
    hits = adams(words_path=args.words, rules_path=args.rules, target_path=args.targets,
                 model=model, tokenizer=tokenizer, threshold=1 - args.budget, log_rules_cnt_for_each_word=args.log)
    save_file = args.save
    if save_file == 'stdout':
        f_out = sys.stdout
    else:
        f_out = open(save_file, 'w')
    for word, guess, rule, rank, dup, acc_guessed in hits:
        f_out.write(f"{word}\t{guess}\t{' '.join(rule)}\t{rank}\t{dup}\t{acc_guessed}\n")
    f_out.close()
    pass


if __name__ == '__main__':
    wrapper()
    pass