"""
PPSM for passbert, referred to as PBSM
"""
import argparse
import os
import string
import sys
import numpy
import math
from typing import List
# the folder is ``src/``
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


def needed_items():
    letters = string.ascii_letters
    digits = string.digits
    symbols = string.punctuation
    needed = [*letters, *digits, *symbols]
    return needed


def to_array(*args):
    """批量转numpy的array
    """
    results = [numpy.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


def load_bert(config_path: str, checkpoint_path: str):
    import passbert.tokenizers as pt
    from passbert.models import build_transformer_model
    tokenizer = pt.PasswordTokenizer()
    bert = build_transformer_model(
        config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)
    return tokenizer, bert


def eval_pwd(pwd: str, tokenizer, bert, max_len: int, needed_ids: List[int]):
    # generate masked password
    probabilities = []
    for i in range(len(pwd)):
        chr_list = list(pwd)
        chr_list[i] = '\t' #  tokenizer._token_mask
        wrappered = [tokenizer._token_start, *chr_list, tokenizer._token_end]
        # last = max_len - len(wrappered)
        # wrappered += [tokenizer._token_pad] * last
        token_ids = []
        for idx, token in enumerate(wrappered):
            if token == '\t':
                got_id = tokenizer._token_mask_id
                masked_index = idx
            else:
                got_id = tokenizer.token_to_id(token)
            token_ids.append(got_id)
        segment_ids = [0] * len(wrappered)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        probas = bert.predict([token_ids, segment_ids])[0]
        # masked_index = i + 1
        masked_item = pwd[i]
        masked_id = tokenizer.token_dict[masked_item]
        tar = probas[masked_index:masked_index + 1]
        # ps = [math.e ** x for x in tar[0]]
        ps = tar[0]
        # print(ps)
        total = sum(ps[_i] for _i in needed_ids)
        # ps = [ps[i] / total for i in needed_ids]
        # print(f"{ps[masked_id]} / {total} = {ps[masked_id] / total}")
        # print(pwd[i], all(x > 0 for x in ps))
        prob = ps[masked_id] / total
        # print(pwd, pwd[i], masked_id, prob, sum(ps), ps)
        probabilities.append(prob)
    return probabilities


def wrapper():
    cli = argparse.ArgumentParser("PassBert Strength Meter")
    cli.add_argument('-p', '--pwd-list', dest='pwd_list',
                     type=str, default=None)
    cli.add_argument('--config', dest='config_path', type=str, default=None)
    cli.add_argument('--checkpoint', dest='checkpoint_path',
                     type=str, default=None)
    cli.add_argument('--max-len', type=int, dest='max_len', default=32)
    cli.add_argument('--threshold', type=float, dest='threshold', default=0.0)
    args = cli.parse_args()
    f_pwd = sys.stdout
    if args.pwd_list is not None:
        f_pwd = open(args.pwd_list, 'r')
    max_len, threshold = args.max_len, args.threshold
    tokenizer, bert = load_bert(
        config_path=args.config_path, checkpoint_path=args.checkpoint_path)
    needed = needed_items()
    needed_ids = [tokenizer.token_dict[token] for token in needed]
    for line in f_pwd:
        line = line.strip('\r\n')
        probabilities = eval_pwd(
            pwd=line, tokenizer=tokenizer, bert=bert, max_len=max_len, needed_ids=needed_ids)
        weak_indices = []
        for i, p in enumerate(probabilities):
            if p > threshold:
                weak_indices.append(i)
        print(line, probabilities)
        pass
    pass


if __name__ == '__main__':
    wrapper()
