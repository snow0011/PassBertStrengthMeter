"""
This file read the templates and converts them to the format of BERT input.
"""
from collections import defaultdict
import os
from re import T
import sys
import pickle
import string
import random
import bisect
import math
import queue
import argparse
from itertools import product
from typing import Dict, Tuple, Set, List
from tqdm import tqdm

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import passbert.tokenizers as pt
from passbert.snippets import to_array
from passbert.models import build_transformer_model


def load_bert(config_path:str, checkpoint_path:str):
    # dict_path = '/disk/cw/nlp-guessing/bert/bert_base_dir/vocab.txt'
    tokenizer = pt.PasswordTokenizer()
    bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)
    return tokenizer, bert


def read_templates(templates_file: str) -> Tuple[Dict[str, Set[Tuple]], Dict[Tuple, Set[str]]]:
    with open(templates_file, 'rb') as f_templates_file:
        res = pickle.load(f_templates_file)
        templates_dict, template2passwords = res
        pass
    template2strpasswords = {}
    for template, passwords in template2passwords.items():
        strpasswords = set()
        for pwd in passwords:
            strpasswords.add("".join(pwd))
        template2strpasswords[template] = strpasswords
    return templates_dict, template2strpasswords


def fetch_replacements(template: Tuple, bert, tokenizer, wanted: List[str], mask_in_template: str = '\t'):
    token_ids = []
    masked_indices = []
    wrappered_template = [tokenizer._token_start, *template, tokenizer._token_end]
    for idx, token in enumerate(wrappered_template):
        if token == mask_in_template:
            got_id = tokenizer._token_mask_id
            masked_indices.append(idx - 1)
        else:
            got_id = tokenizer.token_to_id(token)
        token_ids.append(got_id)
    segment_ids = [0] * len(wrappered_template)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    probas = bert.predict([token_ids, segment_ids])[0]
    repl4template: Dict[int, Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, float]], List[float]]] = {}
    for idx in masked_indices:
        tar = probas[idx:idx+1]
        total = sum([prob for _idx, prob in enumerate(tar[0]) if wanted[_idx] is not None])
        replacements = {wanted[_idx]: (prob / total, -math.log2(prob / total)) for _idx, prob in enumerate(tar[0]) if wanted[_idx] is not None}
        repl_pairs = [(k, ml2p) for k, (_, ml2p) in replacements.items()]
        cum, cum_sum = 0, []
        for k, _ in repl_pairs:
            prob, _ = replacements[k]
            cum += prob
            cum_sum.append(cum)
        # replacements is a dict containing {characters: (probability, minus log probability)}
        # repl_pairs is a list contianing (characters, minus log probability)
        repl4template[idx] = (replacements, repl_pairs, cum_sum)
        pass
    return repl4template


def sample_n(repl4template: Dict[int, Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, float]], List[float]]] , num_samples: int):
    """
    For a template with replacements, sample some passwords to apply Monte Carlo
    """
    sampled_probabilities = []
    for _ in range(num_samples):
        p = 0
        for _, (_, repl_pairs, cum_sum) in repl4template.items():
            rand = random.random()
            rand_idx = bisect.bisect_right(cum_sum, rand)
            try:
                _, ml2p = repl_pairs[rand_idx]
            except IndexError as e:
                print(f"len: {len(repl_pairs)}, rand_idx: {rand_idx}")
                print("cum_sum: ", cum_sum)
                print(e)
                sys.exit(1)
            p += ml2p
        sampled_probabilities.append(p)
    return sampled_probabilities


def generate_compositions(threshold: float, repl4template: Dict[int, Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, float]], List[float]]]):
    ordered_replacements = [sorted(repl4template[k][1], key=lambda x: x[1]) for k in sorted(repl4template.keys())]
    compositions = []
    dfs(ordered_repl=ordered_replacements, cur = [], saved=compositions, threshold=threshold, cur_ml2p=.0)
    return compositions


class ReplItems:
    def __init__(self, replacements: Set[str], ml2p: float) -> None:
        self.replacements = replacements
        self.ml2p = ml2p
        self.next_repl: ReplItems = None
        pass

    def set_next(self, repl_items):
        self.next_repl = repl_items
    pass

class QueueItem:
    def __init__(self, composition: List[ReplItems], ml2p: float, level: int=0) -> None:
        self.composition = composition
        self.ml2p = ml2p
        self.level = level
        pass

    def __lt__(self, other):
        return self.ml2p < other.ml2p

    def get_next(self):
        level = self.level
        next_queue_items: List[QueueItem] = []
        for idx, repl_items in enumerate(self.composition):
            if idx < level:
                continue
            next_repl = repl_items.next_repl
            if next_repl is None:
                continue
            composition = list(self.composition)
            origin  = composition[idx]
            composition[idx] = next_repl
            ml2p = self.ml2p - origin.ml2p + next_repl.ml2p
            
            next_queue_item = QueueItem(composition=composition, ml2p=ml2p, level=idx)
            res = product(*[repl_items.replacements for repl_items in composition])
            # print(f"idx = {idx}, res = ", list(res))
            next_queue_items.append(next_queue_item)
        return next_queue_items
    pass

def ordered_compositions(num_guesses: int, repl4template: Dict[int, Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, float]], List[float]]]):
    ordered_replacements = [repl4template[k][1] for k in sorted(repl4template.keys())]
    ordered_grouped_replacements = []
    # generate repl_items 
    base_ml2p = .0
    for replacements in ordered_replacements:
        ml2p_to_repl = defaultdict(set)
        for repl, ml2p in replacements:
            ml2p_to_repl[ml2p].add(repl)
            pass
        prev_item = None
        for ml2p in sorted(ml2p_to_repl.keys(), reverse=True):
            repl_items = ReplItems(ml2p_to_repl.get(ml2p), ml2p=ml2p)
            repl_items.set_next(repl_items=prev_item)
            prev_item = repl_items
            pass

        ordered_grouped_replacements.append(prev_item)
        base_ml2p += prev_item.ml2p
    queue_item = QueueItem(composition=ordered_grouped_replacements, ml2p=base_ml2p, level=0)
    pq = queue.PriorityQueue()
    pq.put_nowait((queue_item.ml2p, queue_item))
    generated_guesses = 0
    while not pq.empty():
        _, q_item = pq.get_nowait()
        composition = q_item.composition
        res = product(*[repl_items.replacements for repl_items in composition])
        for guess in res:
            generated_guesses += 1
            yield guess, q_item.ml2p
            if generated_guesses >= num_guesses:
                return
        next_q_items = q_item.get_next()
        for nqi in next_q_items:
            pq.put_nowait((nqi.ml2p, nqi))
        pass
    pass


def generate_guesses(template: List[str], compositions: List[Tuple[List[str], float]], masked_indices: List[int]):
    for comp, ml2p in compositions:
        tplt = list(template)
        for mi, itm in zip(masked_indices, comp):
            tplt[mi] = itm
        yield tplt, ml2p


def dfs(ordered_repl: List[List[Tuple[str, float]]], cur: List[str], saved: List[Tuple[List[str], float]], threshold: float, cur_ml2p: float):
    index = len(cur)
    if cur_ml2p > threshold:
        return
    if index == len(ordered_repl):
        saved.append((list(cur), cur_ml2p))
        return
    max_ml2p = threshold - cur_ml2p
    for repl, ml2p in ordered_repl[index]:
        if ml2p > max_ml2p:
            break
        cur.append(repl)
        dfs(ordered_repl, cur, saved, threshold, cur_ml2p + ml2p)
        cur.pop()
        pass
    pass


def calc_prob(pwd: List[str], repl4template: Dict[int, Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, float]], List[float]]]):
    final_ml2p = .0
    for masked_idx, (replacements, _, _) in repl4template.items():
        try:
            itm = pwd[masked_idx]
        except IndexError as e:
            print(e)
            print(f"{pwd}, {masked_idx}")
            sys.exit(2)
        _, ml2p = replacements[itm]
        final_ml2p += ml2p 
    return final_ml2p


def get_wanted_items():
    LETTERS = string.ascii_letters
    NUMBERS = string.digits
    SPECIALS = string.punctuation
    characters = LETTERS + NUMBERS + SPECIALS
    items = list(characters)
    return items


def monte_carlo_preprocess(sampled_probabilities: List[float]):
    logn = math.log2(len(sampled_probabilities))
    values = [2 ** (v - logn) for v in sampled_probabilities]
    cum = 0
    positions = []
    for v in values:
        cum += v
        positions.append(cum)
    return positions


def wrapper(**kwargs):
    config_path = kwargs.get("config_path", '/disk/cw/nlp-guessing/models/bert_config.json')
    checkpoint_path= kwargs.get("checkpoint_path", '/disk/cw/nlp-guessing/models/passbert_bert_model_test.ckpt')
    threshold = kwargs.get('threshold', 10000)
    templates_file = kwargs.get('templates_file', '/disk/cw/datasets/templates4neopets/sampled.pickle')
    guesses_path = kwargs.get("guesses_path", '/disk/cw/datasets/templates4neopets/guesses')
    tokenizer, bert = load_bert(config_path=config_path, checkpoint_path=checkpoint_path)
    templates_dict, template2passwords = read_templates(templates_file=templates_file)
    mask_in_template = '\t'
    items = get_wanted_items()
    wanted_items = [None for _ in range(tokenizer._vocab_size + 1)]
    for itm in items:
        token2id = tokenizer.token_dict[itm]
        wanted_items[token2id] = itm
    # results = {}
    if not os.path.exists(guesses_path):
        os.mkdir(guesses_path)
    for cls_name, templates in templates_dict.items():
        if cls_name is not "common":
            continue
        print("Class name: ", cls_name)
        guesses_path4cls_name = os.path.join(guesses_path, cls_name)
        if not os.path.exists(guesses_path4cls_name):
            os.mkdir(guesses_path4cls_name)
        # results4templates = {}
        for template in tqdm(templates, desc=f"{cls_name:>10}"):
            """
            {'rare': {'\t\ta\t\t\tlla': [('kiaavella', 18.79692094296601, 51584995.28579091)], 
            '\t91\t51\t\t\t': [('S9125112D', 36.547169416524895, 55313390.349879354)]}, 
            'super-rare': {'1\t55\t\t\ta\t': [('1255474as', 29.39285448961602, 77565616.0400397)]}}
            """
            repl4template = fetch_replacements(template=template, bert=bert, tokenizer=tokenizer, wanted=wanted_items, mask_in_template=mask_in_template)
            """
            sampled_probabilities = sample_n(repl4template=repl4template, num_samples=num_samples)
            sorted_probabilities = sorted(sampled_probabilities)
            positions = monte_carlo_preprocess(sampled_probabilities=sampled_probabilities)
            passwords = template2passwords[template]
            pwd_info_list = []
            for pwd in passwords:
                ml2p = calc_prob(pwd=pwd, repl4template=repl4template)
                pos_idx = bisect.bisect_right(sorted_probabilities, ml2p)
                pos = positions[pos_idx - 1] if pos_idx > 0 else 1
                pwd_info_list.append((pwd, ml2p, pos))
                break
            pwd_info_list = sorted(pwd_info_list, key=lambda x: x[2])
            results4templates[template] = pwd_info_list
            
            # you may convert guess number threshold to probability threshold, or directly provide a probability threshold
            if convert_to_probability:
                n_idx = bisect.bisect_right(positions, threshold)
                print(f"original threshold is {threshold}, index is {n_idx}, pos is {positions[n_idx]}, pos -1 is {positions[n_idx - 1]}")
                print(f"prob is {sorted_probabilities[:n_idx]}, len(sorted_prob) = {len(sorted_probabilities)}")
                print(f"prob for the last is {sorted_probabilities[-7:]}")
                threshold = sorted_probabilities[n_idx] if n_idx < len(sorted_probabilities) else len(sorted_probabilities) - 1
            print(f"threshold is {threshold}")

            unsorted_compositions = generate_compositions(threshold=17, repl4template=repl4template)
            unsorted_guesses = generate_guesses(template=template, compositions=unsorted_compositions, masked_indices=sorted(repl4template.keys()))
            """
            compositions = ordered_compositions(num_guesses=threshold, repl4template=repl4template)
            guesses = generate_guesses(template=template, compositions=compositions, masked_indices=sorted(repl4template.keys()))
            # guesses = generate_guesses(template=template, compositions=compositions, masked_indices=sorted(repl4template.keys()))
            filename = ''.join(template).replace(mask_in_template, ' ')
            guesses_file = os.path.join(guesses_path4cls_name, f"guesses[{filename}].txt")
            cracked_file = os.path.join(guesses_path4cls_name, f"cracked[{filename}].txt")
            passwords = template2passwords[template]
            with open(guesses_file, 'w') as f_guesses, open(cracked_file, 'w') as f_cracked:
                # print(guesses)
                rank = 0
                for guess, ml2p in guesses:
                    # print(guess)
                    rank += 1
                    guess = ''.join(guess)
                    if guess in passwords:
                        f_cracked.write(f"{guess}\t{ml2p}\t{rank}\n")
                    f_guesses.write(f"{guess}\t{ml2p}\n")
                f_guesses.flush()
                f_cracked.flush()
                """
                for ung, ml2p in unsorted_guesses:
                    f_guesses.write(f"{''.join(ung)}\t{ml2p}\n")
                f_guesses.flush()
                """
                pass
        # results[cls_name] = results4templates
    """
    with open(save_results, 'wb') as f_save_results:
        pickle.dump(results, f_save_results)    
    print(f"Done!", file=sys.stderr)
    """
    pass


def wrapper_with_cli():
    cli = argparse.ArgumentParser("Enumerating password candidates based on templates")
    cli.add_argument('--config', dest='config', type=str, required=True, help='config path of bert')
    cli.add_argument('--checkpoint', dest='checkpoint', type=str, required=True, help='checkpoint path of bert')
    cli.add_argument('-n', '--num-guesses', dest='num_guesses', type=int, required=True, help='the number of guesses generated')
    cli.add_argument('-t', '--templates', dest='templates_file', type=str, required=True, help='templates file used to generate guesses')
    cli.add_argument('-s', '--save-guesses-folder', dest='save_folder', type=str, required=True, help='save guesses in this folder')
    args = cli.parse_args()
    wrapper(config_path=args.config, checkpoint_path=args.checkpoint, templates_file=args.templates_file,
            threshold=args.num_guesses, guesses_path=args.save_folder)
    pass


if __name__ == '__main__':
    wrapper_with_cli()
    pass