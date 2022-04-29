"""
Cleaning rules applied for each word
"""
import argparse
import os


def read_hits_from_rule_based_attack_result(result_file: str):
    uniq_guess_cnt = set()
    acc = 0
    with open(result_file, 'r') as f_result:
        for line in f_result:
            line = line.strip('\r\n')
            word, guess, rule, rank, cnt, _ = line.split('\t')
            if guess not in uniq_guess_cnt:
                uniq_guess_cnt.add(guess)
                cnt = int(cnt)
                acc += cnt
                yield f"{word}\t{guess}\t{rule}\t{rank}\t{cnt}\t{acc}\n"
            pass
        pass
    pass


def wrapper():
    cli = argparse.ArgumentParser(
        "Fixing bugs when generating results, i.e., removing duplicate guessed passwords")
    cli.add_argument("-r", '--result-file', type=str, required=True,
                     help='result file generated from rule_based_attack.py')
    cli.add_argument("-s", '--save', type=str, required=True,
                     help='save fixed result in this file')
    cli.add_argument("--overwrite", action='store_true',
                     help='overwirte the file is `--save` exists')
    args = cli.parse_args()
    if os.path.exists(args.save) and not args.overwrite:
        raise Exception('File already exists and you dont overwrite it.')
    hits = read_hits_from_rule_based_attack_result(args.result_file)
    with open(args.save, 'w') as f_save:
        for hit in hits:
            f_save.write(hit)
    pass


if __name__ == '__main__':
    wrapper()
    pass
