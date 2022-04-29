# We apply edit distance to password similarity detect.
# For example, the source password is intention and the target password is execution
# Then we first get:
# inte*ntion
# *execution
# dsskiskkkk
# *nt*iu****
#
# The second step, eliminate "insert" by x(replace current letter by two letters)
# 
# inte*ntion
# *execution
# dssxkskkkk
# *ntecu****

import itertools
import json
import string
import numpy as np
import argparse

def load_password_letter():
    LETTERS = string.ascii_letters
    NUMBERS = string.digits
    SPECIALS = string.punctuation
    SPACE = " "
    return LETTERS + NUMBERS + SPECIALS + SPACE

def load_inplace_trans_dict():
    actions = {}
    # load simple operations
    actions[('k',None)] = len(actions)
    actions[('d',None)] = len(actions)
    letters = load_password_letter()
    for ch in letters:
        actions[('s',ch)] = len(actions)
    # load complex operations
    for ch in letters:
        for ch2 in letters:
            actions[('x',ch+ch2)] = len(actions)
    return actions

def load_reverse_trans_dict():
    mapper = load_inplace_trans_dict()
    ans = {}
    for i,item in mapper.items():
        ans[item] = i
    return ans

def find_med_backtrace(str1, str2, cutoff=-1):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example: ('password', 'Passw0rd') -> 2.0, [('s', 'P', 0), ('s', '0', 5)]
    '''
    # op_arr_str = ["d", "i", "c", "s"]

    # Definitions:
    n = len(str1)
    m = len(str2)
    D = np.full((n + 1, m + 1), np.inf)
    trace = np.full((n + 1, m + 1), None)
    trace[1:, 0] = list(zip(range(n), np.zeros(n, dtype=int)))
    trace[0, 1:] = list(zip(np.zeros(m, dtype=int), range(m)))
    # Initialization:
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)

    # Fill the matrices:
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete = D[i - 1, j] + 1
            insert = D[i, j - 1] + 1
            if (str1[i - 1] == str2[j - 1]):
                sub = np.inf
                copy = D[i - 1, j - 1]
            else:
                sub = D[i - 1, j - 1] + 1
                copy = np.inf
            op_arr = [delete, insert, copy, sub]
            D[i, j] = np.min(op_arr)
            op = np.argmin(op_arr)
            if (op == 0):
                # delete, go down
                trace[i, j] = (i - 1, j)
            elif (op == 1):
                # insert, go left
                trace[i, j] = (i, j - 1)
            else:
                # copy or subsitute, go diag
                trace[i, j] = (i - 1, j - 1)
    # print(trace)
    # Find the path of transitions:
    i = n
    j = m
    cursor = trace[i, j]
    path = []
    while (cursor is not None):
        # 3 possible directions:
        #         print(cursor)
        if (cursor[0] == i - 1 and cursor[1] == j - 1):
            # diagonal - sub or copy
            if (str1[cursor[0]] != str2[cursor[1]]):
                # substitute
                path.append(("s", str2[cursor[1]], cursor[0]))
            i = i - 1
            j = j - 1
        elif (cursor[0] == i and cursor[1] == j - 1):
            # go left - insert
            path.append(("i", str2[cursor[1]], cursor[0]))
            j = j - 1
        else:
            # (cursor[0] == i - 1 and cursor[1] == j )
            # go down - delete
            path.append(("d", None, cursor[0]))
            i = i - 1
        cursor = trace[cursor[0], cursor[1]]
        # print(len(path), cursor)
    md = D[n, m]
    del D, trace
    return md, list(reversed(path))



# Decoder - given a word and a path of transition, recover the final word:
def path2word(word, path):
    '''This function decodes the word in which the given path transitions the input
    word into.  Input parameters: original word, transition path Output: decoded
    word

    '''
    if not path:
        return word
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if (j < path_len and path[j][2] == i):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            final_word.append(word[i])
            i += 1
    return ''.join(final_word)

def inplace_edit(pwd1, pwd2):
    path = find_med_backtrace(pwd1, pwd2)
    dist = path[0]
    edits = path[1]
    is_arrive = True
    # inplace_op = [['k',None] for _ in range(len(pwd1))]
    inplace_op = [['s',ch] for ch in pwd1]
    for op in edits:
        if op[0] == 's' or op[0] == 'd':
            inplace_op[op[2]] = [op[0], op[1]]
    for op in edits:
        if op[0] != 'i':
            continue
        pos = op[2]
        char = op[1]
        if pos >= len(pwd1):
            inplace_op.append(['s',char])
            continue
        if inplace_op[pos][0] == 'k':
            inplace_op[pos] = ['x',char+pwd1[pos]]
        elif inplace_op[pos][0] == 's':
            inplace_op[pos] = ['x',char+inplace_op[pos][1]]
        else:
            is_arrive = False
    return (dist, path, inplace_op if is_arrive else [], is_arrive)

def pair_reader_temp(csv_file):
    for line in open(csv_file, "r"):
        line = line.strip().split(",",1)
        if len(line[1]) <= 4 or line[1][0] != '"':
            continue
        print(line[1][1:-1].replace('""','"'))
        if len(line) == 2:
            yield (line[0], json.loads(line[1][1:-1].replace('""','"')))

def pair_reader(csv_file):
    for line in open(csv_file, "r"):
        line = line.strip().split(",",1)
        if len(line[1]) <= 4:
            continue
        if len(line) == 2:
            yield (line[0], json.loads(line[1]))

def encode_inplace_edit(mapper, path):
    res = [0 for _ in range(len(path))]
    for i, op in enumerate(path):
        res[i] = mapper[tuple(op)]
    return res

def decode_inplace_edit(mapper, path):
    res = []
    for op in path:
        res.append(mapper[op])
    return res

def recover_inplace_edit(pwd, decode_path):
    res = [ch for ch in pwd+"   "]
    for i, item in enumerate(decode_path):
        if item[0] == 'k':
            continue
        if item[0] == 's' or item[0] == 'x':
            res[i] = item[1]
        if item[0] == 'd':
            res[i] = ''
    return "".join(res).strip(' ')

class Filter:
    def __init__(self):
        self.less_4 = 0
        self.less_4_valid = 0
        self.total = 0
        self.total_valid = 0
        pass

    def filter(self, pwd,record):
        if not record[3]:
            return False
        if record[0] >= 4:
            return False
        if len(record[2]) - len(pwd) > 3:
            return False
        return True

    def add(self, record):
        self.total += 1
        if record[0] <= 4:
            self.less_4 += 1
            if record[3]:
                self.less_4_valid += 1
        if record[3]:
                self.total_valid += 1

    def show(self):
        print(f"Total: {self.total}")
        print(f"Total valid: {self.total_valid}")
        print(f"Less than 4: {self.less_4}")
        print(f"Less than 4 valid: {self.less_4_valid}")

def test():
    pwd1 = "1q2w3e4r5t"
    pwd2 = "qwertyu"
    # pwd1 = "intention"
    # pwd2 = "execution"
    res = inplace_edit(pwd1, pwd2)
    mapper = load_inplace_trans_dict()
    de_mapper = load_reverse_trans_dict()
    en = encode_inplace_edit(mapper, res[2])
    de = decode_inplace_edit(de_mapper, en)
    rec = recover_inplace_edit(pwd1, de)
    print(pwd1)
    print(pwd2)
    print(rec)
    print("Path: ",res[1])
    print("Inplace: ",res[2])
    print("Arrive: ",res[3])

def main():
    cli = argparse.ArgumentParser("Get inplace edit path of source and target password")
    cli.add_argument("-c","--csv",dest="csv",help="csv file of name and passwords pair",required=True)
    cli.add_argument("-o","--output",dest="output",help="Output file",required=True)
    cli.add_argument("--human",dest="human",action="store_true",default=False)
    args = cli.parse_args()
    output = open(args.output, "w")
    stats = Filter()
    mapper = load_inplace_trans_dict()
    for i,items in enumerate(pair_reader(args.csv)):
        name,pwds = items
        if len(pwds) > 10:
            continue
        for src, tar in itertools.permutations(pwds, 2):
            if len(src) > 30 or len(tar) > 30:
                continue
            res = inplace_edit(src, tar)
            stats.add(res)
            if stats.filter(src, res):
                if args.human:
                    output.write(f"{src}\t{tar}\t{res[2]}\n")
                else:
                    output.write(f"{src}\t{tar}\t{encode_inplace_edit(mapper, res[2])}\n")
            del res
        if i % 1000 == 999:
            print("Batch ", i)
    stats.show()
    output.close()
    # print(len(load_inplace_trans_dict()))
    test()

if __name__ == '__main__':
    main()