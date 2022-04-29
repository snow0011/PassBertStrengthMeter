import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import passbert.tokenizers as pt

def main():
    # print(pt.load_default_pass_vocab())

    # word_list = ["123","456"] + list(pt.load_default_pass_vocab().keys())

    tokenizer = pt.Tokenizer(token_dict="/disk/cw/nlp-guessing/Prepassword/data/vocab.txt", do_lower_case=True)
    token_ids, segment_ids = tokenizer.encode(u"科学技术是第一生产力")
    print(token_ids)
    # print(1, tokenizer.tokenize("123456789."))

    # print("vocaburary size: ",tokenizer._vocab_size)

    # print("max word length: ", tokenizer._word_maxlen)
    pass

if __name__ == '__main__':
    main()

