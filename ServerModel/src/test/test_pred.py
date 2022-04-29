import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import passbert.tokenizers as pt
from passbert.snippets import to_array
from passbert.models import build_transformer_model

config_path =  '/disk/cw/nlp-guessing/models/bert_config.json'
checkpoint_path= '/disk/cw/nlp-guessing/models/manysame-bert4keras.ckpt'

tokenizer = pt.PasswordTokenizer()
bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)


def wrapper():
    print(f"vocab_size: {tokenizer._vocab_size}")
    token_ids, segment_ids = tokenizer.encode("passwordpassword")
    print(f"before masked token_ids = {token_ids}")
    idx = 10
    token_ids[idx] = tokenizer._token_mask_id
    print("masked token_ids = ", token_ids)
    print(f"token length: {len(token_ids)}")
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    print(f"token length: {len(token_ids)}")
    probas = bert.predict([token_ids, segment_ids])[0]
    tar = probas[idx:idx+1]
    print(f"len(probas) = {len(probas)}, tar argmax = {tar.argmax(axis=1)}")
    print("decoded", tokenizer.decode(tar.argmax(axis=1)), ':hello')
    pass


if __name__ == '__main__':
    wrapper()