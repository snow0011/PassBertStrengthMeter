# convert the bert tokenizer to json format

import json
import os ,sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from passbert.tokenizers import PasswordTokenizer

def main():
    output_json = "/disk/yjt/BertMeter/PassBertStrengthMeter/model/JS/TPG/vocab.json"
    tokenizer = PasswordTokenizer()
    package = {}
    package['start'] = tokenizer._token_start
    package['end'] = tokenizer._token_end
    package['unk'] = tokenizer._token_unk
    package['pad'] = tokenizer._token_pad
    package['mask'] = tokenizer._token_mask
    package['dict'] = tokenizer.token_dict
    with open(output_json, "w") as f:
        json.dump(package, f)
    pass

if __name__ == '__main__':
    main()