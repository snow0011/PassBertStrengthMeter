#!/bin/bash
PYTHON=/disk/cw/anaconda3/envs/prepwd/bin/python

# params for bert4multilabels training
MULTILABEL_PY=/disk/cw/codes/Prepassword/src/tasks/bert4multilabels.py
CONFIG_PATH=/disk/yjt/PasswordSimilarity/model/bert/csdn-rockyou/passbert_csdnroc_model.json
MODEL_PATH=/disk/yjt/PasswordSimilarity/model/bert/csdn-rockyou/passbert_csdnroc_model.ckpt
RULES_PATH=/disk/cw/downloads/InsidePro-PasswordsPro.rule
TRAINING_PATH=/disk/cw/downloads/passpro-csdn-178-result
SAVE_H5=/disk/cw/downloads/roberta-rockyou100w-fine-tuned.h5  # save as .h5 please to simplity the loading process when we predict
BATCH_SIZE=512
STEPS_PER_EPOCH=100
EPOCHS=1

# params for rule based attack
RULE_BASED_ATTACK_PY=/disk/cw/codes/Prepassword/src/evaluating/rule_based_attack.py
WORDS_PATH=/disk/cw/corpora/rockyou-rand3w.txt
RULES_PATH=/disk/cw/downloads/InsidePro-PasswordsPro.rule
TARGETS_PATH=/disk/cw/corpora/rockyou-tar.txt
MODEL_PATH_ADAMS="$SAVE_H5"
BUDGET=0.6
SAVE_HITS=stdout

set -eux
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$MULTILABEL_PY" --config "$CONFIG_PATH" --model "$MODEL_PATH" --rules "$RULES_PATH" \
    --batch-size "$BATCH_SIZE" \
    --steps-per-epoch "$STEPS_PER_EPOCH" \
    --epochs "$EPOCHS" \
    --training-path "$TRAINING_PATH" \
    --save "$SAVE_H5"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$RULE_BASED_ATTACK_PY" \
    -w "$WORDS_PATH" \
    -r "$RULES_PATH" \
    -t "$TARGETS_PATH" \
    -m "$MODEL_PATH_ADAMS" \
    -b "$BUDGET" \
    -s "$SAVE_HITS"