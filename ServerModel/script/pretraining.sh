#!/bin/bash

PROJECT_BASE=/disk/yjt/Prepassword
CODE_BASE=${PROJECT_BASE}/src
SAVE_CKPT=/disk/yjt/Prepassword/model/demo/demo.ckpt
TFRD_FILE=${PROJECT_BASE}/tfrecords/demo.tfrecord
CONFIG_FILE=${PROJECT_BASE}/config/bert_config.json
LOG_FILE=${PROJECT_BASE}/result/log/log.txt

CUDA_VISIBLE_DEVICES=-1 python ${CODE_BASE}/pretraining/pretraining.py \
-m "roberta" \
--save-ckpt ${SAVE_CKPT} \
-i ${TFRD_FILE} \
-c ${CONFIG_FILE} \
--warmup-steps 31250 \
--total-steps 125000 \
--steps-per-epoch 10000 \
--log ${LOG_FILE}

