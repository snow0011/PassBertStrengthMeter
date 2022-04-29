#!/bin/bash

PROJECT_BASE=/disk/yjt/Prepassword
CODE_BASE=${PROJECT_BASE}/src
SAVE_CKPT=/disk/yjt/Prepassword/model/passbert/roc_finetune.ckpt
PRE_CKPT=/disk/yjt/Prepassword/model/passbert/roc-coverted.ckpt
TFRD_FILE=${PROJECT_BASE}/tfrecords/rockyou.tfrecord
CONFIG_FILE=${PROJECT_BASE}/config/bert_config.json
LOG_FILE=${PROJECT_BASE}/result/log/log.txt

CUDA_VISIBLE_DEVICES=-1 python ${CODE_BASE}/pretraining/finetune.py \
-m "roberta" \
--save-ckpt ${SAVE_CKPT} \
-i ${TFRD_FILE} \
-c ${CONFIG_FILE} \
--warmup-steps 31250 \
--total-steps 125000 \
--steps-per-epoch 10000 \
-pre ${PRE_CKPT} \
--log ${LOG_FILE}

