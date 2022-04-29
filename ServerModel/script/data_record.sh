#!/bin/bash

PROJECT_BASE=/disk/yjt/Prepassword
CODE_BASE=${PROJECT_BASE}/src
DATA_PATH=/disk/yjt/data/csdnn_new.txt
TFRD_FILE=${PROJECT_BASE}/tfrecords/demo.tfrecord

python ${CODE_BASE}/pretraining/data_utils.py \
-m "roberta" \
-i ${DATA_PATH} \
--seq-len 32 \
--dup-factor 1 \
-n 10000 \
-s ${TFRD_FILE} 
