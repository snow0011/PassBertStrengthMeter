#!/bin/bash

BASE=/disk/yjt/LSTM

weight=${BASE}/result/weight/csdn-character.h5

echo $weight

python ${BASE}/PSMBuilder/src/check_h5.py ${weight}