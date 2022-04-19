BASE=/disk/yjt/LSTM

# Character level
# arch=${BASE}/result/arch/csdn-all-character-lstm-256.arch
# weight=${BASE}/result/weight/csdn-all-character-lstm-256.h5
# config=/disk/yjt/LSTM/json/character/csdn-train-lstm-256.json
# csv=/disk/yjt/LSTM/result/csv/178-all-guess-character-lstm-256.tsv
# Subword Level
arch=${BASE}/result/arch/csdn-subword-cxt-6.arch
weight=${BASE}/result/weight/csdn-subword-cxt-6.h5
config=/disk/yjt/LSTM/json/context/178-subword-guess-cxt-6.json
csv=/disk/yjt/LSTM/result/csv/youku-guess-subword-cxt-6.tsv
voc=/disk/yjt/LSTM/voc/raw/csdnn_1.8.txt

outfile=${BASE}/PSMBuilder/root/model/

python ${BASE}/PSMBuilder/src/convert_model_to_json.py --arch ${arch} --weight ${weight} --out-dir ${outfile}

out_config=/disk/yjt/LSTM/PSMBuilder/json/csdn-config.json

python ${BASE}/PSMBuilder/src/convert_config_json.py \
--csv $csv \
--config $config \
--ofile $out_config \
--voc $voc

cp $out_config /disk/yjt/LSTM/PSMBuilder/root/model/

