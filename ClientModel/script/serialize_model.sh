
BASE=/disk/yjt/LSTM

arch=${BASE}/result/arch/csdn-all-character-cnn-256.arch

weight=${BASE}/result/weight/csdn-all-character-cnn-256.h5

outfile=${BASE}/PSMBuilder/json/csdn-all-weight.json

model_config=${BASE}/json/character/csdn-train-lstm-256.json

csv=${BASE}/result/csv/youku-all-guess-character-lstm-256.tsv

model_ouput=${BASE}/PSMBuilder/json/csdn-all-model.json

csv_output=${BASE}/PSMBuilder/json/csdn-all-csv.json

model_and_csv=${BASE}/PSMBuilder/json/csdn-all-csv_and_model.json

python ${BASE}/neural_network_cracking/utils/serializer_for_js.py --model-file ${arch} --weight-file ${weight} --ofile ${outfile}

python ${BASE}/neural_network_cracking/utils/extract_js_info.py --config ${model_config} --ofile ${model_ouput}

set -aux

python ${BASE}/neural_network_cracking/utils/extract_prob_to_gn.py --ifile ${csv} --ofile ${csv_output}

jq --slurp '.[0] * .[1]' ${model_ouput} ${csv_output} > ${model_and_csv}


server_dir=${BASE}/neural_network_cracking/js/examples/

cp ${model_and_csv} ${server_dir}

cp ${outfile} ${server_dir}

