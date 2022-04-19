BASE=/disk/yjt/LSTM

arch=${BASE}/result/arch/csdn-all-character-lstm-256.arch

weight=${BASE}/result/weight/csdn-all-character-lstm-256.h5

tensorflowjs_converter --input_format $arch \
                            path/to/my_model.h5 \
                            path/to/tfjs_target_dir