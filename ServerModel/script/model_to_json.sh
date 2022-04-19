tensorflowjs_converter \
--input_format tf_frozen_model \
--output_node_names='lambda_1/strided_slice' \
/disk/yjt/BertMeter/PassBertStrengthMeter/model/TPG/training_nokeep.pb  \
/disk/yjt/BertMeter/PassBertStrengthMeter/model/JS/TPG


tensorflowjs_converter \
--input_format tf_saved_model \
--output_node_names 'lambda_1/strided_slice:0' \
--saved_model_tags=serve \
/disk/yjt/BertMeter/PassBertStrengthMeter/model/TPG/savedmodel  \
/disk/yjt/BertMeter/PassBertStrengthMeter/model/JS/TPG

tensorflowjs_converter \
--input_format tf_saved_model \
--output_node_names 'lambda_1/strided_slice:0' \
--saved_model_tags=serve \
/disk/yjt/BertMeter/PassBertStrengthMeter/model/CPG/savedmodel  \
/disk/yjt/BertMeter/PassBertStrengthMeter/model/JS/CPG
