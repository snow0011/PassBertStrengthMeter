# PassBertStrengthMeter

Password Strength Metter implemented by PassBert.

It reads a pretained model and evaluate weak characters in a given password.


## Environment

- Python 3.7
- TensorFlow 1.14
- Keras 2.3.1

## Script

### 1. BERT Configuration

We adopt the configuration file to set basic hyper-parameters of our PassBERT model.
Such configurations should be consistent with the ones in pre-training process.
The following parameters are used in our implementation (Path: `ServerModel/config/bert_config_medium.json`).

```json
{
    "hidden_size": 256, 
    "hidden_act": "gelu", 
    "initializer_range": 0.02, 
    "vocab_size": 99, 
    "hidden_dropout_prob": 0.1, 
    "num_attention_heads": 2, 
    "type_vocab_size": 2, 
    "max_position_embeddings": 512, 
    "num_hidden_layers": 4, 
    "intermediate_size": 512, 
    "attention_probs_dropout_prob": 0.1
}
```

### 2. Employ CPG attack

Use the python script `ServerModel/src/evaluating/templates_cmd.py` to employ CPG attack:

```bash
python ServerModel/src/evaluating/templates_cmd.py \
--config CONFIG_PATH \
--checkpoint MODEL_PATH
```

Here, the Configure path `CONFIG_PATH` represents the configuration file in the first script.
The model path `MODEL_PATH` denotes the path of trained password (we have open-sourced in `ServerModel/model/CPG/`).

Then, you can input a template with `*` implying masking characters.
The CPG model will output the top 10 possible paosswords.

### 3. Employ TPG attack

Use the python script `ServerModel/src/tasks/bert4sim.py` to employ TPG attack.
Different from script of CPG attack, the script includes the training process and testing process.

```python
# Edit the Configure class in ServerModel/src/tasks/bert4sim.py
class Config:
    def __init__(self):
        self.maxlen = 32
        self.batch_size = 128
        self.epochs = 3
        self.layers_num = 4
        self.lr = 1e-5
        self.cpu_num = 10
        self.mode = "cmd" # train, cmd, test  <--- Change running mode
        # bert配置
        self.config_path = ''
        self.checkpoint_path = ''
        self.label_path = ''
        self.model_save = ''
        
        self.model_load = MODEL_PATH #              <--- Model path
        self.pwd_pairs = TARGET_INFO #              <--- Test information
        self.output_csv = OUTPUT_CSV #              <--- Output path

```

Here, the `mode` parameter denotes the running mode of current process.
Among them, `train` denotes the re-training process, `cmd` denotes the process to answer user's request in terminal and `test` denotes the process to guess passwords of a specific user.
`MODEL_PATH` denotes the path to trained TPG model (we have open-sourced in `ServerModel/model/TPG/`).
`TARGET_INFO` denotes the path to the information of a specific user. The format should be one line with a source password and a target password sperated by `\t`.

To simply test, you can open `cmd` mode and input a password when everything is done.
Then the TPG model will output the password variants which are similar with the input one.

### 4. Employ ARPG attack

Use the python script `ServerModel/src/evaluating/rule_based_attack.py` to employ ARPG attack.

```bash
python ServerModel/src/evaluating/rule_based_attack.py \
--words WORD_PATH \
--rules RULE_PATH \
--targets TARGET_PATH \
--model MODEL_PATH \
--budget BUDGET \
--save OUTPUT_CSV \
--log LOG_PATH
```

Here, `WORD_PATH` denotes the words used to generate attack.
`RULE_PATH` denotes the path to mangling rules (e.g., PasswordPRO or Generated)
`TARGET_PATH` denotes the path to target passwords (one password per line).
`MODEL_PATH` denotes the path to trained ARPG model (we have open-sourced our PasswordPro model in `ServerModel/model/ARPG/`).
`BUDGET` denotes the threshold when choosing a rule.

## Pre-training

To construct a pre-trained model from specific dataset, we also offer the code and script.
The code is located in `ServerModel/src/pretrain/`.

### 1. Create a dataset

To boost the process of reading passwords, we adopt the `tfrecord` technique to store the password text.
To convert a password file to tfrecord format, we can use the script `ServerModel/src/pretrain/data_utils.py`.

```bash
python ServerModel/src/pretrain/data_utils.py \
-m "roberta" \
-i DATA_PATH \
--seq-len 32 \
--dup-factor 1 \
-s TFRECORD_PATH
```

Here, the `DATA_PATH` denotes the path of training passwords.
`TFRECORD_PATH` denotes the path of output records.

### 2. Set the configuration

To start the pre-training, we should set the hyper-parameters.

```bash
python ServerModel/src/pretrain/pretraining.py \
--lr LEARNING_RATE \
--seq-len SEQLEN \
--batch-size BATCH_SIZE \
-m "roberta" \
--save-ckpt SAVE_CKPT \
-i TFRD_FILE \
-c CONFIG_FILE \
--warmup-steps 31250 \
--total-steps 125000 \
--steps-per-epoch 10000 \
--log LOG_FILE
```

Here, `LEARNING_RATE` denotes the learning rate.
`SEQLEN` denotes the maximal length of passwords.
`BATCH_SIZE` denotes the batch size during pre-training.
`SAVE_CKPT` denotes the saved path of checkpoint file.
`TFRD_FILE` denotes the training data.
`CONFIG_FILE` denotes the BERT paramaters.

## Fine-tuning

To fine-tune the pre-trained model to down-stream tasks, we offer three detailed process of fine-tuning.
Also, we offer a code template for user to tailor the pre-trained model to specific tasks.

### CPG

The CPG task has same process with pre-training.
Thus we reuse the code for convenience.

```bash
python ServerModel/src/pretrain/finetune.py \
-m PRE_TRAINED_MODEL \
--lr LEARNING_RATE \
--seq-len SEQLEN \
--batch-size BATCH_SIZE \
-m "roberta" \
--save-ckpt SAVE_CKPT \
-i TFRD_FILE \
-c CONFIG_FILE \
--warmup-steps 31250 \
--total-steps 125000 \
--steps-per-epoch 10000 \
--log LOG_FILE
```

The parameters are the same with pre-training except we need to set the pre-trained model `PRE_TRAINED_MODEL`.

### TPG 

Use the python script `ServerModel/src/tasks/bert4sim.py` to re-train the TPG model.
Different from script of CPG attack, the TPG script includes the training process and testing process.

```python
# Edit the Configure class in ServerModel/src/tasks/bert4sim.py
class Config:
    def __init__(self):
        self.maxlen = 32 #              <--- Sequence length
        self.batch_size = 128 #              <--- Batch size
        self.epochs = 3 #              <--- Epoch of re-training
        self.layers_num = 4 #              <--- Layers of transformers
        self.lr = 1e-5 #              <--- Learning rate
        self.cpu_num = 10 #        
        self.mode = "train" # train, cmd, test  <--- Change running mode
        # bert配置
        self.config_path = '' #              <---  BERT Configure path
        self.checkpoint_path = ''#              <--- Pre-trained model
        self.label_path = ''#              <--- Path of training labels
        self.model_save = ''#              <--- Model saved path
        
        self.model_load = MODEL_PATH 
        self.pwd_pairs = TARGET_INFOinformation
        self.output_csv = OUTPUT_CSV
```

Then execute the python file to launch the fine-tuning process.

```bash
python ServerModel/src/tasks/bert4sim.py
```

### ARPG

Use the python script `ServerModel/src/tasks/bert4multilabels.py` to re-train an ARPG model.

```bash
python ServerModel/src/tasks/bert4multilabels.py \
--config CONFIG_PATH \
--model MODEL_PATH \
--rules RULES_PATH \
--batch-size BATCH_SIZE \
--steps-per-epoch STEPS_PER_EPOCH \
--epochs EPOCHS \
--training-path TRAINING_PATH \
--save SAVE_H5
```

Here, `CONFIG_PATH` denotes the path to configure file.
`MODEL_PATH` denotes the path to the pre-trained model.
`RULES_PATH` denotes the path to Hashcat rules (e.g., PasswordPro or Generated).
`BATCH_SIZE` denotes the batch size during re-training (default 256).
`STEPS_PER_EPOCH` denotes the steps in per epoch.
`EPOCHS` denotes the total epochs in re-training.
`SAVE_H5` denotes the path to save `.h5` model.
`TRAINING_PATH` denotes the training labels.
You can generate the same format of labels by scripts in [ADaMs](https://github.com/TheAdamProject/adams).

### Customized fine-tuning

To support more password tasks with our pre-trained model, we offer a fine-tuning template to tailor the model to specific tasks.
Use and change the script `ServerModel/src/tasks/bert4task.py` to customize a specific password model.

The following code shows the process to tailor the pre-trained model for a specific task (a classification task for example).
You can change this to adapt your tasks by modify the output layers.

```python
# tailor the model for fine-tuning
def build_model_for_fine_tuning(config_path: str, checkpoint_path: str, num_classes: int):
    # read the weights of transformer blocks
    model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)

    # tailor the model for a classification task
    output = model.output
    output = Dense(512, activation='tanh')(output)
    output = Dense(num_classes, activation='sigmoid')(output)
    output = Lambda(lambda x: x[:, 0])(output)
    model = Model(model.input, output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5))
    model.summary()
    return model
```

Then you can execute the template the start a customized fine-tuning process.
