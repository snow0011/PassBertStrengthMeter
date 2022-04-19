# PassBertStrengthMeter

## Introduction

PassBertStrengthMeter is the open source library to apply BERT model to password evaluation. We hope the PassBert model will help solve more password understanding tasks.

## Usage

The ServerModel consists the python source code used in our project. We adopt the [bert4keras](https://github.com/bojone/bert4keras) library as our based framework. 

The ClintModel is the client meter to evaluate password strength in user's browser. We deploy our PassBert model by [Tensorflowjs](https://js.tensorflow.org/api/latest/). 

## Environment

- Python 3.7
- TensorFlow 1.14
- TensorFlowjs 3.15.0
- Keras 2.3.1

