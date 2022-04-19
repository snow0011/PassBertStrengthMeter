import logging
from keras.models import load_model
import argparse
import tensorflowjs as tfjs


class ModelSerializer():
    def __init__(self,
                 weightfile=None):
        self.weightfile = weightfile
        self.model = None

    def load_model(self):
        logging.info('Loading model weights')
        self.model = load_model(self.weightfile)
        logging.info('Done loading model')
        return self.model


def main():
    cli = argparse.ArgumentParser("Convert model to json")
    cli.add_argument("--arch")
    cli.add_argument("--weight")
    cli.add_argument("--out-dir")
    args = cli.parse_args()
    model = ModelSerializer(args.arch, args.weight).load_model()
    tfjs.converters.save_keras_model(model, args.out_dir)
    pass

if __name__ == '__main__':
    main()