import re
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib
import matplotlib.pyplot as plt

def count_params(checkpoint, excluding_vars=[], verbose=True):
    vdict = checkpoint.get_variable_to_shape_map()
    cnt = 0
    for name, shape in vdict.items():
        skip = False
        for evar in excluding_vars:
            if re.search(evar, name):
                skip = True
        if skip:
            continue
        if verbose:
            print(name, shape)
        cnt += np.prod(shape)
    cnt = cnt / 1e6
    print("Total number of parameters: {:.2f}M".format(cnt))
    return cnt

if __name__ == "__main__":

    checkpoint_path = 'simclr-checkpoints/simclrv2/finetuned_100pct/r50_1x_sk0/'
    checkpoint = tf.train.load_checkpoint(checkpoint_path)
    _ = count_params(checkpoint, excluding_vars=['global_step', "Momentum", 'ema', 'memory', 'head'], verbose=False)