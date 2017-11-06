import json

import numpy as np
import tensorflow as tf

from helper import *
from capsule import CapsuleMultiMNIST

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'arch', 'architecture-multimnist.json', 'network architecture')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'root of log dir')
tf.app.flags.DEFINE_string('logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'restore from dir (not from *.ckpt)')
tf.app.flags.DEFINE_string('msg', '-MultiMNIST', 'Additional message')


def main():
    with open(args.arch) as fp:
        arch = json.load(fp)
    data = MultiMNISTIndexReader(
        train_index='MultiMNIST_index_train.npf',
        batch_size=arch['training']['batch_size'],
        data_format='channels_last',
        capacity=2**10, min_after_dequeue=2**9
    )
    dirs = validate_log_dirs(args)
    arch.update({'logdir': dirs['logdir']})
    net = CapsuleMultiMNIST(arch=arch)
    loss = net.loss(data.x, data.y, data.xi, data.xj)
    # net.inspect(data.example)
    # loss_t = net.loss(data.x_t, data.y_t)
    net.train(loss, loss_t=None)

if __name__ == '__main__':
    main()
