import tensorflow as tf
import os
from datetime import datetime
import numpy as np

def char2tanh(x):
    ''' [0, 255] -> [-1., 1.] '''
    return x / 127.5 - 1.

def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])

def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])

def hwc_to_chw(x):
    return tf.transpose(x, [2, 0, 1])

def chw_to_hwc(x):
    return tf.transpose(x, [1, 2, 0])


def squash(x):
    '''
    `x`: [n, J, V]
    '''
    with tf.name_scope('Squash'):
        x2 = tf.square(x)
        x_sn = tf.reduce_sum(x2, -1, keep_dims=True)  # [n, J], squared norm
        v = x_sn / (1. + x_sn) * x / tf.sqrt(x_sn)
        return v


def make_linear_perturbation(J, V, R, m=.25):
    '''
    R = 21 is `int`
    J = 10
    V = 16
    '''
    with tf.name_scope('MakeLinearInterpBasis'):
        I = tf.expand_dims(tf.eye(V), -1)  # [V, V, 1]
        a = m * (tf.range(0, R, dtype=tf.float32) / ((R - 1) / 2) - 1)
        I = I * a  # [V, V, 21]
        I = tf.transpose(I, [2, 1, 0])  # [R, V, V]
        I = tf.reshape(I, [-1, V])  # [V*21, V]
        I = tf.expand_dims(I, 1)
        I = tf.tile(I, [1, J, 1])  # [V*21, 10, V]
        y = tf.ones([R * V, ], tf.int32)
        return I, y

class mnist(object):
    ''' MNIST batcher
    *padded as 32x32 image 
    '''
    def __init__(self,
        dataset=None,
        batch_size=32,
        batch_size_t=100, 
        data_format='channels_first',
        capacity=512,
        min_after_dequeue=256,
        shift=None,  # TODO
        dimension=28,
        num_threads=4,
        ):
        with tf.device('cpu'):
            with tf.name_scope('MNISTInputPipeline'):
                (x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()
                y, y_t = tf.cast(y, tf.int64), tf.cast(y_t, tf.int64)        
                x, x_t = char2tanh(x), char2tanh(x_t)
                x, x_t = tf.constant(x, dtype=tf.float32), tf.constant(x_t, dtype=tf.float32)
                x, x_t = tf.expand_dims(x, -1), tf.expand_dims(x_t, -1)
                # exemplar data
                self.example = tf.gather(x, [1, 3, 5, 7, 2, 0, 13, 15, 17, 4])

                if dimension == 32:  # TODO
                    x = tf.keras.backend.spatial_2d_padding(x, ([2, 2], [2, 2]))
                    x_t = tf.keras.backend.spatial_2d_padding(x_t, ([2, 2], [2, 2]))
                if data_format == 'channels_first':
                    # x, x_t = tf.expand_dims(x, 1), tf.expand_dims(x_t, 1)
                    x, x_t = nhwc_to_nchw(x), nhwc_to_nchw(x_t)
                # else:
                #     x, x_t = tf.expand_dims(x, -1), tf.expand_dims(x_t, -1)
                self.x, self.y = tf.train.shuffle_batch(
                    [x, y],
                    batch_size=batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=True,
                    num_threads=num_threads,
                )
                self.x_t, self.y_t = tf.train.batch(
                    [x_t, y_t],
                    batch_size=batch_size_t,
                    capacity=capacity,
                    # min_after_dequeue=min_after_dequeue,
                    enqueue_many=True,
                    num_threads=num_threads,
                )


class MultiMNISTIndexReader(object):
    def __init__(self,
        train_index='MultiMNIST_index_train.npf',  # TODO
        batch_size=32,
        batch_size_t=100, 
        data_format='channels_first',
        capacity=512,
        min_after_dequeue=256,
        # shift=None,  # TODO
        # dimension=28,
        num_threads=4,
        ):

        with tf.device('cpu'):
            with tf.name_scope('MNISTInputPipeline'):
                (x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()
                y, y_t = tf.cast(y, tf.int64), tf.cast(y_t, tf.int64)        
                # x, x_t = char2tanh(x), char2tanh(x_t)
                x, x_t = tf.constant(x, dtype=tf.float32), tf.constant(x_t, dtype=tf.float32)
                x, x_t = tf.expand_dims(x, -1), tf.expand_dims(x_t, -1)
                # exemplar data
                self.example = tf.gather(x, [1, 3, 5, 7, 2, 0, 13, 15, 17, 4])
                
                # Training
                i = np.fromfile(train_index, np.int64)
                i = np.reshape(i, [x.shape[0], -1, 3])
                i = tf.constant(i[:, :, 0])  # [N=60K, M=1K]

                x_i, y_i, i_i = tf.train.slice_input_producer([x, y, i])  # i_i: [M=1K,]
                ii = tf.random_uniform([], 0, 1000, tf.int64)
                ii = i_i[ii] # []

                x_j = x[ii]  # [28, 28, 1]
                y_j = y[ii]  # []
                y_j = tf.reshape(y_j, [1,])
                y_i = tf.reshape(y_i, [1,])

                x_i = tf.image.resize_image_with_crop_or_pad(x_i, 44, 44)
                x_i = tf.random_crop(x_i, size=[36, 36, 1])
                x_i = char2tanh(x_i)

                x_j = tf.image.resize_image_with_crop_or_pad(x_j, 44, 44)
                x_j = tf.random_crop(x_j, size=[36, 36, 1], seed=9527)  # TODO
                x_j = char2tanh(x_j)

                x_merge = tf.concat(
                    [
                        tf.expand_dims(x_i, -1),
                        tf.expand_dims(x_j, -1)
                    ], 
                    -1
                )
                x_merge = tf.reduce_max(x_merge, -1) # [b=1, h, w, c]
                y_merge = tf.concat([y_i, y_j], -1)

                self.x, self.y, self.xi, self.xj = tf.train.shuffle_batch(
                    [x_merge, y_merge, x_i, x_j],
                    batch_size=batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    # enqueue_many=True,
                    num_threads=num_threads,
                )


                # pad_and_crop (x_i, x_j)   # TODO should also store the (h, w) index of x_i (?)
                # max_sum (x_i, x_j)


# class MultiMNIST(object):
#     ''' MNIST batcher
#     *padded as 32x32 image 
#     '''
#     def __init__(self,
#         dataset='./MultiMNIST.tfr',
#         batch_size=256,
#         batch_size_t=100, 
#         data_format='channels_first',
#         capacity=2048,
#         min_after_dequeue=1536,
#         # shift=None,  # TODO
#         # dimension=28,
#         num_threads=4,
#         ):
#         '''
#         Return:
#             36x36 image
#         '''
#         with tf.device('cpu'):
#             with tf.name_scope('MNISTInputPipeline'):
#                 filename_queue = tf.gfile.Glob(dataset)
#                 filename_queue = tf.train.string_input_producer(filename_queue)

#                 reader = tf.TFRecordReader()
#                 _, data = reader.read(filename_queue)

#                 features = {
#                     'image': tf.FixedLenFeature([], dtype=tf.string),
#                     'label': tf.FixedLenFeature([2], dtype=tf.int64),
#                 }
#                 data = tf.parse_single_example(data, features)

#                 x = tf.image.decode_png(data['image'])
#                 x = char2tanh(tf.to_float(x))
#                 x = tf.reshape(x, [36, 36, 1])  # `batch` doesn't accept Tensor with none shape 
                
#                 y = data['label']

#                 if data_format == 'channels_first':
#                     x = hwc_to_chw(x)

#                 self.x, self.y = tf.train.shuffle_batch(
#                     [x, y],
#                     batch_size=batch_size,
#                     capacity=capacity,
#                     min_after_dequeue=min_after_dequeue,
#                     # enqueue_many=True,
#                     num_threads=num_threads,
#                 )
#                 # self.x_t, self.y_t = tf.train.batch(
#                 #     [x_t, y_t],
#                 #     batch_size=batch_size_t,
#                 #     capacity=capacity,
#                 #     # min_after_dequeue=min_after_dequeue,
#                 #     enqueue_many=True,
#                 #     num_threads=4,
#                 # )

def validate_log_dirs(args):
    ''' Create a default log dir (if necessary) '''
    def get_default_logdir(logdir_root):
        STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
        logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
        print('Using default logdir: {}'.format(logdir))
        return logdir

    if args.logdir and args.restore_from:
        raise ValueError(
            'You can only specify one of the following: ' +
            '--logdir and --restore_from')

    if args.logdir and args.log_root:
        raise ValueError(
            'You can only specify either --logdir or --logdir_root')

    if args.logdir_root is None:
        logdir_root = 'logdir'

    if args.logdir is None:
        logdir = get_default_logdir(logdir_root)

    if args.msg:
        logdir += args.msg

    # Note: `logdir` and `restore_from` are exclusive
    if args.restore_from is None:
        restore_from = logdir
    else:
        restore_from = args.restore_from

    return {
        'logdir': logdir,
        'logdir_root': logdir_root,
        'restore_from': restore_from,
    }
