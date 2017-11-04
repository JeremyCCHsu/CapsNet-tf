import tensorflow as tf
import os
from datetime import datetime


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


class MultiMNIST(object):
    ''' MNIST batcher
    *padded as 32x32 image 
    '''
    def __init__(self,
        dataset='./MultiMNIST.tfr',
        batch_size=256,
        batch_size_t=100, 
        data_format='channels_first',
        capacity=2048,
        min_after_dequeue=1536,
        # shift=None,  # TODO
        # dimension=28,
        num_threads=4,
        ):
        '''
        Return:
            36x36 image
        '''
        with tf.device('cpu'):
            with tf.name_scope('MNISTInputPipeline'):
                filename_queue = tf.gfile.Glob(dataset)
                filename_queue = tf.train.string_input_producer(filename_queue)

                reader = tf.TFRecordReader()
                _, data = reader.read(filename_queue)

                features = {
                    'image': tf.FixedLenFeature([], dtype=tf.string),
                    'label': tf.FixedLenFeature([2], dtype=tf.int64),
                }
                data = tf.parse_single_example(data, features)

                x = tf.image.decode_png(data['image'])
                x = char2tanh(tf.to_float(x))
                x = tf.reshape(x, [36, 36, 1])  # `batch` doesn't accept Tensor with none shape 
                
                y = data['label']

                if data_format == 'channels_first':
                    x = hwc_to_chw(x)

                self.x, self.y = tf.train.shuffle_batch(
                    [x, y],
                    batch_size=batch_size,
                    capacity=capacity,
                    min_after_dequeue=min_after_dequeue,
                    # enqueue_many=True,
                    num_threads=num_threads,
                )
                # self.x_t, self.y_t = tf.train.batch(
                #     [x_t, y_t],
                #     batch_size=batch_size_t,
                #     capacity=capacity,
                #     # min_after_dequeue=min_after_dequeue,
                #     enqueue_many=True,
                #     num_threads=4,
                # )

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
