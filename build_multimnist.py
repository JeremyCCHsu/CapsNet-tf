import tensorflow as tf
import numpy as np
from pdb import set_trace

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tf_example(image_string, classes):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_string])),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=classes)),
            }
        )
    )


def pad_to44(img, shape=[44, 44, 1]):
    '''shape=[h, w]'''
    result = np.zeros([44, 44, 1])
    result[8:36, 8:36, :] = img
    return result


def random_crop(img, shape=36):
    i, j = np.random.choice(range(9), 2)
    return img[i: i + shape, j: j + shape, :]


class MultiMNISTBuilder(object):
    def __init__(self, n_proliferation=1000, num_class=10, shape=[36, 36, 1]):
        self.num_class = num_class
        self.n_per_class, self.remainder = divmod(
            n_proliferation, num_class - 1)

        # a simple TF graph here
        self.np_img = tf.placeholder(tf.uint8, shape=shape)
        self.png_img = tf.image.encode_png(self.np_img)
        
        self.sess = None
        self.tfr_writer = None

    
    def build(self, oFilename, target='training'):
        ''' build training or testing set '''
        if target == 'training':
            x, _ = self._load_mnist()
            N_OUTPUT = 60000000
        elif target == 'testing':
            _, x = self._load_mnist()
            N_OUTPUT = 10000000
        else:
            raise ValueError('Only `training` and `testing` are supported.')

        c = 1
        self.sess = tf.Session()
        tfr_writer = tf.python_io.TFRecordWriter(oFilename)
        for i in range(10):
            x_digit_i = x[i]
            other_class = set(range(10)) - set([i])
            for xi in x_digit_i:
                xi = random_crop(pad_to44(xi)) 
                for j in other_class:
                    Nj = x[j].shape[0]

                    index = np.random.choice(range(Nj), self.n_per_class, replace=False)
                    imgs_from_that_class = x[j][index]

                    for xo in imgs_from_that_class:
                        self._pad_crop_merge_save((xi, i), (xo, j), tfr_writer)
                        print('\rProcessing {:08d}/{:08d}...'.format(c, N_OUTPUT), end='')
                        c += 1

                for _ in range(self.remainder):
                    j = np.random.choice(list(other_class))
                    Nj = x[j].shape[0]
                    index = np.random.choice(range(Nj))
                    xo = x[j][index]

                    self._pad_crop_merge_save((xi, i), (xo, j), tfr_writer)
                    print('\rProcessing {:08d}/{:08d}...'.format(c, N_OUTPUT), end='')
                    c += 1
        print()
        self.sess.close()
        tfr_writer.close()

    def _load_mnist(self):
        (x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()

        if len(x.shape) == 3:
            x = np.expand_dims(x, -1)
            x_t = np.expand_dims(x_t, -1)

        x = [x[y==i] for i in range(self.num_class)]
        x_t = [x_t[y_t==i] for i in range(self.num_class)]
        return x, x_t

    def _pad_crop_merge_save(self, xi_i, xo_j, writer):
        xi, i = xi_i
        xo, j = xo_j
        xo = random_crop(pad_to44(xo))
        combined_img = np.concatenate([xi, xo], -1)
        combined_img = np.max(combined_img, -1, keepdims=True)

        png_encoded = self.sess.run(
            self.png_img, feed_dict={self.np_img: combined_img})

        ex = make_tf_example(png_encoded, [i, j])
        writer.write(ex.SerializeToString())

if __name__ == '__main__':
    builder = MultiMNISTBuilder()
    builder.build('./MultiMNIST_train.tfr', 'training')
    builder.build('./MultiMNIST_test.tfr', 'testing')  # about 2 hr
