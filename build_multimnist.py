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

class MultiMNISTIBuilder2(MultiMNISTBuilder):
    def __init__(self, n_proliferation=1000, num_class=10, shape=[36, 36, 1]):
        self.num_class = num_class
        self.n_per_class, self.remainder = divmod(
            n_proliferation, num_class - 1)

        self.n_proliferation = n_proliferation
        # a simple TF graph here
        self.np_img = tf.placeholder(tf.uint8, shape=shape)
        self.png_img = tf.image.encode_png(self.np_img)
        
        self.sess = None
        self.tfr_writer = None

    def build(self, oFilename, target='training'):
        # ''' build training or testing set '''
        # if target == 'training':
        #     x, _ = self._load_mnist()
        #     N_OUTPUT = 60000000
        #     N = 60000
        # elif target == 'testing':
        #     _, x = self._load_mnist()
        #     N_OUTPUT = 10000000
        #     N = 10000
        # else:
        #     raise ValueError('Only `training` and `testing` are supported.')

        (x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()
        N_train = x.shape[0]
        N_test = x_t.shape[0]
        N_proliferate = 1000
        K = 10
        H = 8  # allowable starting index (from top-left)
        W = 8  # allowable starting index (from top-left)

        # for training
        # def (x, y, oFilename, shapes=[K, N, M, P])
        N = N_train
        M = N_proliferate
        P = 3  # index, h_i, w_i

        output = np.zeros([N, M, P], dtype=int)
        index_all = set(range(N))
        index_of_class = set([np.where(y==i)[0] for i in range(K)])
        for k in range(K):
            index_of_other_class = list(index_all - index_of_class[k])
            for i in index_of_class[k]:
                output[i, :, 0] = np.random.choice(
                    index_of_other_class, M, replace=False)
                output[i, :, 1] = np.random.choice(range(H + 1), M)
                output[i, :, 2] = np.random.choice(range(W + 1), M)

        with open('MultiMNIST_index_train.npf', wb) as fp:
            fp.write(output.tostring())

class MultiMNISTIndexBuilder(object):
    def __init__(self, num_class=10, num_proliferate=1000, H=8, W=8):
        ''' # allowable starting index (from top-left)'''
        (x, y), (x_t, y_t) = tf.keras.datasets.mnist.load_data()
        self.num_class = num_class
        self.num_proliferate = num_proliferate

        self.H = H
        self.W = W

        self._build(x, y, 'MultiMNIST_index_train.npf')
        self._build(x_t, y_t, 'MultiMNIST_index_test.npf')
    
    def _get_kmp(self):
        return self.num_class, self.num_proliferate, 3

    def _get_hw(self):
        return self.H, self.W

    def _build(self, x, y, oFilename):
        K, M, P = self._get_kmp()
        N = x.shape[0]
        H, W = self._get_hw()

        output = np.zeros([N, M, P], dtype=np.int64)
        index_all = set(range(N))
        index_of_class = [np.where(y==i)[0] for i in range(K)]
        counter = 1
        for k in range(K):
            index_of_other_class = list(index_all - set(index_of_class[k]))
            for i in index_of_class[k]:
                print('\rProcessing {:5d}/{:5d}'.format(counter, N), end='')
                counter += 1
                # set_trace()
                output[i, :, 0] = np.random.choice(
                    index_of_other_class, M, replace=False)
                output[i, :, 1] = np.random.choice(range(H + 1), M)
                output[i, :, 2] = np.random.choice(range(W + 1), M)
            index_all = set(range(N))
        print()

        with open(oFilename, 'wb') as fp:
            fp.write(output.tostring())


if __name__ == '__main__':
    # builder = MultiMNISTBuilder()
    # builder.build('./MultiMNIST_train.tfr', 'training')
    # builder.build('./MultiMNIST_test.tfr', 'testing')  # about 2 hr
    builder = MultiMNISTIndexBuilder()
