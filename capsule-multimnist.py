import json

import numpy as np
import tensorflow as tf

from helper import MultiMNISTIndexReader, validate_log_dirs

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'arch', 'architecture-multimnist.json', 'network architecture')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'root of log dir')
tf.app.flags.DEFINE_string('logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'restore from dir (not from *.ckpt)')
tf.app.flags.DEFINE_string('msg', '-MultiMNIST', 'Additional message')

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
        a = m * (tf.range(0, R, dtype=tf.float32) / ((R - 1)/2) - 1)
        I = I * a  # [V, V, 21]
        I = tf.transpose(I, [2, 1, 0])  # [R, V, V]
        I = tf.reshape(I, [-1, V])  # [V*21, V]
        I = tf.expand_dims(I, 1)
        I = tf.tile(I, [1, J, 1])  # [V*21, 10, V]
        y = tf.ones([R * V,], tf.int32)
        return I, y


class CapsuleNet(object):
    def __init__(self, arch):
        self.arch = arch
        self._C = tf.make_template('Recognizer', self._recognize)
        self._G = tf.make_template('Generator', self._generate)

    def _get_shape_JDUV(self):
        J = self.arch['num_class']  # 10
        D = self.arch['Primary Capsule']['depth']  # 32
        U = self.arch['Primary Capsule']['dim']  # 8
        V = self.arch['Digit Capsule']['dim']  # 16
        return J, D, U, V

    def _recognize(self, x):
        '''`x`: [b, h, w, c]
        '''
        J, D, U, V = self._get_shape_JDUV()
        net = self.arch['recognizer']
        assert D * U == net['output'][-1]

        self.W = tf.get_variable('W', shape=[J, V, U], dtype=tf.float32)

        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            if i + 1 == len(net['output']):
                activation = None
            else:
                activation = tf.nn.relu
            x = tf.layers.conv2d(x, o, k, s, activation=activation)

        S = tf.shape(x) # [n, h', w', c']
        I = S[1] * S[2] * D

        primary = tf.reshape(x, [-1, S[1], S[2], D, U])
        primary = tf.reshape(primary, [-1, I, U])

        u = primary  # NOTE: iteratively process the previous capsule `u`
        B = tf.zeros([tf.shape(x)[0], I, J])  # the "attention" matrix
        for _ in range(self.arch['Digit Capsule']['nRouting']):
            v, B = self._stack(u, B)

        return v


    def _stack(self, u, B):
        '''
        I, J, Nu, Nv = 6*6*32, 10, 8, 16
        Input:
            `u`: [n, I, Nu]
            `b`: [n, I, J]
        Return:
            `V`: [n, J, V]
            `B`: [n, I, J]
        '''
        with tf.name_scope('Capsule'):
            uji = tf.tensordot(u, self.W, [[2], [2]])  # [n, I, U] dot [J, V, U] => [n, I, J, V]

            C = tf.nn.softmax(B, dim=1)  # [n, I, J]
            C = tf.expand_dims(C, -1)    # [n, I, J, 1] (necessary for broadcasting)

            S = tf.reduce_sum(C * uji, 1)  # [n, J, V]
            v_ = squash(S)                 # [n, J, V]
            v = tf.expand_dims(v_, 1)      # [n, 1, J, V]

            dB = tf.reduce_sum(uji * v, -1)
            B = B + dB
            return v_, B

    def _generate(self, v, y):
        '''
        Input:
            `y`: [n,]
            `v`: Mask. [n, J=10, V=16]
        Return:
            `x`: Image [n, h, w, c]
        '''
        J, _, _, V = self._get_shape_JDUV()

        Y = tf.one_hot(y, J) # [n, J]
        Y = tf.expand_dims(Y, -1)  # [n, J, 1]

        x = v * Y
        x = tf.reshape(x, [-1, J * V])

        net = self.arch['generator']
        for o in net['output']:
            x = tf.layers.dense(x, o, tf.nn.relu)

        h, w, c = self.arch['hwc']
        x = tf.layers.dense(x, h * w * c, tf.nn.tanh)
        return tf.reshape(x, [-1, h, w, c])


    def _get_loss_parameter(self):
        return self.arch['']

    def loss(self, x, y):
        '''
        `x`: [b, 28, 28, 1]
        `y`: label [b]
        '''
        v = self._C(x)  # [n, J=10, V=16]
        xh = self._G(v, y)  # [n, h, w, c]

        J, _, _, _ = self._get_shape_JDUV()
        with tf.name_scope('Loss'):
            tf.summary.image('x', x, 4)
            tf.summary.image('xh', xh, 4)
            tf.summary.image('V', tf.expand_dims(v, -1), 4)

            hparam = self.arch['loss']

            l_reconst = hparam['reconst weight'] * \
                tf.reduce_mean(tf.reduce_sum(tf.square(x - xh), [1, 2, 3]))
            tf.summary.scalar('l_reconst', l_reconst)

            v_norm = tf.norm(v, axis=-1)  # [n, J=10]
            tf.summary.histogram('v_norm', v_norm)

            Y = tf.one_hot(y, J)  # [n, J=10]
            loss = Y * tf.square(tf.maximum(0., hparam['m+'] - v_norm)) \
                + hparam['lambda'] * (1. - Y) * \
                tf.square(tf.maximum(0., v_norm - hparam['m-']))
            loss = tf.reduce_mean(tf.reduce_sum(loss, -1))

            tf.summary.scalar('loss', loss)
            loss += l_reconst

            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(y, tf.argmax(v_norm, 1)),
                    tf.float32
                ))
            return {'L': loss, 'acc': acc, 'reconst': l_reconst}

    def inspect(self, x):
        with tf.name_scope('Inpector'):
            J, _, _, V = self._get_shape_JDUV()
            R = self.arch['valid']['spacing']
            m = self.arch['valid']['magnitude']
            h, w, c = self.arch['hwc']

            v = self._C(x)  # 10, J=10, V=16, generated from exemplar images
            v_eps, y_eps = make_linear_perturbation(J, V, R, m)
            for i in range(10):
                vi = tf.expand_dims(v[i], 0)  # [1, J=10, V=16]
                vi = vi + v_eps # [V*21, 10, V]

                xh = self._G(vi, i * y_eps)

                xh = tf.reshape(xh, [1, R, V, h, w, c])
                xh = tf.transpose(xh, [0, 1, 3, 2, 4, 5])
                xh = tf.reshape(xh, [1, R * h, V * w, c])

                tf.summary.image('xh{}'.format(i), xh)


    def train(self, loss, loss_t):
        global_step = tf.Variable(0)
        dirs = validate_log_dirs(args)
        dirs.update({'logdir': dirs['logdir'] + args.msg})

        hparam = self.arch['training']
        maxIter = hparam['num_epoch'] * 60000 // hparam['batch_size']
        optimizer = tf.train.AdamOptimizer()
        opt = optimizer.minimize(loss['L'], global_step=global_step)

        sv = tf.train.Supervisor(
            logdir=dirs['logdir'],
            # save_summaries_secs=120,
            global_step=global_step,
        )
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
        with sv.managed_session(config=sess_config) as sess:
            for it in range(maxIter):
                if it % hparam['update_freq'] == 0:
                    a = list()
                    for _ in range(100):
                        a_ = sess.run(loss_t['acc'])
                        a.append(a_)
                    a_t = np.mean(a)

                    l, a, _ = sess.run([loss['L'], loss['acc'], opt])
                    print(
                        '\rIter {}/{}: loss = {:.4e}, acc={:.2f}%; test acc={:.2f}'.format(
                            it, maxIter, l, a_t * 100., a * 100.),
                        end=''
                    )
                else:
                    sess.run(opt)
            print()

# def unittest():
#     x = tf.placeholder(tf.float32, [None, 28, 28, 1])
#     y = tf.placeholder(tf.int32, [None,])
#     CapNet = CapsuleNet(arch=None)
#     yh = CapNet._C(x)
#     print(yh)
#     loss = CapNet.loss(x, y)
#     print(loss)
#     # CapNet.train()
#     # set_trace()

class CapsuleMultiMNIST(CapsuleNet):
    def _pick(self, v, y):
        ''' v: [b, J, V]
        `y`: [b,]
        '''
        i = tf.expand_dims(tf.range(tf.shape(v)[0]), -1)
        y = tf.expand_dims(tf.cast(y, tf.int32), -1)
        return tf.gather_nd(v, tf.concat([i, y], -1))

    def loss(self, x, y, xi, xj):
        v = self._C(x)  # [n, J=10, V=16]
        tf.summary.image('V', tf.expand_dims(v, -1), 4)

        xh_ = self._G(
            tf.concat([v, v], 0),
            tf.concat([y[:, 0], y[:, 1]], 0)
        )

        # TODO: an exp on rescaling the DigitCaps
        with tf.name_scope('Experiment'):
            v_norm = tf.norm(v + 1e-6, axis=-1, keep_dims=True)
            v_ = v / v_norm
            xh_exp = self._G(
                tf.concat([v_, v_], 0),
                tf.concat([y[:, 0], y[:, 1]], 0)
            )
            xhie, xhje = tf.split(xh_exp, 2)
            tf.summary.image('xei', xhie, 4)
            tf.summary.image('xej', xhje, 4)
        

        with tf.name_scope('Loss'):
            xhi, xhj = tf.split(xh_, 2)
            xhx = tf.concat([xhi, xhj, - tf.ones_like(xhi)], -1)  # pad by -1 (float) = 0 (uint8)
            with tf.name_scope('Images'):
                tf.summary.image('x', x, 4)
                tf.summary.image('xhx', xhx, 4)
                tf.summary.image('xhi', xhi, 4)
                tf.summary.image('xhj', xhj, 4)
                tf.summary.image('xi', xi, 4)
                tf.summary.image('xj', xj, 4)

            # xh = tf.concat([tf.expand_dims(xhi, -1), tf.expand_dims(xhj, -1)], -1)
            # xh = tf.reduce_max(xh, -1)

            hparam = self.arch['loss']
            r = hparam['reconst weight']

            # l_reconst = .5 * hparam['reconst weight'] * tf.reduce_mean(
            #     tf.reduce_sum(tf.square(xi - xhi) + tf.square(xj - xhj), [1, 2, 3]) #+
            #     # tf.reduce_sum(tf.square(xj - xhj), [1, 2, 3])
            # )

            # TODO: Should I treat it individually? (the other digit as noise?)
            x_ = tf.concat([xi, xj], 0)
            l_reconst = r * tf.reduce_mean(tf.reduce_sum(tf.square(xh_ - x_), [1, 2, 3]))
            tf.summary.scalar('l_reconst', l_reconst)

            v_norm = tf.norm(v, axis=-1)  # [n, J=10]
            tf.summary.histogram('v_norm', v_norm)

            J, _, _, _ = self._get_shape_JDUV()
            Yi = tf.one_hot(y[:, 0], J)
            Yj = tf.one_hot(y[:, 1], J)  # [n, J]
            Y = Yi + Yj

            # tf.summary.histogram('v_norm_ans', v_norm * Y)
            # tf.summary.histogram('v_norm_not', v_norm * (1. - Y))
            tf.summary.histogram('v_norm_i', self._pick(v_norm, y[:, 0]))
            tf.summary.histogram('v_norm_j', self._pick(v_norm, y[:, 1]))
            tf.summary.histogram(
                'v_norm_ans', 
                tf.concat(
                    [self._pick(v_norm, y[:, 0]), self._pick(v_norm, y[:, 1])],
                    0
                )
            )

            l, m, M = hparam['lambda'], hparam['m-'], hparam['m+']
            loss = Y * tf.square(tf.maximum(0., M - v_norm)) \
                + l * (1. - Y) * tf.square(tf.maximum(0., v_norm - m))
            loss = tf.reduce_mean(tf.reduce_sum(loss, -1))
            tf.summary.scalar('loss', loss)

            loss = loss + l_reconst

            # loss = Y * tf.square(tf.maximum(0., hparam['m+'] - v_norm)) \
            #     + hparam['lambda'] * (1. - Y) * \
            #     tf.square(tf.maximum(0., v_norm - hparam['m-']))

            # acc = tf.reduce_mean(
            #     tf.cast(
            #         tf.equal(y, tf.argmax(v_norm, 1)),
            #         tf.float32
            #     ))

            # TODO: HOW TO CALCULATE THE "ACCURACY" in MultiMNIST?
            acc = tf.cast(tf.nn.in_top_k(v_norm, y[:, 0], 2), tf.float32) \
                + tf.cast(tf.nn.in_top_k(v_norm, y[:, 1], 2), tf.float32)
            acc = tf.reduce_mean(acc) / 2.
            tf.summary.scalar('UR', acc)
            acc = tf.cast(tf.nn.in_top_k(v_norm, y[:, 0], 2), tf.float32) \
                * tf.cast(tf.nn.in_top_k(v_norm, y[:, 1], 2), tf.float32)
            acc = tf.reduce_mean(acc)
            tf.summary.scalar('EM', acc)
            return {'L': loss, 'acc': acc, 'reconst': l_reconst}



    def train(self, loss, loss_t):
        global_step = tf.Variable(0, name='global_step')
        dirs = validate_log_dirs(args)
        # dirs.update({'logdir': dirs['logdir'] + args.msg})

        hparam = self.arch['training']
        maxIter = hparam['num_epoch'] * 60000000 // hparam['batch_size']  # TODO
        learning_rate = tf.train.exponential_decay(
            1e-3, global_step,
            5000, 0.8, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        opt = optimizer.minimize(loss['L'], global_step=global_step)

        sv = tf.train.Supervisor(
            logdir=dirs['logdir'],
            # save_summaries_secs=120,
            global_step=global_step,
        )
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
        with sv.managed_session(config=sess_config) as sess:
            for it in range(maxIter):
                if it % hparam['update_freq'] == 0:
                    # a = list()
                    # for _ in range(100):
                    #     a_ = sess.run(loss_t['acc'])
                    #     a.append(a_)
                    # a_t = np.mean(a)

                    l, a, _ = sess.run([loss['L'], loss['acc'], opt])
                    print(
                        '\rIter {}/{}: loss = {:.4e}, acc={:.2f}%;'.format(
                            it, maxIter, l, a * 100.), #, a_t * 100.),
                        end=''
                    )
                else:
                    sess.run(opt)
            print()


def main():
    with open(args.arch) as fp:
        arch = json.load(fp)
    data = MultiMNISTIndexReader(
        train_index='MultiMNIST_index_train.npf',
        batch_size=arch['training']['batch_size'],
        data_format='channels_last',
        capacity=2**12, min_after_dequeue=2**11
    )
    net = CapsuleMultiMNIST(arch=arch)
    loss = net.loss(data.x, data.y, data.xi, data.xj)
    net.inspect(data.example)
    # loss_t = net.loss(data.x_t, data.y_t)
    net.train(loss, loss_t=None)

if __name__ == '__main__':
    main()
