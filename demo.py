import tensorflow as tf
from setting import *
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import matplotlib.image as img
import h5py
import os
import numpy as np
import cv2
import tool_box.blur as blur
from scipy.io import savemat

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('img_size', SIZE_INPUT, 'size of the image')
tf.app.flags.DEFINE_integer('kernel_size', SIZE_KERNEL, 'size of the kernel')
tf.app.flags.DEFINE_integer('learning_rate', LEARNING_RATE, 'size of the kernel')
tf.app.flags.DEFINE_integer('num_h5_file', NUM_FILES, 'number of training h5 files.')
tf.app.flags.DEFINE_integer('num_batch', NUM_BATCH, 'number of patches in each h5 file.')
tf.app.flags.DEFINE_integer('batch_size', SIZE_BATCH, 'Batch size.')


def guided_filter(data):
    r = 15
    eps = 1.0
    shape = data.shape
    batch_q = np.zeros((shape[0], shape[1], shape[2], NUM_CHANNEL))
    for i in range(shape[0]):
        for j in range(NUM_CHANNEL):
            I = data[i, :, :, j]
            p = data[i, :, :, j]
            ones_array = np.ones([shape[1], shape[2]])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            q = mean_a * I + mean_b
            batch_q[i, :, :, j] = q
    return batch_q


class Network_for_angle:
    def inference(self, img_rainy, regularizer=None):
        # layzer 1   64*64
        with tf.variable_scope('conv_1'):
            kernel = tf.get_variable('kernel', [5, 5, NUM_CHANNEL, 12],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [12], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(img_rainy, kernel, [1, 1, 1, 1], padding='VALID')
            conv_output = tf.nn.relu(tf.nn.bias_add(conv1, biases))
        # 60*60
        with tf.variable_scope('pool_2'):
            kernel = tf.get_variable('kernel', [3, 3, 12, 12], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [12], initializer=tf.constant_initializer(0.0))
            pool_2 = tf.nn.conv2d(conv_output, kernel, strides=[1, 2, 2, 1], padding='SAME')
            pool_2 = tf.nn.relu(tf.nn.bias_add(pool_2, biases))
        # 30*30
        with tf.variable_scope('conv_3'):
            kernel = tf.get_variable('weight', [5, 5, 12, 24], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [24], initializer=tf.constant_initializer(0.0))
            conv3 = tf.nn.conv2d(pool_2, kernel, [1, 1, 1, 1], padding='VALID')
            conv_output = tf.nn.relu(tf.nn.bias_add(conv3, biases))
        # 26*26
        with tf.variable_scope('pool_4'):
            kernel = tf.get_variable('kernel', [3, 3, 24, 24], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [24], initializer=tf.constant_initializer(0.0))
            pool_4 = tf.nn.conv2d(conv_output, kernel, strides=[1, 2, 2, 1], padding='SAME')
            pool_4 = tf.nn.relu(tf.nn.bias_add(pool_4, biases))
        # 13*13*24
        poolshape = pool_4.get_shape().as_list()
        nodes = poolshape[1] * poolshape[2] * poolshape[3]
        reshape = tf.reshape(pool_4, [tf.shape(img_rainy)[0], nodes])
        # 4056
        with tf.variable_scope('fc_5'):
            weight = tf.get_variable('weight', [nodes, 100], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [100], initializer=tf.constant_initializer(0.1))
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(weight))
            fc = tf.nn.relu(tf.matmul(reshape, weight) + biases)
        with tf.variable_scope('fc_6'):
            weight = tf.get_variable('weight', [100, 2], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [2], initializer=tf.constant_initializer(0.1))
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(weight))
            fc = tf.nn.relu(tf.matmul(fc, weight) + biases)
        return fc

    def readdata(self, name):
        with h5py.File(name, 'r') as hf:
            data_rainy = np.array(hf.get('rainy'))
            data_angle = np.array(hf.get('angle'))
            data_lenth = np.array(hf.get('r_lenth'))
            return data_rainy, np.hstack((data_angle, data_lenth))

    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.img_rainy = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, NUM_CHANNEL])
            self.angle = tf.placeholder(tf.float32, shape=[None, 2])
            self.output = self.inference(self.img_rainy)
            self.loss = tf.reduce_mean(tf.square(self.output - self.angle))  # + tf.add_n(tf.get_collection('losses'))
            self.lr = tf.placeholder(tf.float32, shape=[])
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.saver = tf.train.Saver(max_to_keep=5)
            self.config = tf.ConfigProto()
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.5  # GPU setting
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
        self.load()

    def load(self):
        with self.graph.as_default():
            if tf.train.get_checkpoint_state('./model/model_angle/'):
                ckpt = tf.train.latest_checkpoint('./model/model_angle/')
                self.saver.restore(self.sess, ckpt)
                print("Loading model")
            else:
                self.sess.run(tf.global_variables_initializer())
                print("No model to load")

    def train(self):
        with self.graph.as_default():
            for j in range(EPOCH):
                lr_ = FLAGS.learning_rate * (0.2 ** j)
                for h5_num in range(FLAGS.num_h5_file - 1):
                    train_data_name = 'train' + str(h5_num + 1) + '.h5'
                    data_rainy, Angle = self.readdata(DATA_PATH + train_data_name)
                    Angle[:, 0] = (Angle[:, 0] - 45) / 90
                    Angle[:, 1] = (Angle[:, 1] - 15) / 15
                    data_rainy = data_rainy - guided_filter(data_rainy)
                    shape = data_rainy.shape
                    data_size = int(shape[0] / FLAGS.batch_size)
                    for batch_num in range(data_size):
                        rand_index = np.arange(int(batch_num * FLAGS.batch_size),
                                               int((batch_num + 1) * FLAGS.batch_size))
                        batch_rainy = data_rainy[rand_index, :, :, :]
                        batch_angle = Angle[rand_index, :]
                        _, lossvalue = self.sess.run([self.train_op, self.loss], feed_dict={
                            self.img_rainy: batch_rainy,
                            self.angle: batch_angle,
                            self.lr: lr_
                        })

                        print('training %d epoch, %d h5file %d batch, error is %.4f' % (
                            j + 1, h5_num + 1, batch_num + 1, lossvalue))

                model_name = 'model-epoch'
                save_path_full = os.path.join('./model/model_angle/', model_name)
                self.saver.save(self.sess, save_path_full, global_step=j + 1)

    def test(self, img_rainy):
        with self.graph.as_default():
            img_rainy = img_rainy - guided_filter(img_rainy)
            shape = img_rainy.shape
            output = np.zeros((shape[0], 2))
            for i in range(shape[0] // 100):
                output[i * 100:(i + 1) * 100, :] = self.sess.run(self.output, feed_dict={
                    self.img_rainy: img_rainy[i * 100:(i + 1) * 100, :, :, :]
                })
            if shape[0] // 100 * 100 != shape[0]:
                output[shape[0] // 100 * 100:shape[0], :] = self.sess.run(self.output, feed_dict={
                    self.img_rainy: img_rainy[shape[0] // 100 * 100:shape[0], :, :, :]
                })
            return output

    def __del__(self):
        self.sess.close()


class Network_for_derain:

    def readfile(self, file):
        with h5py.File(file, 'r') as hf:
            V = hf.get('V')
            t = hf.get('t')
            mean = hf.get('mean')
            std = hf.get('std')
            return np.array(V), np.array(t), np.array(mean), np.array(std)

    def _get_variable(self,
                      name,
                      shape,
                      initializer,
                      weight_decay=0.0,
                      dtype='float',
                      trainable=True):

        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               trainable=trainable)

    def bn(self, x, c):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))

        beta = self._get_variable('beta',
                                  params_shape,
                                  initializer=tf.zeros_initializer())
        gamma = self._get_variable('gamma',
                                   params_shape,
                                   initializer=tf.ones_initializer())

        moving_mean = self._get_variable('moving_mean',
                                         params_shape,
                                         initializer=tf.zeros_initializer(),
                                         trainable=False)
        moving_variance = self._get_variable('moving_variance',
                                             params_shape,
                                             initializer=tf.ones_initializer(),
                                             trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, 0.9997)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, 0.9997)
        tf.add_to_collection(UPDATE_OPS, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            c, lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)

        return x

    def create_kernel(self, name, shape, initializer=tf.contrib.layers.xavier_initializer()):
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-8)

        new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                        regularizer=regularizer, trainable=True)
        return new_variables

    def inference_kernel(self, images, detail, Kernel, is_training):
        c = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
        #  layer 1
        with tf.variable_scope('conv_1'):
            kernel = self.create_kernel(name='weights_1', shape=[3, 3, NUM_CHANNEL, 24])
            biases = tf.Variable(tf.constant(0.1, shape=[24], dtype=tf.float32), trainable=True, name='biases_1')

            conv1 = tf.nn.conv2d(detail, kernel, [1, 1, 1, 1], padding='SAME')
            bias1 = tf.nn.bias_add(conv1, biases)

            kernel_k = self.create_kernel(name='weights_k', shape=[1, 1, self.t, 12])
            biases_k = tf.Variable(tf.constant(0.1, shape=[12], dtype=tf.float32), trainable=True, name='biases_k')
            conv_k = tf.nn.conv2d(Kernel, kernel_k, [1, 1, 1, 1], padding='SAME')
            bias_k = tf.nn.bias_add(conv_k, biases_k)

            tmp1 = tf.concat([bias_k, bias1], 3)
            bias1 = self.bn(tmp1, c)

            conv_shortcut = tf.nn.relu(bias1)
        #  layers 2 to 25
        for i in range(12):
            with tf.variable_scope('conv_%s' % (i * 2 + 2)):
                kernel = self.create_kernel(name=('weights_%s' % (i * 2 + 2)), shape=[3, 3, 36, 36])
                biases = tf.Variable(tf.constant(0.1, shape=[36], dtype=tf.float32), trainable=True,
                                     name=('biases_%s' % (i * 2 + 2)))

                conv_tmp1 = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
                bias_tmp1 = tf.nn.bias_add(conv_tmp1, biases)

                bias_tmp1 = self.bn(bias_tmp1, c)

                out_tmp1 = tf.nn.relu(bias_tmp1)

            with tf.variable_scope('conv_%s' % (i * 2 + 3)):
                kernel = self.create_kernel(name=('weights_%s' % (i * 2 + 3)), shape=[3, 3, 36, 36])
                biases = tf.Variable(tf.constant(0.1, shape=[36], dtype=tf.float32), trainable=True,
                                     name=('biases_%s' % (i * 2 + 3)))

                conv_tmp2 = tf.nn.conv2d(out_tmp1, kernel, [1, 1, 1, 1], padding='SAME')
                bias_tmp2 = tf.nn.bias_add(conv_tmp2, biases)

                bias_tmp2 = self.bn(bias_tmp2, c)

                bias_tmp2 = tf.nn.relu(bias_tmp2)
                conv_shortcut = tf.add(conv_shortcut, bias_tmp2)

        # layer 26
        with tf.variable_scope('conv_26'):
            kernel = self.create_kernel(name='weights_26', shape=[3, 3, 36, NUM_CHANNEL])
            biases = tf.Variable(tf.constant(0.1, shape=[NUM_CHANNEL], dtype=tf.float32), trainable=True,
                                 name='biases_26')

            conv_final = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
            bias_final = tf.nn.bias_add(conv_final, biases)

            neg_residual = self.bn(bias_final, c)
            final_out = tf.nn.relu(tf.add(images, neg_residual))
            # final_out = tf.add(images, bias_final)
        return final_out

    def __init__(self):
        self.graph = tf.Graph()
        self.V, self.t, self.mean, self.std = self.readfile('./data_generation/matrix/matrix.h5')
        with self.graph.as_default():
            self.img_rainy = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, NUM_CHANNEL])
            self.kernel = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, self.t])
            self.detail = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, NUM_CHANNEL])
            self.labels = tf.placeholder(tf.float32, shape=[None, FLAGS.img_size, FLAGS.img_size, NUM_CHANNEL])
            self.is_traing = tf.placeholder(tf.bool, shape=[])

            self.output = self.inference_kernel(self.img_rainy, self.detail, self.kernel,
                                                is_training=self.is_traing)
            self.loss = tf.reduce_mean(tf.square(self.output - self.labels))

            self.lr = tf.placeholder(tf.float32, shape=[])

            g_optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            batchnorm_updates = tf.get_collection(UPDATE_OPS)
            batchnorm_updates_ops = tf.group(*batchnorm_updates)

            self.train_op = tf.group(g_optim, batchnorm_updates_ops)

            self.saver = tf.train.Saver(max_to_keep=5)
            self.config = tf.ConfigProto()
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.5  # GPU setting
            self.config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=self.config)
        self.load()

    def readdata(self, name):
        with h5py.File(name, 'r') as hf:
            data_rainy = hf.get('rainy')
            data_clean = hf.get('clean')
            data_kernel = hf.get('kernel')
            return np.array(data_rainy), np.array(data_clean), np.array(data_kernel)

    def kernel_to_PCA(self, kernel):
        shape = kernel.shape
        kernel = kernel.reshape((shape[0], shape[1] * shape[2]))
        for i in range(shape[1] * shape[2]):
            kernel[:, i] = (kernel[:, i] - self.mean[i]) / self.std[i]
        kernel = np.dot(kernel, self.V)
        tmp = np.zeros(dtype=np.float32, shape=[shape[0], FLAGS.img_size, FLAGS.img_size, self.t])
        for i in range(FLAGS.img_size):
            for j in range(FLAGS.img_size):
                tmp[:, i, j, :] = kernel
        return tmp

    def load(self):
        with self.graph.as_default():
            if tf.train.get_checkpoint_state('./model/model_derain/'):
                ckpt = tf.train.latest_checkpoint('./model/model_derain/')
                self.saver.restore(self.sess, ckpt)
                print("Loading model")
            else:
                self.sess.run(tf.global_variables_initializer())
                print("No model to load")

    def train(self):
        with self.graph.as_default():
            for j in range(EPOCH):
                lr_ = FLAGS.learning_rate * (0.2 ** j)
                for h5_num in range(FLAGS.num_h5_file - 1):
                    train_data_name = 'train' + str(h5_num + 1) + '.h5'

                    data_rainy, data_clean, data_kernel = self.readdata(DATA_PATH + train_data_name)

                    data_kernel = self.kernel_to_PCA(data_kernel)
                    data_detail = data_rainy - guided_filter(data_rainy)

                    shape = data_clean.shape
                    data_size = int(shape[0] / FLAGS.batch_size)
                    for batch_num in range(data_size):
                        rand_index = np.arange(int(batch_num * FLAGS.batch_size),
                                               int((batch_num + 1) * FLAGS.batch_size))
                        batch_rainy = data_rainy[rand_index, :, :, :]
                        batch_detail = data_detail[rand_index, :, :, :]
                        batch_labels = data_clean[rand_index, :, :, :]
                        batch_kernel = data_kernel[rand_index, :, :, :]

                        _, lossvalue = self.sess.run([self.train_op, self.loss], feed_dict={
                            self.img_rainy: batch_rainy,
                            self.detail: batch_detail,
                            self.labels: batch_labels,
                            self.kernel: batch_kernel,
                            self.lr: lr_,
                            self.is_traing: True
                        })

                        print('training %d epoch, %d h5file %d batch, error is %.4f' % (
                            j + 1, h5_num + 1, batch_num + 1, lossvalue))

                model_name = 'model-epoch'
                save_path_full = os.path.join('./model/model_derain', model_name)
                self.saver.save(self.sess, save_path_full, global_step=j + 1)

    def test(self, img_rainy, kernel):
        kernel = self.kernel_to_PCA(kernel)
        # Y, Cr, Cb = rgb2ycrcb(img_rainy)
        # Y = np.expand_dims(Y, axis=3)
        detail = img_rainy - guided_filter(img_rainy)
        with self.graph.as_default():
            shape = img_rainy.shape
            output = np.zeros(shape)
            for i in range(shape[0] // 100):
                output[i * 100:(i + 1) * 100, :, :, :] = self.sess.run(self.output, feed_dict={
                    self.img_rainy: img_rainy[i * 100:(i + 1) * 100, :, :, :],
                    self.detail: detail[i * 100:(i + 1) * 100, :, :, :],
                    self.kernel: kernel[i * 100:(i + 1) * 100, :, :],
                    self.is_traing: False
                })
            if shape[0] // 100 * 100 != shape[0]:
                output[shape[0] // 100 * 100:shape[0], :, :, :] = self.sess.run(self.output, feed_dict={
                    self.img_rainy: img_rainy[shape[0] // 100 * 100:shape[0], :, :, :],
                    self.detail: detail[shape[0] // 100 * 100:shape[0], :, :, :],
                    self.kernel: kernel[shape[0] // 100 * 100:shape[0], :, :],
                    self.is_traing: False
                })
            return output


def img2patch(img_rainy, step):
    shape = img_rainy.shape
    input = np.zeros(
        (
            ((shape[0] - SIZE_INPUT) // step + 2) * ((shape[1] - SIZE_INPUT) // step + 2),
            SIZE_INPUT,
            SIZE_INPUT,
            3
        )
    )
    num = 0
    ones = np.zeros(shape)
    for i in range((shape[0] - SIZE_INPUT) // step + 1):
        for j in range((shape[1] - SIZE_INPUT) // step + 1):
            input[num, :, :, :] = img_rainy[i * step:i * step + SIZE_INPUT, j * step:j * step + SIZE_INPUT, :]
            ones[i * step:i * step + SIZE_INPUT, j * step:j * step + SIZE_INPUT, :] = ones[
                                                                                      i * step:i * step + SIZE_INPUT,
                                                                                      j * step:j * step + SIZE_INPUT,
                                                                                      :] + 1
            num = num + 1
        input[num, :, :, :] = img_rainy[i * step:i * step + SIZE_INPUT, shape[1] - SIZE_INPUT:shape[1], :]
        ones[i * step:i * step + SIZE_INPUT, shape[1] - SIZE_INPUT:shape[1], :] = ones[i * step:i * step + SIZE_INPUT,
                                                                                  shape[1] - SIZE_INPUT:shape[1], :] + 1
        num = num + 1
    for j in range((shape[1] - SIZE_INPUT) // step + 1):
        input[num, :, :, :] = img_rainy[shape[0] - SIZE_INPUT:shape[0], j * step:j * step + SIZE_INPUT, :]
        ones[shape[0] - SIZE_INPUT:shape[0], j * step:j * step + SIZE_INPUT, :] = ones[shape[0] - SIZE_INPUT:shape[0],
                                                                                  j * step:j * step + SIZE_INPUT, :] + 1
        num = num + 1
    input[num, :, :, :] = img_rainy[shape[0] - SIZE_INPUT:shape[0], shape[1] - SIZE_INPUT:shape[1], :]
    ones[shape[0] - SIZE_INPUT:shape[0], shape[1] - SIZE_INPUT: shape[1], :] = ones[shape[0] - SIZE_INPUT:shape[0],
                                                                               shape[1] - SIZE_INPUT: shape[1], :] + 1
    num = num + 1
    return input, ones


def patch2img(output, ones, step):
    shape = ones.shape
    derained = np.zeros(shape)
    num = 0
    for i in range((shape[0] - SIZE_INPUT) // step + 1):
        for j in range((shape[1] - SIZE_INPUT) // step + 1):
            derained[i * step:i * step + SIZE_INPUT, j * step:j * step + SIZE_INPUT, :] = derained[
                                                                                          i * step:i * step + SIZE_INPUT,
                                                                                          j * step:j * step + SIZE_INPUT,
                                                                                          :] + output[num, :, :, :]
            num = num + 1
        derained[i * step:i * step + SIZE_INPUT, shape[1] - SIZE_INPUT:shape[1], :] = derained[
                                                                                      i * step:i * step + SIZE_INPUT,
                                                                                      shape[1] - SIZE_INPUT:shape[1],
                                                                                      :] + output[num, :, :, :]
        num = num + 1
    for j in range((shape[1] - SIZE_INPUT) // step + 1):
        derained[shape[0] - SIZE_INPUT:shape[0], j * step:j * step + SIZE_INPUT, :] = derained[
                                                                                      shape[0] - SIZE_INPUT:shape[0],
                                                                                      j * step:j * step + SIZE_INPUT,
                                                                                      :] + output[num, :, :, :]
        num = num + 1
    derained[shape[0] - SIZE_INPUT:shape[0], shape[1] - SIZE_INPUT:shape[1], :] = derained[
                                                                                  shape[0] - SIZE_INPUT:shape[0],
                                                                                  shape[1] - SIZE_INPUT:shape[1],
                                                                                  :] + output[num, :, :, :]
    num = num + 1
    derained = derained / ones
    return derained


def getkernel(angle):
    shape = angle.shape
    Kernel = np.zeros((shape[0], SIZE_KERNEL, SIZE_KERNEL))
    angle[:, 0] = angle[:, 0] * 90 + 45
    angle[:, 1] = angle[:, 1] * 15 + 15
    for i in range(shape[0]):
        r_lenth = angle[i, 1]
        ang = angle[i, 0]
        kernel = blur.motionblur(r_lenth, ang)
        tmp = np.zeros((SIZE_KERNEL, SIZE_KERNEL))
        row, col = kernel.shape
        tmp[(SIZE_KERNEL - row) // 2: (SIZE_KERNEL - row) // 2 + row,
        (SIZE_KERNEL - col) // 2: (SIZE_KERNEL - col) // 2 + col] = kernel
        Kernel[i, :, :] = tmp
    return Kernel


if __name__ == '__main__':
    angle = Network_for_angle()
    angle.train()
    derain = Network_for_derain()
    derain.train()
    for i in range(1, 13):
        file = "./img/%d.png" % (i)
        img_rainy = img.imread(file)
        if img_rainy.dtype == np.uint8:
            img_rainy = (1.0 / 255.0) * np.float32(img_rainy)
        img_rainy = img_rainy[:, :, 0:3]

        input, ones = img2patch(img_rainy, 16)
        Angle = angle.test(input)

        kernel = getkernel(Angle)

        final_output = derain.test(input, kernel)
        final_output[np.where(final_output < 0.)] = 0.
        final_output[np.where(final_output > 1.)] = 1.

        derained = patch2img(final_output, ones, 16)
        img.imsave('./output/%d.png' % i, derained, dpi=1)
        savemat('./output/%d.mat' % i, {'derained':derained})
