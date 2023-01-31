import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import inspect
import os

class conv2d(Layer):
    def __init__(self,
                data_dict = 'data_dict',
                layer_name = "layer_name",
                padding="SAME"):

        super(conv2d, self).__init__()
        self.strides = [1,1,1,1]
        self.padding = padding

        # Load predefined weight value
        self.data_dict = data_dict

        self.layer_name = layer_name

    def build(self, input_shape):

        self.w = self.get_conv_weight(self.layer_name)
        
        self.biases = self.get_bias(self.layer_name)

        super(conv2d, self).build(input_shape=input_shape)

    def call(self, input):
        
        x = tf.nn.bias_add(tf.nn.conv2d(input, self.w, strides=self.strides, padding=self.padding), self.biases)

        return tf.nn.relu(x)
    
    def get_conv_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

# class fc2d(Layer):
#     def __init__(self, data_dict='data_dict'):
#         super(fc2d, self).__init__()
#         self.data_dict = data_dict

#     def build(self,input_shape):

#         self.w = self.get_fc_weight()
       
#         self.biases = self.get_bias()

#         super(fc2d, self).build(input_shape)

#     def call(self, input):
        
#         shape = input.get_shape().as_list()
#         dim = 1
#         for d in shape[1:]:
#             dim *= d
#         fc = tf.reshape(input, [-1, dim])

#         fc = tf.matmul(fc, self.w)
#         fc = tf.nn.bias_add(fc, self.biases)

#         return fc

#     def get_fc_weight(self):
#         return tf.constant(self.data_dict[self.name][0], name="weights")

#     def get_bias(self):
#         return tf.constant(self.data_dict[self.name][1], name="biases")

class maxpool2d(Layer):
    def __init__(self, kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", layer_name="layer_name"):
        
        super(maxpool2d, self).__init__()
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.layer_name = layer_name
    
    def call(self, x):
        return tf.nn.max_pool(x, ksize= self.kernel_size, strides=self.strides, padding= self.padding, name=self.layer_name)

class Vgg16(Model):
    def __init__(self, vgg16_npy_path=None, VGG_MEAN = [103.939, 116.779, 123.68]):
        super(Vgg16, self).__init__()

        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)

        data_dict = np.load(vgg16_npy_path, encoding='latin1', allow_pickle=True).item()
        print("npy file loaded")

        self.VGG_MEAN = VGG_MEAN

        self.conv1_1 = conv2d(data_dict = data_dict, layer_name="conv1_1")
        self.conv1_2 = conv2d(data_dict = data_dict, layer_name="conv1_2")
        self.pool1   = maxpool2d(layer_name='pool1')

        self.conv2_1 = conv2d(data_dict = data_dict, layer_name="conv2_1")
        self.conv2_2 = conv2d(data_dict = data_dict, layer_name="conv2_2")
        self.pool2   = maxpool2d(layer_name='pool2')

        self.conv3_1 = conv2d(data_dict = data_dict, layer_name="conv3_1")
        self.conv3_2 = conv2d(data_dict = data_dict, layer_name="conv3_2")
        self.conv3_3 = conv2d(data_dict = data_dict, layer_name="conv3_3")
        self.pool3   = maxpool2d(layer_name='pool3')

        # self.conv4_1 = conv2d(data_dict, name="conv4_1")
        # self.conv4_2 = conv2d(data_dict, name="conv4_2")
        # self.conv4_3 = conv2d(data_dict, name="conv4_3")
        # self.pool4   = maxpool2d(name='pool4')

        # self.conv5_1 = conv2d(data_dict, name="conv5_1")
        # self.conv5_2 = conv2d(data_dict, name="conv5_2")
        # self.conv5_3 = conv2d(data_dict, name="conv5_3")
        # self.pool5   = maxpool2d(name='pool5')

    def call(self, bgr, training="training"):
        
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        bgr_scaled = tf.scalar_mul(255.0, bgr)

        # Convert RGB to BGR
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr_scaled)
        bgr = tf.concat(axis=3, values=[
            blue - self.VGG_MEAN[0],
            green - self.VGG_MEAN[1],
            red - self.VGG_MEAN[2],
        ])

        conv1_1 = self.conv1_1(bgr)
        conv1_2 = self.conv1_2(conv1_1)
        pool1   = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2   = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3   = self.pool3(conv3_3)

        # conv4_1 = self.conv4_1(pool3)
        # conv4_2 = self.conv4_2(conv4_1)
        # conv4_3 = self.conv4_3(conv4_2)
        # pool4   = self.pool4(conv4_3)

        # conv5_1 = self.conv5_1(pool4)
        # conv5_2 = self.conv5_2(conv5_1)
        # conv5_3 = self.conv5_3(conv5_2)
        # pool5   = self.pool5(conv5_3)

        return pool1, pool2, pool3