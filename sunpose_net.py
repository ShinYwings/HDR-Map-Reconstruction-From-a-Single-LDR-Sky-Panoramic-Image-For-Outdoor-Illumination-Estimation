import tensorflow as tf
from tensorflow.keras import Model
import ops
import distortion_aware_ops as distortion_aware_ops
import tensorflow_addons as tfa

class sunposeLayer(Model):
    def __init__(self, filter_out, k_h=3, k_w=3, strides=1, dilation_rate=1):
        super(sunposeLayer, self).__init__()
        self.conv1 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides) 
        # self.conv1 = distortion_aware_ops.conv2d(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)
        self.norm1 = tfa.layers.InstanceNormalization()
        self.actv1 = ops.relu()
        
        self.conv2 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides) 
        # self.conv2 = distortion_aware_ops.conv2d(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)
        self.norm2 = tfa.layers.InstanceNormalization()
        self.actv2 = ops.relu()

    def call(self, x, training="training"):

        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1)
        actv1  = self.actv1(norm1)

        conv2 = self.conv2(actv1)
        norm2 = self.norm2(conv2)
        actv2  = self.actv2(norm2)

        return actv2

class model(Model):
    def __init__(self, im_height=32, im_width= 128, da_kernel_size=3, dilation_rate=1):
        super(model, self).__init__()

        self.fc_dim = int(im_height*im_width)

        # Sun position encoder
        self.sunlayer1 = sunposeLayer(32, k_h=7, k_w=7)
        self.pool1_s  = ops.maxpool2d(kernel_size=2)

        self.sunlayer2 = sunposeLayer(64, k_h=3, k_w=3)
        self.pool2_s  = ops.maxpool2d(kernel_size=2)

        self.sunlayer3 = sunposeLayer(128, k_h=3, k_w=3)
        self.pool3_s  = ops.maxpool2d(kernel_size=2)

        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(self.fc_dim)
        self.actv1_s  = ops.relu()
        self.fc2 = tf.keras.layers.Dense(self.fc_dim)
        self.actv2_s  = ops.relu()

    def sunposeEstimation(self, x, training="training"):
        sunlayer1 = self.sunlayer1(x, training)
        pool1_s = self.pool1_s(sunlayer1)

        sunlayer2 = self.sunlayer2(pool1_s, training)
        pool2_s = self.pool2_s(sunlayer2)

        sunlayer3 = self.sunlayer3(pool2_s, training)
        pool3_s = self.pool3_s(sunlayer3)

        flat = self.flat(pool3_s)
        fc1 = self.fc1(flat)
        actv1_s = self.actv1_s(fc1)
        fc2 = self.fc2(actv1_s)
        actv2_s = self.actv2_s(fc2)

        sm = tf.nn.softmax(actv2_s)
        
        return sm, [sunlayer1, sunlayer2, sunlayer3]