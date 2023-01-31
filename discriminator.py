import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import distortion_aware_ops as distortion_aware_ops
import tensorflow_addons as tfa

class downsampling(Layer):

    def __init__(self, filters, kernel_size, strides=2, apply_norm=True):
        super(downsampling, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", 
                                                kernel_initializer=tf.random_normal_initializer(0., 0.02),
                                                use_bias=False)
        # CycleGAN way
        # self.norm = tfa.layers.InstanceNormalization()
        self.norm = tf.keras.layers.BatchNormalization()
        self.actv = tf.keras.layers.LeakyReLU()
        self.apply_norm = apply_norm

    def call(self, x, training="training"):
        x = self.conv(x)
        if self.apply_norm:
            # x = self.norm(x)
            x = self.norm(x, training)
        x = self.actv(x)
        
        return x

class model(Model):
    def __init__(self, im_height=32, im_width= 128, da_kernel_size=3, dilation_rate=1):
        super(model, self).__init__()

        self.d1 = downsampling(64, 4, strides=2, apply_norm=False) # n,16,64,64
        self.d2 = downsampling(128, 4, strides=2, apply_norm=True) # n,8,32,128
        self.d3 = downsampling(256, 4, strides=2, apply_norm=True) # n,4,16,256
        self.d4 = downsampling(512, 4, strides=1, apply_norm=True) # n,4,16,512
        
        self.out = tf.keras.layers.Conv2D(1,4,strides=1,
                                kernel_initializer = tf.random_normal_initializer(0., 0.02))
    
    def call(self, x, training="training"):
        x = tf.concat(x, axis=-1)
        x = self.d1(x, training)
        x = self.d2(x, training)
        x = self.d3(x, training)
        x = self.d4(x, training)

        x = self.out(x)
        # if use lsgan, do not use sigmoid
        return x