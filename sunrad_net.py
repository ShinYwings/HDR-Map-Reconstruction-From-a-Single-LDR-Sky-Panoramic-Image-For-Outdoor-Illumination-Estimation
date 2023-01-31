import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
import tensorflow_addons as tfa
import ops
import numpy as np

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

class sunRadNet(Model):
    def __init__(self, epsilon=1e-5, pi=np.math.pi):
        super(sunRadNet, self).__init__()
        
        self.epsilon = epsilon
        self.deltafunc_const = tf.sqrt(pi)

        self.d1 = downsampling(64, 4, strides=2, apply_norm=False) # n,16,64,64
        self.d2 = downsampling(128, 4, strides=2, apply_norm=True) # n,8,32,128
        self.d3 = downsampling(256, 4, strides=2, apply_norm=True) # n,4,16,256
        self.d4 = downsampling(512, 4, strides=1, apply_norm=True) # n,4,16,512
        
        self.flat = tf.keras.layers.Flatten()
        self.gamma  = tf.keras.layers.Dense(1)
        self.beta  = tf.keras.layers.Dense(1)
        
    def call(self, x, actv_map, training="training"):
        
        d1 = self.d1(actv_map, training)
        d2 = self.d2(d1, training)
        d3 = self.d3(d2, training)
        d4 = self.d4(d3, training)

        flat = self.flat(d4)
        gamma = self.gamma(flat)
        beta = self.beta(flat)

        gamma_in = tf.nn.sigmoid(gamma)
        gamma_in = tf.reshape(gamma_in, [-1, 1, 1, 1])
        beta_in = tf.nn.sigmoid(beta)
        beta_in = tf.reshape(beta_in, [-1, 1, 1, 1])
        
        # Direc delta function (0 ~ infty)
        x = -tf.pow(tf.subtract(1., x), 2.)
        x = tf.divide(x, (beta_in + self.epsilon))
        x = tf.math.exp(x)
        x = tf.multiply(x, gamma_in)
        _const = tf.multiply(beta_in, self.deltafunc_const)
        x = tf.divide(x, (_const + self.epsilon))
        x = tf.where(x > 30000., 30000. , x)
        
        return x, gamma_in, beta_in