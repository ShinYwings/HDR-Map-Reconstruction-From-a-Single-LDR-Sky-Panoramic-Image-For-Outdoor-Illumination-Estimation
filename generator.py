import tensorflow as tf
from tensorflow.keras import Model
import ops
import distortion_aware_ops as distortion_aware_ops
import tensorflow_addons as tfa
from sunrad_net import sunRadNet
import tf_utils

class resBlock(Model):

    def __init__(self, filter_in, filter_out, k_h=3, k_w=3, strides=1, dilation_rate=1):
        super(resBlock, self).__init__()
        self.conv1 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides) 
        # self.conv1 = distortion_aware_ops.conv2d(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)
        self.norm1 = tfa.layers.InstanceNormalization()
        
        self.conv2 = ops.conv2d(output_channels=filter_out, k_h=k_h, k_w=k_w, strides=strides) 
        # self.conv2 = distortion_aware_ops.conv2d(filter_out, kernel_size=k_h, strides=strides, dilation_rate=dilation_rate)
        self.norm2 = tfa.layers.InstanceNormalization()
        
        if filter_in == filter_out:
            self.identity = lambda x : x
        else:
            self.identity = ops.conv2d(filter_out, k_h=1, k_w=1, strides=1)
    
    def call(self, x):

        conv1 = self.conv1(x)
        norm1 = self.norm1(conv1)
        actv1  = tf.nn.leaky_relu(norm1, 0.1)

        conv2 = self.conv2(actv1)
        norm2 = self.norm2(conv2)

        return tf.add(self.identity(x), norm2)

class resLayer(Model):

    def __init__(self, filters, filter_in, k_h, k_w, strides=1, dilation_rate=1):
        super(resLayer, self).__init__()
        self.sequence = list()

        for f_in, f_out in zip([filter_in]+ list(filters), filters):
            self.sequence.append(resBlock(f_in, f_out, k_h=k_h, k_w=k_w, strides=strides, dilation_rate=dilation_rate))
    
    def call(self, x):
        for unit in self.sequence:
            x=unit(x)
        return x

class model(Model):
    def __init__(self, batch_size = 32, im_height=32, im_width= 128, da_kernel_size=3, dilation_rate=1):
        super(model, self).__init__()

        self.fc_dim = int(im_height*im_width)
        self.im_height = im_height
        self.im_width = im_width

        # sky encode fully conv layer
        self.conv1_d = ops.conv2d(output_channels=32, k_h=7, k_w=7, strides=1) # TODO stride applied
        self.norm1_d = tfa.layers.InstanceNormalization()

        self.conv2_d = ops.conv2d(output_channels=64, k_h=3, k_w=3, strides=2) # TODO stride applied
        self.norm2_d = tfa.layers.InstanceNormalization()

        self.conv3_d = ops.conv2d(output_channels=128, k_h=3, k_w=3, strides=2) # TODO stride applied
        self.norm3_d = tfa.layers.InstanceNormalization()

        self.res = resLayer((128,128,128,128,128,128), 128, k_h=da_kernel_size, k_w=da_kernel_size, strides=1, dilation_rate=dilation_rate)

        # sky_decode
        self.conv3_f = ops.deconv2d(output_channels=64, output_imshape=[int(im_height/ 2), int(im_width/ 2)], k_h=3, k_w=3, method='resize')
        self.norm3_f = tfa.layers.InstanceNormalization()

        self.conv2_f = ops.deconv2d(output_channels=32, output_imshape=[int(im_height), int(im_width)], k_h=3, k_w=3, method='resize')
        self.norm2_f = tfa.layers.InstanceNormalization()

        self.conv1_f = ops.conv2d(output_channels=3, k_h=7, k_w=7, strides=1)

        # sun_decode
        self.conv3_u = ops.deconv2d(output_channels=64, output_imshape=[int(im_height/ 2), int(im_width/ 2)], k_h=3, k_w=3, method='resize')
        self.norm3_u = tfa.layers.InstanceNormalization()

        self.conv2_u = ops.deconv2d(output_channels=32, output_imshape=[int(im_height), int(im_width)], k_h=3, k_w=3, method='resize')
        self.norm2_u = tfa.layers.InstanceNormalization()

        self.conv1_u = ops.conv2d(output_channels=3, k_h=7, k_w=7, strides=1)

        # enhanceSunRadiance
        self.sun = sunRadNet()

    def encode(self, x, training="training"):
        
        conv1_d = self.conv1_d(x)
        norm1_d = self.norm1_d(conv1_d)
        actv1_d = tf.nn.leaky_relu(norm1_d, 0.1)

        conv2_d = self.conv2_d(actv1_d)
        norm2_d = self.norm2_d(conv2_d)
        actv2_d = tf.nn.leaky_relu(norm2_d, 0.1)

        conv3_d = self.conv3_d(actv2_d)
        norm3_d = self.norm3_d(conv3_d)
        actv3_d = tf.nn.leaky_relu(norm3_d, 0.1)

        out = self.res(actv3_d)

        return out

    def sky_decode(self, x, _input, training="training"):

        conv3_f = self.conv3_f(x)
        norm3_f = self.norm3_f(conv3_f)
        actv3_f = tf.nn.leaky_relu(norm3_f, 0.1)

        conv2_f = self.conv2_f(actv3_f)
        norm2_f = self.norm2_f(conv2_f)
        actv2_f = tf.nn.leaky_relu(norm2_f, 0.1)
        
        conv1_f = self.conv1_f(actv2_f)
        sky_pred = tf.nn.leaky_relu(conv1_f, 0.1)

        sky_pred = tf.add_n([_input, sky_pred])
        sky_pred = tf.nn.relu(sky_pred)
        return sky_pred
        
    def sun_decode(self, x, sun_cam1, sun_cam2, sun_cam3, sun_rad, training="training"):
        
        # In order to share the spatial information, I chose add gate for gradient distribution
        # sun_cam3_t = tf.tile(sun_cam3, [1,1,1,128])
        # skip3 = tf.add_n([sun_cam3_t, x])
        
        # conv3_u = self.conv3_u(skip3)
        conv3_u = self.conv3_u(x)
        norm3_u = self.norm3_u(conv3_u)
        actv3_u = tf.nn.leaky_relu(norm3_u, 0.1)

        # sun_cam2_t = tf.tile(sun_cam2, [1,1,1,64])
        # skip2 = tf.add_n([sun_cam2_t, actv3_u])

        # conv2_u = self.conv2_u(skip2)
        conv2_u = self.conv2_u(actv3_u)
        norm2_u = self.norm2_u(conv2_u)
        actv2_u = tf.nn.leaky_relu(norm2_u, 0.1)

        # sun_cam1_t = tf.tile(sun_cam1, [1,1,1,32])
        # skip1 = tf.add_n([sun_cam1_t, actv2_u])

        # conv1_u = self.conv1_u(skip1)
        conv1_u = self.conv1_u(actv2_u)
        actv1_u = tf.nn.leaky_relu(conv1_u, 0.1)
        
        # "add" preserves radiance value of the sun
        sun_rad_t = tf.add_n([sun_rad, actv1_u])
        sun_rad_t = tf.nn.relu(sun_rad_t)
        return sun_rad_t

    def sun_rad_estimation(self, jpeg_img_float, sun_cam1, sun_cam2, sun_cam3, sunpose_pred, training="training"):
        
        normed_sunpose_pred = tf.divide(sunpose_pred, tf.reduce_max(sunpose_pred))
        resized_sum_cam2 = tf.image.resize(sun_cam2, size=(self.im_height, self.im_width))
        resized_sum_cam3 = tf.image.resize(sun_cam3, size=(self.im_height, self.im_width))
        
        plz = tf.concat([jpeg_img_float, sun_cam1, resized_sum_cam2, resized_sum_cam3], axis=-1)
        sun_rad, gamma, beta = self.sun(normed_sunpose_pred, plz, training)

        sun_rad_t = tf.tile(sun_rad, [1,1,1,3])

        return sun_rad_t, gamma, beta

    def blending(self, sky_pred, sun_pred, training="training"):
        # concat vs add  : add win!
        out = tf.add_n([sky_pred, sun_pred])

        return out