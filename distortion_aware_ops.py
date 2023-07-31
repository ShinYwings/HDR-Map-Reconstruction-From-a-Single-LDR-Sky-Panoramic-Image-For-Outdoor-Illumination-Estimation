import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class conv2d(Layer):
    """Only support "channel last" data format"""
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='VAILD',
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 skydome=True):
        
        super(conv2d, self).__init__()
        self.filters = filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dilation_rate = dilation_rate
        self.skydome = skydome

    def build(self, input_shape):
        _, h, w, input_dim = input_shape
        k_h, k_w = self.kernel_size, self.kernel_size
        kernel_shape = [int(k_h*k_w*input_dim), self.filters]
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype)
       
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            trainable=True,
            dtype=self.dtype)

        self.offset = self.distortion(h, w, skydome=self.skydome)

        super(conv2d, self).build(input_shape=input_shape)

    def call(self, inputs):
        
        # add padding if needed
        inputs = self._pad_input(inputs)       
        offset = self.offset

        # some length
        batch_size, in_h, in_w, channel_in = inputs.get_shape() # input feature map size
        out_h, out_w = offset.get_shape()[1: 3]  # output feature map size
        
        filter_h, filter_w = self.kernel_size, self.kernel_size
        
        # get x, y axis offset
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
        
        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.cast(i, dtype=tf.float32) for i in [y, x]]


        # add offset
        y = tf.add_n([y, y_off])
        x = tf.add_n([x, x_off])
        y = tf.clip_by_value(y, 0, in_h - 1)
        
        # consider 360 degree
        x= tf.where( x < 0 , tf.add(x, in_w), x)
        x= tf.where( x > in_w - 1 , tf.subtract(x, in_w), x)

        y, x = [tf.tile(i, [batch_size, 1, 1, 1]) for i in [y, x]] # a pixel in the output feature map has several same offsets 

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), dtype=tf.int32) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1

        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        
        # consider 360 degree
        x0_w, x1_w = x0, x1 # leave original one in order to yield weights
        x0, x1 = [tf.where( i < 0 , tf.add(i, in_w), i) for i in [x0, x1]]
        x0, x1 = [tf.where( i > in_w - 1 , tf.subtract(i, in_w), i) for i in [x0, x1]]
    
        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]

        p0, p1, p2, p3 = [tf.cast(self._get_pixel_values_at_point(inputs, i)
                            , dtype=tf.float32) for i in indices]

        # cast to float
        x0_w, x1_w, y0, y1 = [tf.cast(i, dtype=tf.float32) for i in [x0_w, x1_w, y0, y1]]
        
        # weights
        w0 = tf.multiply(tf.subtract(y1, y), tf.subtract(x1_w, x))
        w1 = tf.multiply(tf.subtract(y1, y), tf.subtract(x, x0_w))
        w2 = tf.multiply(tf.subtract(y, y0), tf.subtract(x1_w, x))
        w3 = tf.multiply(tf.subtract(y, y0), tf.subtract(x, x0_w))

        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]

        # bilinear interpolation
        pixels = tf.add_n([tf.multiply(w0, p0), tf.multiply(w1, p1), 
                        tf.multiply(w2, p2), tf.multiply(w3, p3)])

        pixels = tf.reshape(pixels, [batch_size, out_h* out_w, filter_h* filter_w* channel_in])

        out = tf.matmul(pixels, self.kernel)
        
        out = tf.nn.bias_add(out, self.bias)

        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters])
        
        return out

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        
        Don't use dilated filter itself, only offset can expand the kernel gap

        :param inputs:
        :return: padded input feature map
        """
        in_shape = inputs.get_shape().as_list()[1: 3]
        padding_list = []
        for i in range(2):
            same_output = (in_shape[i] + self.strides - 1) // self.strides
            valid_output = (in_shape[i] - self.kernel_size + self.strides) // self.strides
            if same_output == valid_output:
                padding_list += [0, 0]
            else:
                p = self.kernel_size - 1
                p_0 = p // 2
                padding_list += [p_0, p - p_0]
        if sum(padding_list) != 0:
            padding = [[0, 0],
                        [padding_list[0], padding_list[1]],  # top, bottom padding
                        [padding_list[2], padding_list[3]],  # left, right padding
                        [0, 0]]
            inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map

        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [tf.cast(i, dtype=tf.int32) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(i,
                                               [1, self.kernel_size, self.kernel_size, 1],
                                               [1, self.strides, self.strides, 1],
                                               [1, 1, 1, 1], # dilation_rate must be 1
                                               'VALID')
                for i in [x, y]]   # shape [1, 1, feat_w - kernel_size + 1, feat_h * kernel_size]    [0 1 2 0 1 2 0 1 2]
        return y, x

    def _get_pixel_values_at_point(self, inputs, indices):
        """get pixel values

        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.get_shape().as_list()[0: 4]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))

        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

    def make_grid(self):
        # R_grid should be upside down because image pixel coordinate is orientied from top left
        assert self.kernel_size % 2 == 1, "kernel_size must be odd number, current kernel size : {}".format(self.kernel_size)
        grid = []
        r = self.kernel_size // 2
        
        for y in range(r, -r-1, -1):
            for x in range(r, -r-1, -1):
                grid.append([x,y])

        return grid

    def distortion(self, h, w, skydome=True):

        pi = np.math.pi
        n = self.kernel_size // 2
        middle = n * (self.kernel_size + 1)
        
        unit_w = tf.divide(2 * pi, w)
        unit_h = tf.divide(pi, h * 2 if skydome else h)

        rho = tf.math.tan(unit_w) * self.dilation_rate

        v = tf.constant([0., 1., 0.])

        r_grid= self.make_grid()
        
        x = int(w * 0.5)
        
        res = list()

        for y in range(0,h):

            # radian
            theta = (x - 0.5 * w) * unit_w
            phi   = (h - y) * unit_h if skydome else (h * 0.5 - y) * unit_h

            x_u = tf.math.cos(phi)*tf.math.cos(theta)
            y_u = tf.math.sin(phi)
            z_u = tf.math.cos(phi)*tf.math.sin(theta)
            p_u = tf.constant([x_u.numpy(), y_u.numpy(), z_u.numpy()])

            t_x = tf.linalg.cross(v, p_u)
            t_y = tf.linalg.cross(p_u,t_x)
            
            r_sphere = list()
            for r in r_grid:
                r_sphere.append(tf.multiply(rho, tf.add(r[0] * t_x, r[1] * t_y)))
            r_sphere = tf.squeeze(r_sphere)
            p_ur = tf.add(p_u, r_sphere)
            
            k = list()
            for ur_i in p_ur:
                if ur_i[0] > 0:
                    theta_r = tf.math.atan2(ur_i[2], ur_i[0])
                elif ur_i[0] < 0:
                    if ur_i[2] >=0:
                        theta_r = tf.math.atan2(ur_i[2], ur_i[0]) + pi
                    else:
                        theta_r = tf.math.atan2(ur_i[2], ur_i[0]) - pi
                else:
                    if ur_i[2] > 0:
                        theta_r = pi*0.5
                    elif ur_i[2] < 0:
                        theta_r = -pi*0.5
                    else:
                        raise Exception("undefined coordinates")
                        
                phi_r = tf.math.asin(ur_i[1])

                x_r = (tf.divide(theta_r, pi) + 1)*0.5*w
                y_r = (1. - tf.divide(2*phi_r, pi))*h if skydome else (0.5 - tf.divide(phi_r, pi))*h

                k.append([y_r, x_r])

            offset = tf.subtract(k, k[middle])
            res.append(offset)

        res = tf.convert_to_tensor(res)
        
        res = tf.stack([res] * w)
        res = tf.transpose(res, [1, 0, 2, 3])   # 32, 128, 9, 2 ( h, w, grid #, (y,x) )
        res = tf.expand_dims(res, 0)  # 1, 32, 128, 9, 2 ( b, h, w, grid #, (y,x) )
        
        return res

class deconv2d(Layer):
    """Only support resized method"""
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 output_imshape=[],
                 padding='VAILD',
                 dilation_rate=1,
                 skydome=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        
        super(deconv2d, self).__init__()
        self.filters = filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.output_imshape = tf.cast(output_imshape, dtype=tf.int32).numpy()
        self.padding=padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.dilation_rate = dilation_rate
        self.skydome=skydome

    def build(self, input_shape):

        k_h, k_w = self.kernel_size, self.kernel_size
        kernel_shape = [int(k_h*k_w*input_shape[-1]), self.filters]
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype)
       
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            trainable=True,
            dtype=self.dtype)

        self.offset = self.distortion(self.output_imshape[0], self.output_imshape[1], skydome=self.skydome)

        super(deconv2d, self).build(input_shape=input_shape)

    def call(self, inputs):
        
        # resized method refered from "https://distill.pub/2016/deconv-checkerboard/"
        im_resized = tf.image.resize(inputs, (self.output_imshape[0], self.output_imshape[1]), method=tf.image.ResizeMethod.BILINEAR)
        self.strides = 1 # make sure the "strides == 1" in a way of resize deconv method.

        # add padding if needed
        im_resized = self._pad_input(im_resized)       
        offset = self.offset
        
        # some length
        batch_size, in_h, in_w, channel_in = im_resized.get_shape() # input feature map size
        out_h, out_w = offset.get_shape()[1: 3]  # output feature map size
        
        filter_h, filter_w = self.kernel_size, self.kernel_size
        
        # get x, y axis offset
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
        
        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.cast(i, dtype=tf.float32) for i in [y, x]]

        # add offset
        y = tf.add_n([y, y_off])
        x = tf.add_n([x, x_off])
        y = tf.clip_by_value(y, 0, in_h - 1)
        
        # consider 360 degree
        x= tf.where( x < 0 , tf.add(x, in_w), x)
        x= tf.where( x > in_w - 1 , tf.subtract(x, in_w), x)

        y, x = [tf.tile(i, [batch_size, 1, 1, 1]) for i in [y, x]] # a pixel in the output feature map has several same offsets 

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), dtype=tf.int32) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1

        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        
        # consider 360 degree
        x0_w, x1_w = x0, x1 # leave original one in order to yield weights
        x0, x1 = [tf.where( i < 0 , tf.add(i, in_w), i) for i in [x0, x1]]
        x0, x1 = [tf.where( i > in_w - 1 , tf.subtract(i, in_w), i) for i in [x0, x1]]
    
        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]

        p0, p1, p2, p3 = [tf.cast(self._get_pixel_values_at_point(im_resized, i)
                            , dtype=tf.float32) for i in indices]

        # cast to float
        x0_w, x1_w, y0, y1 = [tf.cast(i, dtype=tf.float32) for i in [x0_w, x1_w, y0, y1]]
        
        # weights
        w0 = tf.multiply(tf.subtract(y1, y), tf.subtract(x1_w, x))
        w1 = tf.multiply(tf.subtract(y1, y), tf.subtract(x, x0_w))
        w2 = tf.multiply(tf.subtract(y, y0), tf.subtract(x1_w, x))
        w3 = tf.multiply(tf.subtract(y, y0), tf.subtract(x, x0_w))

        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]

        # bilinear interpolation
        pixels = tf.add_n([tf.multiply(w0, p0), tf.multiply(w1, p1), 
                        tf.multiply(w2, p2), tf.multiply(w3, p3)])

        pixels = tf.reshape(pixels, [batch_size, out_h* out_w, filter_h* filter_w* channel_in])

        out = tf.matmul(pixels, self.kernel)
        
        out = tf.nn.bias_add(out, self.bias)

        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters])
        
        return out

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        
        Don't use dilated filter itself, only offset can expand the kernel gap

        :param inputs:
        :return: padded input feature map
        """
        in_shape = inputs.get_shape().as_list()[1: 3]
        padding_list = []
        for i in range(2):
            same_output = (in_shape[i] + self.strides - 1) // self.strides
            valid_output = (in_shape[i] - self.kernel_size + self.strides) // self.strides
            if same_output == valid_output:
                padding_list += [0, 0]
            else:
                p = self.kernel_size - 1
                p_0 = p // 2
                padding_list += [p_0, p - p_0]
        if sum(padding_list) != 0:
            padding = [[0, 0],
                        [padding_list[0], padding_list[1]],  # top, bottom padding
                        [padding_list[2], padding_list[3]],  # left, right padding
                        [0, 0]]
            inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map

        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [tf.cast(i, dtype=tf.int32) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(i,
                                               [1, self.kernel_size, self.kernel_size, 1],
                                               [1, self.strides, self.strides, 1],
                                               [1, 1, 1, 1], # dilation_rate must be 1
                                               'VALID')
                for i in [x, y]]   # shape [1, 1, feat_w - kernel_size + 1, feat_h * kernel_size]    [0 1 2 0 1 2 0 1 2]
        return y, x

    def _get_pixel_values_at_point(self, inputs, indices):
        """get pixel values

        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.get_shape().as_list()[0: 4]

        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))

        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)

    def make_grid(self):
        # R_grid should be upside down because image pixel coordinate is orientied from top left
        assert self.kernel_size % 2 == 1, "kernel_size must be odd number, current kernel size : {}".format(self.kernel_size)
        grid = []
        r = self.kernel_size // 2
        
        for y in range(r, -r-1, -1):
            for x in range(r, -r-1, -1):
                grid.append([x,y])

        return grid

    def distortion(self, h, w, skydome=True):

        pi = np.math.pi
        n = self.kernel_size // 2
        middle = n * (self.kernel_size + 1)
        
        unit_w = tf.divide(2 * pi, w)
        unit_h = tf.divide(pi, h * 2 if skydome else h)

        rho = tf.math.tan(unit_w) * self.dilation_rate

        v = tf.constant([0., 1., 0.])

        r_grid= self.make_grid()
        
        x = int(w * 0.5)
        
        res = list()

        for y in range(0,h):

            # radian
            theta = (x - 0.5 * w) * unit_w
            phi   = (h - y) * unit_h if skydome else (h * 0.5 - y) * unit_h

            x_u = tf.math.cos(phi)*tf.math.cos(theta)
            y_u = tf.math.sin(phi)
            z_u = tf.math.cos(phi)*tf.math.sin(theta)
            p_u = tf.constant([x_u.numpy(), y_u.numpy(), z_u.numpy()])

            t_x = tf.linalg.cross(v, p_u)
            t_y = tf.linalg.cross(p_u,t_x)
            
            r_sphere = list()
            for r in r_grid:
                r_sphere.append(tf.multiply(rho, tf.add(r[0] * t_x, r[1] * t_y)))
            r_sphere = tf.squeeze(r_sphere)
            p_ur = tf.add(p_u, r_sphere)
            
            k = list()
            for ur_i in p_ur:
                if ur_i[0] > 0:
                    theta_r = tf.math.atan2(ur_i[2], ur_i[0])
                elif ur_i[0] < 0:
                    if ur_i[2] >=0:
                        theta_r = tf.math.atan2(ur_i[2], ur_i[0]) + pi
                    else:
                        theta_r = tf.math.atan2(ur_i[2], ur_i[0]) - pi
                else:
                    if ur_i[2] > 0:
                        theta_r = pi*0.5
                    elif ur_i[2] < 0:
                        theta_r = -pi*0.5
                    else:
                        raise Exception("undefined coordinates")
                        
                phi_r = tf.math.asin(ur_i[1])

                x_r = (tf.divide(theta_r, pi) + 1)*0.5*w
                y_r = (1. - tf.divide(2*phi_r, pi))*h if skydome else (0.5 - tf.divide(phi_r, pi))*h

                k.append([y_r, x_r])

            offset = tf.subtract(k, k[middle])
            res.append(offset)

        res = tf.convert_to_tensor(res)
        
        res = tf.stack([res] * w)
        res = tf.transpose(res, [1, 0, 2, 3])   # 32, 128, 9, 2 ( h, w, grid #, (y,x) )
        res = tf.expand_dims(res, 0)  # 1, 32, 128, 9, 2 ( b, h, w, grid #, (y,x) )
        
        return res
