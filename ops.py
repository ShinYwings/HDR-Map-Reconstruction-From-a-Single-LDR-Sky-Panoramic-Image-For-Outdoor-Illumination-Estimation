import tensorflow as tf
from tensorflow.keras.layers import Layer

class conv2d(Layer):
    def __init__(self, 
                output_channels="output_channels", 
                strides="strides", 
                k_h="k_h", 
                k_w= "k_w", 
                padding="SAME",
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros'):

        super(conv2d, self).__init__()
        self.output_channels = output_channels
        self.k_w = k_w
        self.k_h = k_h
        self.strides = [1,strides, strides,1]
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        # w_init = tf.random.truncated_normal(shape=[self.k_h,self.k_w, input_shape[-1], self.output_channels])
        # self.w = tf.Variable(initial_value=w_init, trainable=True)
        # bias_init = tf.random.truncated_normal(shape=[self.output_channels])
        # self.biases = tf.Variable(initial_value=bias_init, trainable=True) 

        # [Simple Version]
        self.w = self.add_weight(name="w", 
                                shape=(self.k_h, self.k_w, 
                                        input_shape[-1], self.output_channels),
                               initializer=self.kernel_initializer,
                               trainable=True)
        self.biases = self.add_weight(name="b", shape=(self.output_channels,),
                               initializer=self.bias_initializer ,
                               trainable=True)

        super(conv2d, self).build(input_shape=input_shape)

    def call(self, input):
        return tf.nn.bias_add(tf.nn.conv2d(input, self.w, strides=self.strides, padding=self.padding), self.biases)

class deconv2d(Layer):
    def __init__(self, 
                 output_channels="output_channels", 
                 output_imshape=[],
                 k_h="k_h", 
                 k_w= "k_w",
                 strides=1,
                 padding="SAME", 
                 method='resize', 
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):

        super(deconv2d, self).__init__()
        self.output_channels = output_channels
        self.k_w = k_w
        self.k_h = k_h
        self.strides = strides,
        self.output_imshape = tf.cast(output_imshape, dtype=tf.int32).numpy()
        self.padding = padding
        self.method= method
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):

        if self.method == 'upsample':
            '''deconv method : checkerboard issue'''
            # w_init = tf.random.truncated_normal(shape=[self.k_h, self.k_w, self.output_channels, input_shape[-1]])
            # self.kernel = tf.Variable(initial_value=w_init, trainable=True)
            # bias_init = tf.random.truncated_normal(shape=[self.output_channels])
            # self.biases = tf.Variable(initial_value=bias_init, trainable=True)

            self.kernel = self.add_weight(
                name='kernel_deconv2d',
                shape=(self.k_h, self.k_w, self.output_channels, input_shape[-1]),
                initializer=self.kernel_initializer,
                trainable=True,
                dtype=self.dtype)
       
            self.biases = self.add_weight(
                name='bias_deconv2d',
                shape=(self.output_channels,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)

        elif self.method == 'resize':
            '''resize-conv method http://distill.pub/2016/deconv-checkerboard/'''
            # w_init = tf.random.truncated_normal(shape=[self.k_h, self.k_w, input_shape[-1], self.output_channels])
            # self.w = tf.Variable(initial_value=w_init, trainable=True)
            # bias_init = tf.random.truncated_normal(shape=[self.output_channels])
            # self.biases = tf.Variable(initial_value=bias_init, trainable=True)            

            self.kernel = self.add_weight(
                name='kernel_deconv2d',
                shape= (self.k_h, self.k_w, input_shape[-1], self.output_channels),
                initializer=self.kernel_initializer,
                trainable=True,
                dtype=self.dtype)
       
            self.biases = self.add_weight(
                name='bias_deconv2d',
                shape=(self.output_channels,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)

        super(deconv2d, self).build(input_shape=input_shape)

    def call(self, input):
        batch_size, input_h, _, _ = input.get_shape().as_list() #bhwc
    
        if self.method == 'upsample':
            output_shape = [batch_size, self.output_imshape[0], self.output_imshape[1], self.output_channels]
            strides = int(self.output_imshape[0] / input_h)
            deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input, self.kernel, output_shape=output_shape, strides=strides, padding=self.padding), self.biases)  # deconv
        
        elif self.method == 'resize':
            im_resized = tf.image.resize(input, (self.output_imshape[0], self.output_imshape[1]), method=tf.image.ResizeMethod.BILINEAR)
            strides = [1,1,1,1]
            deconv = tf.nn.bias_add(tf.nn.conv2d(im_resized, self.kernel, strides=strides, padding=self.padding), self.biases)
        
        return deconv

class fc2d(Layer):
    def __init__(self, fc_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(fc2d, self).__init__()
        self.fc_dim = fc_dim
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self,input_shape):

        in_height = input_shape[1]
        in_width = input_shape[2]
        in_channels = input_shape[3]
        self.in_dim = in_height * in_width * in_channels
        
        # w_init = tf.random.truncated_normal(shape=[self.in_dim, self.fc_dim])
        # self.w = tf.Variable(initial_value=w_init, trainable=True)

        # bias_init = tf.random.truncated_normal(shape=[self.fc_dim])
        # self.biases = tf.Variable(initial_value=bias_init, trainable=True) 

        self.w = self.add_weight(
                name='kernel_fc',
                shape= (self.in_dim, self.fc_dim),
                initializer=self.kernel_initializer,
                trainable=True,
                dtype=self.dtype)
       
        self.biases = self.add_weight(
                name='bias_fc',
                shape=(self.fc_dim,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)
        
        super(fc2d, self).build(input_shape)

    def call(self, input):
        
        fc = tf.reshape(input, [-1, self.in_dim])
        fc = tf.matmul(fc, self.w)
        fc = tf.add(fc, self.biases)
        fc = tf.reshape(fc, [-1, 1, 1, self.fc_dim])

        return fc

class dfc2d(Layer):
    def __init__(self, 
                out_height="out_height", 
                out_width="out_width",
                out_channels="out_channels",
                kernel_initializer='glorot_uniform', 
                bias_initializer='zeros',):

        # de-fully connected
        # input_:  [batch, 1, 1, fc_dim]
        super(dfc2d, self).__init__()
        self.out_height = out_height
        self.out_width = out_width
        self.out_channels = out_channels
        self.out_dim = out_height*out_width*out_channels
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):

        # w_init = tf.random.truncated_normal(shape=[self.fc_dim, self.out_dim])
        # self.w = tf.Variable(initial_value=w_init, trainable=True)

        # bias_init = tf.random.truncated_normal(shape=[self.out_dim])
        # self.biases = tf.Variable(initial_value=bias_init, trainable=True)

        # input = tf.reshape(input, [-1, self.fc_dim])
        self.fc_dim = input_shape[-1]
        
        self.w = self.add_weight(
                name = "w_defc",
                shape= (self.fc_dim, self.out_dim),
                initializer=self.kernel_initializer,
                trainable=True)
       
        self.biases = self.add_weight(
                name = "b_defc",
                shape=(self.out_dim),
                initializer=self.bias_initializer,
                trainable=True)
        
        super(dfc2d, self).build(input_shape)

    def call(self, input):
        
        # input = tf.reshape(input, [-1, fc_dim])
        # w_init = tf.random.truncated_normal(shape=[fc_dim, self.out_dim])
        # w = tf.Variable(initial_value=(w_init,), trainable=True)
        # fc = tf.matmul(input, w)
        # bias_init = tf.random.truncated_normal(shape=self.out_dim)
        # biases = tf.Variable(initial_value=(bias_init,), trainable=True)

        input = tf.reshape(input, [-1, self.fc_dim])
        fc = tf.matmul(input, self.w)
        fc = tf.nn.bias_add(fc, self.biases)
        fc = tf.reshape(fc, [-1, self.out_height, self.out_width, self.out_channels])

        return fc

class batch_normalization(Layer):
    
    def __init__(self, decay=0.9, epsilon=1e-5, **kwargs):
        super(batch_normalization, self).__init__(**kwargs)
        self.decay = decay
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=[input_shape[-1], ],
                                     initializer=tf.initializers.ones,
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=[input_shape[-1], ],
                                    initializer=tf.initializers.zeros,
                                    trainable=True)
        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=[input_shape[-1], ],
                                           initializer=tf.initializers.zeros,
                                           trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
                                               shape=[input_shape[-1], ],
                                               initializer=tf.initializers.ones,
                                               trainable=False)
        super(batch_normalization, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        """
        variable = variable * decay + value * (1 - decay)
        """
        delta = variable * self.decay + value * (1 - self.decay)
        return variable.assign(delta)

    def call(self, inputs, training= False):
        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs, list(range(len(inputs.shape) - 1)))
            mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        
        output = tf.nn.batch_normalization(inputs,
                                           mean=mean,
                                           variance=variance,
                                           offset=self.beta,
                                           scale=self.gamma,
                                           variance_epsilon=self.epsilon)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class maxpool2d(Layer):
    def __init__(self, kernel_size="kernel_size", strides=None, padding="SAME"):
        
        super(maxpool2d, self).__init__()

        if strides == None:
            strides = kernel_size
        
        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.strides = [1, strides, strides, 1]
        self.padding = padding
    
    def call(self, x):
        return tf.nn.max_pool(x, ksize= self.kernel_size, strides=self.strides, padding= self.padding)

class avgpool2d(Layer):
    def __init__(self, kernel_size="kernel_size", strides=None, padding="SAME"):

        super(avgpool2d, self).__init__()

        if strides == None:
            strides = kernel_size
        
        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.strides = [1, strides, strides, 1]
        self.padding = padding
    
    def call(self, x):
        return tf.nn.avg_pool(x, ksize= self.kernel_size, strides=self.strides, padding= self.padding)

class elu(Layer):
    def __init__(self):
        super(elu, self).__init__()
    
    def call(self, x):
        return tf.nn.elu(x)

class relu(Layer):
    def __init__(self):
        super(relu, self).__init__()
    
    def call(self, x):
        return tf.nn.relu(x)

class tanh(Layer):
    def __init__(self):
        super(tanh, self).__init__()
    
    def call(self, x):
        return tf.nn.tanh(x)

class sigmoid(Layer):
    def __init__(self):
        super(sigmoid, self).__init__()
    
    def call(self, x):
        return tf.nn.sigmoid(x)

class dropout(Layer):
    def __init__(self, keep_prob=0.5):
        super(dropout, self).__init__()
        self.keep_prob = keep_prob
    
    def call(self, x, isTraining):

        if isTraining:
            return tf.nn.dropout(x, rate=self.keep_prob)
        else:
            return x