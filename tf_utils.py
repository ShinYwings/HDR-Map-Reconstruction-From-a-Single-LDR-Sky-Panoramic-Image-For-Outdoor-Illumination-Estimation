import tensorflow as tf
import tensorflow_addons as tfa
import utils
import numpy as np

PI = np.math.pi

def wasserstein_distance(x, y, b, n):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    
    x = tf.reshape(x,(b,-1))
    y = tf.reshape(y,(b,-1))

    all_values = tf.concat([x,y], axis=-1)
    all_values = tf.sort(all_values)
    reverse_all_values = all_values[:,:-1]

    deltas = tf.math.subtract(all_values[:,1:], reverse_all_values)

    x = x[:,::-1] # 오름차순
    y = y[:,::-1]
    x_cdf_indices = tf.searchsorted(x, reverse_all_values, side="right")
    y_cdf_indices = tf.searchsorted(y, reverse_all_values, side="right")
    x_cdf_indices = tf.cast(x_cdf_indices, dtype=tf.float32)
    y_cdf_indices = tf.cast(y_cdf_indices, dtype=tf.float32)

    x_cdf = tf.math.divide(x_cdf_indices, n)
    y_cdf = tf.math.divide(y_cdf_indices, n)

    output = tf.math.abs(x_cdf - y_cdf)
    output = tf.math.multiply(output, deltas)

    output = tf.math.reduce_sum(output, axis=-1)
    output = tf.reshape(output, [-1,1,1,1])

    return output
    
def compare_luminance(pred, gt):
    """
    Global luminance comparisn , NO Top-K
    """
    b,h,w,c  = pred.shape
    b2,h2,w2,c2 = gt.shape
    
    assert b == b2 and c == c2, "batch size of img1 and img2 must be equal"

    n = h*w # Total pixel number
    
    pred_blue, pred_green, pred_red = tf.split(pred, num_or_size_splits=3, axis=-1)
    gt_blue, gt_green, gt_red = tf.split(gt, num_or_size_splits=3, axis=-1)

    em_distance_blue = wasserstein_distance(pred_blue, gt_blue, b, n)
    em_distance_green = wasserstein_distance(pred_green, gt_green, b, n)
    em_distance_red = wasserstein_distance(pred_red, gt_red, b, n)

    # TODO 06/04 15:17   sum -> mean
    em_distance = (em_distance_blue + em_distance_green + em_distance_red) / 3.
    
    return em_distance

def DoG(img, kernel_size=3, sigma=1.2489996, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    # Difference of Gaussian
    _,h,w,_ = img.get_shape()
    img = tf.image.resize(img, (2*h, 2*w))
    base_image = tfa.image.gaussian_filter2d(img, filter_shape=(kernel_size,kernel_size), sigma=sigma)
    # overlap sigma values in order to subtract images
    gaussian_kernels1 = [1.2262735, 1.5450078, 1.9465878, 2.452547] # base sigma = 1.6
    gaussian_kernels2 = [1.5450078, 1.9465878, 2.452547, 3.0900156]
    gaussian_images1 = [tfa.image.gaussian_filter2d(base_image, filter_shape=(kernel_size,kernel_size), sigma=gaussian_kernel, padding="REFLECT") for gaussian_kernel in gaussian_kernels1]
    gaussian_images2 = [tfa.image.gaussian_filter2d(base_image, filter_shape=(kernel_size,kernel_size), sigma=gaussian_kernel, padding="REFLECT") for gaussian_kernel in gaussian_kernels2]
    dog_image1, dog_image2, dog_image3, dog_image4 = [tf.math.subtract(second_image, first_image) for first_image, second_image in zip(gaussian_images1, gaussian_images2)]

    return dog_image1, dog_image2, dog_image3, dog_image4

def rgb2gray(rgb):
    red, green, blue = tf.split(rgb, num_or_size_splits=3, axis=-1)
    gray = 0.2627*red + 0.6780*green + 0.0593*blue
    return gray

def bgr2gray(bgr):
    blue, green, red = tf.split(bgr, num_or_size_splits=3, axis=-1)
    gray = 0.2627*red + 0.6780*green + 0.0593*blue
    return gray

def rgb2bgr(rgb):
    red, green, blue = tf.split(rgb, num_or_size_splits=3, axis=-1)
    bgr = tf.concat([blue, green, red], axis=3)
    return bgr

def bgr2rgb(bgr):
    blue, green, red = tf.split(bgr, num_or_size_splits=3, axis=-1)
    rgb = tf.concat([red, green, blue], axis=3)
    return rgb

def sphere2world(sunpose, h, w, skydome = True):
    x, y = sunpose
    
    unit_w = tf.divide(2 * PI, w)
    unit_h = tf.divide(PI, h * 2 if skydome else h)
    
    # degree in xy coordinate to radian
    theta = (x - 0.5 * w) * unit_w
    phi   = (h - y) * unit_h if skydome else (h * 0.5 - y) * unit_h

    x_u = tf.math.cos(phi) * tf.math.cos(theta)
    y_u = tf.math.sin(phi)
    z_u = tf.math.cos(phi) * tf.math.sin(theta)
    p_u = [x_u, y_u, z_u]

    return tf.convert_to_tensor(p_u)

def sunpose_init(i, h, w):
    # xy coord to degree
    # gap value + init (half of the gap value)

    x = ((i+1.) - tf.floor(i/w) * w - 1.) * (360.0/w) + (360.0/(w*2.)) 
    y = (tf.floor(i/w)) * (90./h) + (90./(2.*h))

    # deg2rad
    phi = (y) * (PI / 180.)
    theta = (x - 180.0) * (PI / 180.)

    # rad2xyz
    x_u = tf.math.cos(phi) * tf.math.cos(theta)
    y_u = tf.math.sin(phi)
    z_u = tf.math.cos(phi) * tf.math.sin(theta)
    p_u = [x_u, y_u, z_u]
    
    return tf.convert_to_tensor(p_u)

def positional_encoding(_input, with_r=False):
    # coord conv
    b, h, w = _input.get_shape()[0:3]

    w_range = tf.linspace(-1., 1., w)
    h_range = tf.linspace(-1., 1., h)
    x, y = tf.meshgrid(w_range, h_range)
    x, y = [tf.reshape(i, [1, h, w, 1]) for i in [x, y]]
    normalized_coord = tf.concat([x,y], axis=-1)

    if with_r:
        half_h = h * 0.5
        half_w = w * 0.5
        r = tf.sqrt(tf.square(x - half_w) + tf.square(y - half_h))
        normalized_coord = tf.concat([normalized_coord, r], axis=-1)

    normalized_coord = tf.tile(normalized_coord, [b,1,1,1])
    pose_aware_input = tf.concat([_input, normalized_coord], axis=-1)

    return pose_aware_input

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
    
def get_tensor_shape(x):
    a = x.get_shape().as_list()
    b = [tf.shape(x)[i] for i in range(len(a))]
    def _select_one(aa, bb):
        if type(aa) is int:
            return aa
        else:
            return bb
    return [_select_one(aa, bb) for aa, bb in zip(a, b)]

def pass_net_nx(func, in_img, n):
    b, h, w, c, = get_tensor_shape(in_img)
    def _get_nx(x):
        s, r = x//n, x%n
        s = tf.cond(
            tf.equal(r, 0),
            lambda: s,
            lambda: s + 1,
        )
        return n*s
    nx_h = _get_nx(h)
    nx_w = _get_nx(w)
    def _get_rl_rr(x, nx):
        r = nx - x
        rl = r//2
        rr = r - rl
        return rl, rr
    in_img = tf.pad(in_img, [[0, 0], _get_rl_rr(h, nx_h), _get_rl_rr(w, nx_w), [0, 0]], mode='symmetric')
    in_img = tf.reshape(in_img, [b, nx_h, nx_w, c])
    out_img = func(in_img)
    out_img = tf.image.resize_with_crop_or_pad(out_img, h, w)
    return out_img


def sample_1d(
    img,   # [b, h, c]
    y_idx, # [b, n], 0 <= pos < h, dtpye=int32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y_idx)
    
    b_idx = tf.range(b, dtype=tf.int32) # [b]
    b_idx = tf.expand_dims(b_idx, -1)   # [b, 1]
    b_idx = tf.tile(b_idx, [1, n])      # [b, n]
    
    y_idx = tf.clip_by_value(y_idx, 0, h - 1) # [b, n]
    a_idx = tf.stack([b_idx, y_idx], axis=-1) # [b, n, 2]
    
    return tf.gather_nd(img, a_idx)

def interp_1d(
    img, # [b, h, c]
    y,   # [b, n], 0 <= pos < h, dtype=float32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y)
    
    y_0 = tf.floor(y) # [b, n]
    y_1 = y_0 + 1    
    
    _sample_func = lambda y_x: sample_1d(
        img,
        tf.cast(y_x, tf.int32)
    )
    y_0_val = _sample_func(y_0) # [b, n, c]
    y_1_val = _sample_func(y_1)
    
    w_0 = y_1 - y # [b, n]
    w_1 = y - y_0
    
    w_0 = tf.expand_dims(w_0, -1) # [b, n, 1]
    w_1 = tf.expand_dims(w_1, -1)
    
    return w_0*y_0_val + w_1*y_1_val

def get_mean_invcrf(batch_size:int = "batch_size", path = "mean_invEmor.txt"):
    
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    mean_invcrf = [ele.split() for ele in lines]
    mean_invcrf = tf.strings.to_number(mean_invcrf, out_type=tf.float32)
    mean_invcrf = tf.reshape(mean_invcrf, [1,-1])
    mean_invcrf = tf.tile(mean_invcrf, [batch_size, 1])

    return mean_invcrf

def apply_rf(
    x,  # [b, s...]
    rf, # [b, k]
):
    b, *s, = get_tensor_shape(x)
    b, k,  = get_tensor_shape(rf)
    x = interp_1d(
        tf.expand_dims(rf, -1),                              # [b, k, 1] 
        tf.cast((k - 1), tf.float32)*tf.reshape(x, [b, -1]), # [b, ?] 
    ) # [b, ?, 1]
    return tf.reshape(x, [b] + s)

def get_l2_loss(a, b):
    return tf.reduce_mean(tf.square(a - b))

def get_l2_loss_with_mask(a, b):
    return tf.reduce_mean(tf.square(a - b), axis=[1, 2, 3], keepdims=True)

def hdr_logCompression(x, validDR = 10.):
    # 0~1
    # disentangled way
    x = tf.math.multiply(validDR, x)
    numerator = tf.math.log(1.+ x)
    denominator = tf.math.log(1.+validDR)
    output = tf.math.divide(numerator, denominator)

    return output

def hdr_logDecompression(x, validDR = 10.):
    # 0~1
    denominator = tf.math.log(1.+validDR)
    x = tf.math.multiply(x, denominator)
    x = tf.math.exp(x)
    output = tf.math.divide(x-1., validDR)
    
    return output

def createDirectories(path, name="name", dir="dir"):
    
    path = utils.createNewDir(path, dir)
    root_logdir = utils.createNewDir(path, name)
    logdir = utils.createNewDir(root_logdir)

    if dir=="tensorboard":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=False)
        train_summary_writer = tf.summary.create_file_writer(train_logdir)
        test_summary_writer = tf.summary.create_file_writer(test_logdir)
        return train_summary_writer, test_summary_writer, logdir

    if dir=="outputImg":
        train_logdir, test_logdir = utils.createTrainValidationDirpath(logdir, createDir=True)
        return train_logdir, test_logdir

def checkpoint_initialization(model_name : str,
                                pretrained_dir : str,
                                checkpoint_path : str,
                                model="model",
                                optimizer="optimizer",
                                ):
    if pretrained_dir is None:
        checkpoint_path = utils.createNewDir(checkpoint_path, model_name)
    else: checkpoint_path = pretrained_dir
    
    ckpt = tf.train.Checkpoint(
                            epoch = tf.Variable(0),
                            lin=model,
                           optimizer=optimizer,)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    #  if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest {} checkpoint has restored!!'.format(model_name))

    return ckpt, ckpt_manager

def metric_initialization(model_name : str, lr = "lr"):
    
    optimizer = tf.keras.optimizers.Adam(lr)
    train_loss = tf.keras.metrics.Mean(name= 'train_loss_{}'.format(model_name), dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean(name='test_loss_{}'.format(model_name), dtype=tf.float32)

    return optimizer, train_loss, test_loss