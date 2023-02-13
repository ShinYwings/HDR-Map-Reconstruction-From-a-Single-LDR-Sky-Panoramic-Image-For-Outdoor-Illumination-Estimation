import os
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

import utils
import tf_utils
import grad_cam

import generator as g
import discriminator as d
import sunpose_net as sun

from vgg16 import Vgg16

from numpy.random import randint

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)
    
AUTO = tf.data.AUTOTUNE

# Hyper parameters
BATCH_SIZE = 32
EPOCHS = 1000
IMSHAPE = (32, 128, 3)
AZIMUTH_gt = IMSHAPE[1]*0.5-1
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr

# Optimizer Option
LEARNING_RATE = 1e-4

CURRENT_WORKINGDIR = os.getcwd()

SUNPOSE_BIN = tf.convert_to_tensor([tf_utils.sunpose_init(i,IMSHAPE[0],IMSHAPE[1]) for i in range(IMSHAPE[0]*IMSHAPE[1])])

def vMF(x, y, h, w, kappa=80.0):
    # discrete the sky into h*w bins and model the sky probability distirbution. (von Mises-Fisher)
    sp_vec = tf_utils.sphere2world((x, y), h, w, skydome=True)
    sp_vec = tf.expand_dims(sp_vec, axis=0)
    sp_vec = tf.tile(sp_vec, (h*w, 1))
    
    batch_dot = tf.einsum("bc, bc-> b", SUNPOSE_BIN, sp_vec)
    batch_dot = tf.scalar_mul(kappa, batch_dot)
    pdf = tf.math.exp(batch_dot)
    
    return pdf / tf.reduce_sum(pdf)

def _preprocessing(hdr, crf_src, t_src):
    
    b = tf_utils.get_tensor_shape(hdr)[0]

    crf_list = [crf_src[randint(0,len(crf_src))] for _ in range(b)]
    t = [t_src[randint(0,len(t_src))] for _ in range(b)]
    
    crf = tf.convert_to_tensor(crf_list)
    t = tf.convert_to_tensor(t)

    _hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

    # Augment Poisson and Gaussian noise
    sigma_s = 0.08 / 6 * tf.random.uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0,
                                                     dtype=tf.float32, seed=1)
    sigma_c = 0.005 * tf.random.uniform([tf.shape(_hdr_t)[0], 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
    noise_s_map = sigma_s * _hdr_t
    noise_s = tf.random.normal(shape=tf.shape(_hdr_t), seed=1) * noise_s_map
    temp_x = _hdr_t + noise_s
    noise_c = sigma_c * tf.random.normal(shape=tf.shape(_hdr_t), seed=1)
    temp_x = temp_x + noise_c
    _hdr_t = tf.nn.relu(temp_x)

    # Dynamic range clipping
    clipped_hdr_t = tf.clip_by_value(_hdr_t, 0, 1)

    # Camera response function
    ldr = tf_utils.apply_rf(clipped_hdr_t, crf)

    # Quantization and JPEG compression
    quantized_hdr = tf.round(ldr * 255.0)
    quantized_hdr_8bit = tf.cast(quantized_hdr, tf.uint8)
    jpeg_img_list = []
    for i in range(b):
        II = quantized_hdr_8bit[i]
        II = tf.image.adjust_jpeg_quality(II, int(round(float(i)/float(b-1)*10.0+90.0)))
        jpeg_img_list.append(II)
    jpeg_img = tf.stack(jpeg_img_list, 0)
    jpeg_img_float = tf.cast(jpeg_img, tf.float32) / 255.0

    return [_hdr_t, jpeg_img_float]
    
def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'azimuth' : tf.io.FixedLenFeature([], tf.float32),
        'elevation' : tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    hdr = tf.io.decode_raw(example['image'], tf.float32)
    hdr = tf.reshape(hdr, IMSHAPE)
    hdr = hdr[:,:,::-1] # bgr2rgb due to tf.image.rgb2grayscale in preprocessing
    
    hdr_mean = tf.reduce_mean(hdr)
    hdr = 0.5 * hdr / (hdr_mean + 1e-6)

    azimuth = AZIMUTH_gt
    elevation = example['elevation']

    sun_pose = vMF(azimuth ,elevation, IMSHAPE[0], IMSHAPE[1])

    return hdr, sun_pose

def configureDataset(dirpath, train= "train"):

    tfrecords_list = list()
    a = tf.data.Dataset.list_files(os.path.join(dirpath, "*.tfrecord"), shuffle=False)
    tfrecords_list.extend(a)

    ds = tf.data.TFRecordDataset(filenames=tfrecords_list, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.map(_parse_function, num_parallel_calls=AUTO)

    if train:
        ds = ds.shuffle(buffer_size = 10000).batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
    else:
        ds = ds.batch(batch_size=BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
        
    return ds
        
def run(args):
    global BATCH_SIZE, EPOCHS, IMSHAPE, AZIMUTH_gt, LEARNING_RATE, DATASET_DIR, SUNPOSE_BIN
    
    IMSHAPE = (args.imheight, args.imwidth, 3)
    AZIMUTH_gt = IMSHAPE[1]*0.5-1
    
    # Optimizer Option
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batchsize
    EPOCHS = args.epochs
    CURRENT_WORKINGDIR = os.getcwd()
    
    if(IMSHAPE[0] != 32):
        DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, "dataset_{}_{}/tfrecord".format(IMSHAPE[1], IMSHAPE[0]))
    else:
        DATASET_DIR = args.dir
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    PRETRAINED_DIR = args.sky
    SUN_PRETRAINED_DIR = args.sun

    SUNPOSE_BIN = tf.convert_to_tensor([tf_utils.sunpose_init(i,IMSHAPE[0],IMSHAPE[1]) for i in range(IMSHAPE[0]*IMSHAPE[1])])

    """Init Dataset"""
    train_ds = configureDataset(TRAIN_DIR, train=True)
    test_ds  = configureDataset(TEST_DIR, train=False)

    train_crf, test_crf = utils.getDoRF(args.dorf)
    train_t  , test_t   = utils.get_T()

    """CheckPoint Create"""
    checkpoint_path = utils.createNewDir(CURRENT_WORKINGDIR, "checkpoints")

    _gen = g.model(batch_size=BATCH_SIZE, im_height=IMSHAPE[0], im_width=IMSHAPE[1], da_kernel_size=3, dilation_rate=1)
    _sun = sun.model(im_height=IMSHAPE[0], im_width=IMSHAPE[1], da_kernel_size=3, dilation_rate=1)
    _dis = d.model(im_height=IMSHAPE[0], im_width=IMSHAPE[1], da_kernel_size=3, dilation_rate=1)
    vgg = Vgg16(args.vgg)
    vgg2 = Vgg16(args.vgg)
    
    """"Create Output Image Directory"""
    train_summary_writer, test_summary_writer, logdir = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="SKY", dir="tensorboard")
    print(f'tensorboard --logdir={logdir}')
    
    """Model initialization"""
    _, train_total_loss_gen, test_total_loss_gen = tf_utils.metric_initialization("GEN", LEARNING_RATE)
    _, train_total_loss_disc, test_total_loss_disc = tf_utils.metric_initialization("DISC", LEARNING_RATE)

    train_l1_loss = tf.keras.metrics.Mean(name='metric_l1_loss', dtype=tf.float32)
    train_perceptual_loss = tf.keras.metrics.Mean(name= 'metric_perceptual_loss', dtype=tf.float32)
    train_DoG_loss = tf.keras.metrics.Mean(name='metric_DoG_loss', dtype=tf.float32)
    train_gen_loss = tf.keras.metrics.Mean(name= 'metric_gen_loss', dtype=tf.float32)
    train_kl_divergence = tf.keras.metrics.Mean(name= 'metric_kl_divergence', dtype=tf.float32)

    train_generated_loss = tf.keras.metrics.Mean(name= 'metric_generated_loss', dtype=tf.float32)
    train_real_loss = tf.keras.metrics.Mean(name= 'metric_real_loss', dtype=tf.float32)

    test_l1_loss = tf.keras.metrics.Mean(name='test_metric_l1_loss', dtype=tf.float32)
    test_perceptual_loss = tf.keras.metrics.Mean(name= 'test_metric_perceptual_loss', dtype=tf.float32)
    test_DoG_loss = tf.keras.metrics.Mean(name='test_metric_DoG_loss', dtype=tf.float32)
    test_gen_loss = tf.keras.metrics.Mean(name= 'test_metric_gen_loss', dtype=tf.float32)
    test_kl_divergence = tf.keras.metrics.Mean(name= 'test_metric_kl_divergence', dtype=tf.float32)
    
    test_generated_loss = tf.keras.metrics.Mean(name= 'test_metric_generated_loss', dtype=tf.float32)
    test_real_loss = tf.keras.metrics.Mean(name= 'test_metric_real_loss', dtype=tf.float32)

    optimizer_gen  = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    optimizer_disc = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

    if PRETRAINED_DIR is None:
        checkpoint_path = utils.createNewDir(checkpoint_path, "SKY")
    else: checkpoint_path = PRETRAINED_DIR
    
    ckpt = tf.train.Checkpoint(
                            epoch = tf.Variable(0),
                            gen_model=_gen,
                            dis_model=_dis,
                            gen_optimizer=optimizer_gen,
                            disc_optimizer=optimizer_disc)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    #  if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest SKY checkpoint has restored!!')

    """Load Pretrained Sun Model"""
    if SUN_PRETRAINED_DIR is not None:
        optimizer_sun = tf.keras.optimizers.Adam(LEARNING_RATE)
        ckpt_sun, ckpt_manager_sun = tf_utils.checkpoint_initialization(
                                        model_name="SUN",
                                        pretrained_dir=SUN_PRETRAINED_DIR,
                                        checkpoint_path=checkpoint_path,
                                        model=_sun,
                                        optimizer=optimizer_sun)
    
    kl_Divergence = tf.keras.losses.KLDivergence()
  
    # LSGAN
    gen_loss = lambda disc_generated_output : tf.reduce_mean(tf.square(disc_generated_output - 1.))
    real_loss = lambda disc_real_output : tf.reduce_mean(tf.square(disc_real_output - 1.))
    generated_loss = lambda disc_generated_output : tf.reduce_mean(tf.square(disc_generated_output - 0.))
    
    @tf.function
    def generator_in_step(args, training="training"):
        if training:
            jpeg_img_float, hdr_t, sunpose_gt, gen_tape = args
        else:
            jpeg_img_float, hdr_t, sunpose_gt           = args
        
        hdr_t_gamma = tf_utils.hdr_logCompression(hdr_t)
        thr = 0.12

        res_out = _gen.encode(jpeg_img_float, training=training)
        sky_pred_gamma = _gen.sky_decode(res_out, jpeg_img_float, training=training)
        sky_pred_lin = tf_utils.hdr_logDecompression(sky_pred_gamma)

        sunpose_cmf, sunpose_actvMaps= _sun.sunposeEstimation(jpeg_img_float, training=training)
        sunpose_pred = tf.reshape(sunpose_cmf, (-1, IMSHAPE[0], IMSHAPE[1], 1))
        
        if training:
            with gen_tape.stop_recording():
                alpha = tf.reduce_max(sky_pred_lin, axis=[3])
                alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
                alpha_c1 = tf.reshape(alpha, [-1, tf.shape(sky_pred_lin)[1], tf.shape(sky_pred_lin)[2], 1])
                alpha_c3 = tf.tile(alpha_c1, [1, 1, 1, 3])

                sunlayer1, sunlayer2, sunlayer3 = sunpose_actvMaps # [b, h, w, [64, 32, 16]]

                max_arg = tf.math.argmax(sunpose_gt, axis=1)
                max_arg = tf.expand_dims(max_arg, axis=-1)
                y_c     = tf.gather_nd(indices=max_arg, params=sunpose_cmf, batch_dims=1)
                
                sun_cam1 = grad_cam.layer(y_c, sunlayer1)
                sun_cam2 = grad_cam.layer(y_c, sunlayer2)
                sun_cam3 = grad_cam.layer(y_c, sunlayer3)
        else:
            alpha = tf.reduce_max(sky_pred_lin, axis=[3])
            alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
            alpha_c1 = tf.reshape(alpha, [-1, tf.shape(sky_pred_lin)[1], tf.shape(sky_pred_lin)[2], 1])
            alpha_c3 = tf.tile(alpha_c1, [1, 1, 1, 3])
            
            sunlayer1, sunlayer2, sunlayer3 = sunpose_actvMaps # [b, h, w, [64, 32, 16]]

            max_arg = tf.math.argmax(sunpose_gt, axis=1)
            max_arg = tf.expand_dims(max_arg, axis=-1)
            y_c     = tf.gather_nd(indices=max_arg, params=sunpose_cmf, batch_dims=1)
            
            sun_cam1 = grad_cam.layer(y_c, sunlayer1)
            sun_cam2 = grad_cam.layer(y_c, sunlayer2)
            sun_cam3 = grad_cam.layer(y_c, sunlayer3)

        sun_rad_lin, gamma, beta = _gen.sun_rad_estimation(jpeg_img_float, sun_cam1, sun_cam2, sun_cam3, sunpose_pred, training= training)
        sun_rad_gamma = tf_utils.hdr_logCompression(sun_rad_lin)
        sun_pred_gamma = _gen.sun_decode(res_out, sun_cam1, sun_cam2, sun_cam3, sun_rad_gamma, training= training)
        
        # Rescaled sky_pred with alpha blending
        sky_pred_gamma = (1.- alpha_c3) * sky_pred_gamma
        sky_pred_lin = tf_utils.hdr_logDecompression(sky_pred_gamma)

        sun_pred_gamma = alpha_c3 * sun_pred_gamma
        sun_pred_lin = tf_utils.hdr_logDecompression(sun_pred_gamma)
        y_final_gamma = _gen.blending(sky_pred_gamma, sun_pred_gamma, training = training)
        y_final_lin = tf_utils.hdr_logDecompression(y_final_gamma)
        
        """Discriminator (linear comparison)"""
        disc_generated_output = _dis([jpeg_img_float, y_final_lin], training=False)

        """Sun-pose loss"""
        sun_loss = kl_Divergence(sunpose_gt, sunpose_cmf)
        
        """Perceptual loss (gamma comparison)"""
        vgg_pool1, vgg_pool2, vgg_pool3 = vgg(y_final_gamma)
        vgg2_pool1, vgg2_pool2, vgg2_pool3 = vgg2(hdr_t_gamma)

        perceptual_loss  = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)))
        
        """Difference of Gaussian(DoG) loss (linear comparison)"""
        dog_image1, dog_image2, dog_image3, dog_image4 = tf_utils.DoG(y_final_lin)
        dog2_image1, dog2_image2, dog2_image3, dog2_image4 = tf_utils.DoG(hdr_t)

        DoG_loss  = tf.reduce_mean(tf.abs((dog_image1 - dog2_image1)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image2 - dog2_image2)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image3 - dog2_image3)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image4 - dog2_image4)))

        """L1 loss (linear comparison)"""
        l1_loss = tf.reduce_mean(tf.abs(y_final_lin - hdr_t))
        
        """Adversarial loss (gamma comparison)"""
        _gen_loss = gen_loss(disc_generated_output)

        """Total loss"""
        total_gen_loss = tf.reduce_mean(sun_loss + 1000. * DoG_loss + _gen_loss + 10. * l1_loss + 0.01 * perceptual_loss)

        if training:
            train_total_loss_gen(total_gen_loss)
            train_l1_loss(l1_loss)
            train_kl_divergence(sun_loss)
            train_DoG_loss(DoG_loss)
            train_gen_loss(_gen_loss)
            train_perceptual_loss(perceptual_loss)

        else:
            test_total_loss_gen(total_gen_loss)
            test_l1_loss(l1_loss)
            test_kl_divergence(sun_loss)
            test_DoG_loss(DoG_loss)
            test_gen_loss(_gen_loss)
            test_perceptual_loss(perceptual_loss)

        return [total_gen_loss, y_final_gamma, sky_pred_lin, sun_pred_lin, gamma, beta, alpha_c3, sunpose_pred, sun_cam1, sun_cam2, sun_cam3, sun_rad_lin]
    
    @tf.function
    def discriminator_in_step(args, training="training"):
        # jpeg-float (0 ~ 1)
        if training:
            jpeg_img_float, hdr_t, y_final_lin, disc_tape = args
        else:
            jpeg_img_float, hdr_t, y_final_lin            = args

        """Discriminator (Linear Comparison)"""
        disc_real_output = _dis([jpeg_img_float, hdr_t], training=training) #return [b, h, w, 1]
        disc_generated_output = _dis([jpeg_img_float, y_final_lin], training=training) #return [b, h, w, 1]

        """Adversarial loss (Linear Comparison)"""
        _real_loss = real_loss(disc_real_output)
        _generated_loss = generated_loss(disc_generated_output)
        
        """Total loss"""
        # LSGAN
        total_disc_loss = tf.reduce_mean((_generated_loss + _real_loss)*0.5)
        
        if training:
            train_total_loss_disc(total_disc_loss)
            train_generated_loss(_generated_loss)
            train_real_loss(_real_loss)
        else:
            test_total_loss_disc(total_disc_loss)
            test_generated_loss(_generated_loss)
            test_real_loss(_real_loss)

        return total_disc_loss

    @tf.function
    def train_step(ds, sunpose_gt):
        # jpeg-float (0~1)
        hdr_t, jpeg_img_float = ds
        hdr_t = tf_utils.rgb2bgr(hdr_t)
        jpeg_img_float = tf_utils.rgb2bgr(jpeg_img_float)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_args = [jpeg_img_float, hdr_t, sunpose_gt, gen_tape]
            gen_pred = generator_in_step(gen_args, training=True)
            y_final_gamma = gen_pred[1]
            y_final_lin = tf_utils.hdr_logDecompression(y_final_gamma)
            
            # TODO Disc
            disc_args = [jpeg_img_float, hdr_t, y_final_lin, disc_tape]
            total_disc_loss = discriminator_in_step(disc_args, training=True)

        total_gen_loss, y_final_gamma, sky_pred_lin, sun_pred_lin, gamma, beta, alpha_c3, \
            sunpose_pred, sun_cam1, sun_cam2, sun_cam3, sun_rad_lin = gen_pred

        gradients_gen = gen_tape.gradient(total_gen_loss, _gen.trainable_variables+_sun.trainable_variables)
        optimizer_gen.apply_gradients(zip(gradients_gen, _gen.trainable_variables+_sun.trainable_variables))
        
        gradients_disc = disc_tape.gradient(total_disc_loss, _dis.trainable_variables)
        optimizer_disc.apply_gradients(zip(gradients_disc, _dis.trainable_variables))

        y_final_lin = tf_utils.bgr2rgb(y_final_lin)
        sky_pred_lin = tf_utils.bgr2rgb(sky_pred_lin)
        sun_pred_lin = tf_utils.bgr2rgb(sun_pred_lin)
        
        g_out = tf.reduce_max(gamma)
        b_out = tf.reduce_max(beta)

        return [y_final_lin, sky_pred_lin, sun_pred_lin, alpha_c3, sunpose_pred, sun_cam1, sun_cam2, sun_cam3, sun_rad_lin, g_out, b_out]

    @tf.function
    def test_step(ds, sunpose_gt):
        
        hdr_t, jpeg_img_float = ds
        hdr_t = tf_utils.rgb2bgr(hdr_t)
        jpeg_img_float = tf_utils.rgb2bgr(jpeg_img_float)

        gen_args = [jpeg_img_float, hdr_t, sunpose_gt]
        gen_pred = generator_in_step(gen_args, training=False)
        y_final_gamma = gen_pred[1]
        y_final_lin = tf_utils.hdr_logDecompression(y_final_gamma)
        
        disc_args = [jpeg_img_float, hdr_t, y_final_lin]
        _ = discriminator_in_step(disc_args, training=False)

        _, y_final_gamma, sky_pred_lin, sun_pred_lin, gamma, beta, alpha_c3, \
            sunpose_pred, sun_cam1, sun_cam2, sun_cam3, sun_rad_lin = gen_pred
        
        y_final_lin = tf_utils.bgr2rgb(y_final_lin)
        sky_pred_lin = tf_utils.bgr2rgb(sky_pred_lin)
        sun_pred_lin = tf_utils.bgr2rgb(sun_pred_lin)

        g_out = tf.reduce_max(gamma)
        b_out = tf.reduce_max(beta)

        return [y_final_lin, sky_pred_lin, sun_pred_lin, alpha_c3, sunpose_pred, sun_cam1, sun_cam2, sun_cam3, sun_rad_lin, g_out, b_out]

    for epoch in range(1, EPOCHS+1):

        start = time.perf_counter()

        train_total_loss_gen.reset_states()
        train_total_loss_disc.reset_states()
        
        # Sub_train_loss
        train_l1_loss.reset_states()
        train_perceptual_loss.reset_states()
        train_DoG_loss.reset_states()
        train_gen_loss.reset_states()
        train_kl_divergence.reset_states()

        train_generated_loss.reset_states()
        train_real_loss.reset_states()

        # Sub_test_loss
        test_total_loss_gen.reset_states()
        test_total_loss_disc.reset_states()

        test_l1_loss.reset_states()
        test_perceptual_loss.reset_states()
        test_DoG_loss.reset_states()
        test_gen_loss.reset_states()
        test_kl_divergence.reset_states()

        test_generated_loss.reset_states()
        test_real_loss.reset_states()
        
        for (hdr, sunpose_gt) in tqdm(train_ds):
            preprocessed_dataset = tf.py_function(_preprocessing, [hdr, train_crf, train_t], [tf.float32, tf.float32]) 
            pred = train_step(preprocessed_dataset, sunpose_gt)
        
        with train_summary_writer.as_default():
            
            tf.summary.scalar('gen_total_loss', train_total_loss_gen.result(), step=epoch)
            tf.summary.scalar('gen_l1_loss', train_l1_loss.result(), step=epoch)
            tf.summary.scalar('gen_perceptual_loss', train_perceptual_loss.result(), step=epoch)
            tf.summary.scalar('gen_DoG_loss', train_DoG_loss.result(), step=epoch)
            tf.summary.scalar('gen_adv_loss', train_gen_loss.result(), step=epoch)
            tf.summary.scalar('gen_kl_div', train_kl_divergence.result(), step=epoch)

            tf.summary.scalar('disc_total_loss', train_total_loss_disc.result(), step=epoch)
            tf.summary.scalar('disc_generated_loss', train_generated_loss.result(), step=epoch)
            tf.summary.scalar('disc_real_loss', train_real_loss.result(), step=epoch)

        for (hdr, sunpose_gt) in tqdm(test_ds):
            preprocessed_dataset = tf.py_function(_preprocessing, [hdr, test_crf, test_t], [tf.float32, tf.float32])
            pred = test_step(preprocessed_dataset, sunpose_gt)

        with test_summary_writer.as_default():
            
            tf.summary.scalar('gen_total_loss', test_total_loss_gen.result(), step=epoch)
            tf.summary.scalar('gen_l1_loss', test_l1_loss.result(), step=epoch)
            tf.summary.scalar('gen_perceptual_loss', test_perceptual_loss.result(), step=epoch)
            tf.summary.scalar('gen_DoG_loss', test_DoG_loss.result(), step=epoch)
            tf.summary.scalar('gen_adv_loss', test_gen_loss.result(), step=epoch)
            tf.summary.scalar('gen_kl_div', test_kl_divergence.result(), step=epoch)

            tf.summary.scalar('disc_total_loss', test_total_loss_disc.result(), step=epoch)
            tf.summary.scalar('disc_generated_loss', test_generated_loss.result(), step=epoch)
            tf.summary.scalar('disc_real_loss', test_real_loss.result(), step=epoch)
            
        ckpt.epoch.assign_add(1)

        if SUN_PRETRAINED_DIR is not None:
            ckpt_sun.epoch.assign_add(1)

        tf.summary.scalar('g_out', pred[9], step=epoch)
        tf.summary.scalar('b_out', pred[10], step=epoch)
    
        if int(ckpt.epoch) % 10 == 0:            
            if SUN_PRETRAINED_DIR is not None:
                sapa = ckpt_manager_sun.save()
                print(f"Saved sun checkpoint for step {int(ckpt_sun.epoch)}: {sapa}")

            save_path =  ckpt_manager.save()
            print(f"Saved checkpoint for step {int(ckpt.epoch)}: {save_path}")

        print(f'Epoch: {int(ckpt.epoch)}, Train Gen Loss: {train_total_loss_gen.result()}, Train Disc Loss: {train_total_loss_disc.result()}, \
                Test Gen Loss: {test_total_loss_gen.result()}, Test Disc Loss: {test_total_loss_disc.result()}, Elapsed time : {time.perf_counter() - start} seconds')
        
if __name__=="__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(description="train a model")
    parser.add_argument('--dir', type=str, default=os.path.join(CURRENT_WORKINGDIR, "dataset_{}_{}/tfrecord".format(IMSHAPE[1], IMSHAPE[0])))
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--imheight', type=int, default=32)
    parser.add_argument('--imwidth', type=int, default=128)
    parser.add_argument('--sky', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/SKY"))
    parser.add_argument('--sun', type=str, default=os.path.join(CURRENT_WORKINGDIR, "checkpoints/SUN"))
    parser.add_argument('--dorf', type=str, default=os.path.join(CURRENT_WORKINGDIR, 'dorfCurves.txt'))
    parser.add_argument('--vgg', type=str, default=os.path.join(CURRENT_WORKINGDIR, 'vgg16.npy'))
    
    args = parser.parse_args()

    run(args)
