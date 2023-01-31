import os
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm

import utils
import tf_utils
import grad_cam

import sunpose_net as sun
import generator as gen
import discriminator as dis
import refinement_net as ref

from vgg16 import Vgg16

from numpy.random import randint

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

AUTO = tf.data.AUTOTUNE

# Hyper parameters
BATCH_SIZE = 32
EPOCHS = 1000
IMSHAPE = (32,128,3)
AZIMUTH_gt = 63
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr

# Optimizer Option
LEARNING_RATE = 1e-4

CURRENT_WORKINGDIR = os.getcwd()
DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, "tf_records/outdoor_tfrecords")

SKY_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/SKY")
SUN_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/Pretrained_SUN")
REF_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/REF")
    
def _parse_function(example_proto):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    feature_description = {
        'hdr': tf.io.FixedLenFeature([], tf.string),
        'ldr': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    ref_HDR = tf.io.decode_raw(example['hdr'], tf.float32)
    ref_LDR = tf.io.decode_raw(example['ldr'], tf.float32)
    ref_HDR = tf.reshape(ref_HDR, IMSHAPE)
    ref_LDR = tf.reshape(ref_LDR, IMSHAPE)

    ref_HDR = ref_HDR / (1e-6 + tf.reduce_mean(ref_HDR)) * 0.5
    ref_LDR = ref_LDR / 255.0

    return ref_LDR, ref_HDR

def configureDataset(dirpath):

    tfrecords_list = list()
    a = tf.data.Dataset.list_files(os.path.join(dirpath, "*.tfrecords"), shuffle=False)
    tfrecords_list.extend(a)
    
    ds = tf.data.TFRecordDataset(filenames=tfrecords_list, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.map(_parse_function, num_parallel_calls=AUTO)
    
    ds  = ds.shuffle(buffer_size = len(tfrecords_list)).batch(batch_size=BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

    return ds
        
if __name__=="__main__":
    
    ds  = configureDataset(DATASET_DIR)

    """CheckPoint Create"""
    checkpoint_path = utils.createNewDir(CURRENT_WORKINGDIR, "checkpoints")

    _gen  = gen.model(batch_size=BATCH_SIZE, im_height=IMSHAPE[0], im_width=IMSHAPE[1])
    _sun = sun.model()
    _dis = dis.model()
    _ref = ref.model()
    vgg = Vgg16('vgg16.npy')
    vgg2 = Vgg16('vgg16.npy')

    """"Create Output Image Directory"""
    train_summary_writer, test_summary_writer, logdir = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="SKY", dir="tensorboard")
    train_outImgDir, test_outImgDir = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="SKY", dir="outputImg")
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

    ckpt_sky = tf.train.Checkpoint(
                            epoch = tf.Variable(0),
                            gen_model=_gen,
                            dis_model=_dis,
                            gen_optimizer=optimizer_gen,
                            disc_optimizer=optimizer_disc)
    ckpt_manager_sky = tf.train.CheckpointManager(ckpt_sky, SKY_PRETRAINED_DIR, max_to_keep=5)

    if ckpt_manager_sky.latest_checkpoint:
        ckpt_sky.restore(ckpt_manager_sky.latest_checkpoint)
        print('Latest SKY checkpoint has restored!!')
    
    """Load Pretrained Sun Model"""
    optimizer_sun = tf.keras.optimizers.Adam(LEARNING_RATE)
    ckpt_sun, ckpt_manager_sun = tf_utils.checkpoint_initialization(
                                    model_name="SUN",
                                    pretrained_dir=SUN_PRETRAINED_DIR,
                                    checkpoint_path=checkpoint_path,
                                    model=_sun,
                                    optimizer=optimizer_sun)
    
    train_summary_writer_ref, test_summary_writer_ref, logdir_ref = tf_utils.createDirectories(CURRENT_WORKINGDIR, name="ref", dir="tensorboard")
    print(f'tensorboard --logdir={logdir_ref}')

    """Model initialization"""
    optimizer_ref, train_loss_ref, test_loss_ref = tf_utils.metric_initialization("ref", LEARNING_RATE) 

    ckpt_ref, ckpt_manager_ref = tf_utils.checkpoint_initialization(
                                    model_name="REF",
                                    pretrained_dir=REF_PRETRAINED_DIR,
                                    checkpoint_path=checkpoint_path,
                                    model=_ref,
                                    optimizer=optimizer_ref)

    kl_Divergence = tf.keras.losses.KLDivergence()

    # LSGAN
    gen_loss = lambda disc_generated_output : tf.reduce_mean(tf.square(disc_generated_output - 1.))
    real_loss = lambda disc_real_output : tf.reduce_mean(tf.square(disc_real_output - 1.))
    generated_loss = lambda disc_generated_output : tf.reduce_mean(tf.square(disc_generated_output - 0.))
    
    """
    Check out the dataset that properly work
    """
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20,20))
    # i = 0
    # for (ldr, hdr) in ds.take(15):
    #     print(tf.shape(hdr))
    #     ax = plt.subplot(2,15,i+1)
    #     plt.imshow(ldr[0])
    #     ax = plt.subplot(2,15,i+2)
    #     plt.imshow(hdr[0])
    #     plt.axis('off')
    #     i+=2
    # plt.show()

    @tf.function
    def generator_in_step(args, training="training"):
        if training:
            ldr, hdr, gen_tape = args
        else:
            ldr, hdr           = args

        hdr_gamma = tf_utils.hdr_logCompression(hdr)
        thr = 0.12

        res_out = _gen.encode(ldr, training=training)
        sky_pred_gamma = _gen.sky_decode(res_out, ldr, training=training)
        sky_pred_lin = tf_utils.hdr_logDecompression(sky_pred_gamma)
        
        sunpose_cmf, sunpose_actvMaps= _sun.sunposeEstimation(ldr, training=training)
        sunpose_pred = tf.reshape(sunpose_cmf, (-1, IMSHAPE[0], IMSHAPE[1], 1))
        
        if training:
            with gen_tape.stop_recording():
                alpha = tf.reduce_max(sky_pred_lin, axis=[3])
                alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
                alpha_c1 = tf.reshape(alpha, [-1, tf.shape(sky_pred_lin)[1], tf.shape(sky_pred_lin)[2], 1])
                alpha_c3 = tf.tile(alpha_c1, [1, 1, 1, 3])

                sunlayer1, sunlayer2, sunlayer3 = sunpose_actvMaps # [b, h, w, [64, 32, 16]]

                y_c = tf.math.reduce_max(sunpose_cmf, axis=1)

                sun_cam1 = grad_cam.layer(y_c, sunlayer1)
                sun_cam2 = grad_cam.layer(y_c, sunlayer2)
                sun_cam3 = grad_cam.layer(y_c, sunlayer3)
        else:
            alpha = tf.reduce_max(sky_pred_lin, axis=[3])
            alpha = tf.minimum(1.0, tf.maximum(0.0, alpha - 1.0 + thr) / thr)
            alpha_c1 = tf.reshape(alpha, [-1, tf.shape(sky_pred_lin)[1], tf.shape(sky_pred_lin)[2], 1])
            alpha_c3 = tf.tile(alpha_c1, [1, 1, 1, 3])

            sunlayer1, sunlayer2, sunlayer3 = sunpose_actvMaps # [b, h, w, [64, 32, 16]]

            y_c = tf.math.reduce_max(sunpose_cmf, axis=1)

            sun_cam1 = grad_cam.layer(y_c, sunlayer1)
            sun_cam2 = grad_cam.layer(y_c, sunlayer2)
            sun_cam3 = grad_cam.layer(y_c, sunlayer3)

        sun_rad_lin, gamma, beta = _gen.sun_rad_estimation(ldr, sun_cam1, sun_cam2, sun_cam3, sunpose_pred, training= training)
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
        disc_generated_output = _dis([ldr, y_final_lin], training=False)
        
        """Perceptual loss (gamma comparison)"""
        vgg_pool1, vgg_pool2, vgg_pool3 = vgg(y_final_gamma)
        vgg2_pool1, vgg2_pool2, vgg2_pool3 = vgg2(hdr_gamma)

        perceptual_loss  = tf.reduce_mean(tf.abs((vgg_pool1 - vgg2_pool1)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool2 - vgg2_pool2)))
        perceptual_loss += tf.reduce_mean(tf.abs((vgg_pool3 - vgg2_pool3)))
        
        """Difference of Gaussian(DoG) loss (linear comparison)"""
        dog_image1, dog_image2, dog_image3, dog_image4 = tf_utils.DoG(y_final_lin)
        dog2_image1, dog2_image2, dog2_image3, dog2_image4 = tf_utils.DoG(hdr)

        DoG_loss  = tf.reduce_mean(tf.abs((dog_image1 - dog2_image1)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image2 - dog2_image2)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image3 - dog2_image3)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image4 - dog2_image4)))

        """L1 loss (linear comparison)"""
        l1_loss = tf.reduce_mean(tf.abs(y_final_lin - hdr))
        
        """Adversarial loss (gamma comparison)"""
        _gen_loss = gen_loss(disc_generated_output)

        """Total loss"""
        total_gen_loss = tf.reduce_mean(1000. * DoG_loss + _gen_loss + 10. * l1_loss + 0.01 * perceptual_loss)
        
        if training:
            train_total_loss_gen(total_gen_loss)
            train_l1_loss(l1_loss)
            train_DoG_loss(DoG_loss)
            train_gen_loss(_gen_loss)
            train_perceptual_loss(perceptual_loss)

        else:
            test_total_loss_gen(total_gen_loss)
            test_l1_loss(l1_loss)
            test_DoG_loss(DoG_loss)
            test_gen_loss(_gen_loss)
            test_perceptual_loss(perceptual_loss)

        return total_gen_loss, y_final_lin

    @tf.function
    def discriminator_in_step(args, training="training"):
        if training:
            ldr, hdr, y_final_lin, disc_tape = args
        else:
            ldr, hdr, y_final_lin            = args

        """Discriminator (Linear Comparison)"""
        disc_real_output = _dis([ldr, hdr], training=training) #return [b, h, w, 1]
        disc_generated_output = _dis([ldr, y_final_lin], training=training) #return [b, h, w, 1]

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
    def train_step(ldr, hdr):
        
        ldr = tf_utils.rgb2bgr(ldr)
        hdr = tf_utils.rgb2bgr(hdr)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as ref_tape:
            
            gen_args = [ldr, hdr, gen_tape]

            total_gen_loss, y_final_lin = generator_in_step(gen_args, training=True)
            
            disc_args = [ldr, hdr, y_final_lin, disc_tape]

            total_disc_loss = discriminator_in_step(disc_args, training=True)

            # pred_hdr = _ref(y_final_lin, training = True)

            # refloss = tf.reduce_mean(tf.abs(pred_hdr - hdr))
        
        gradients_gen = gen_tape.gradient(total_gen_loss, _gen.trainable_variables)
        optimizer_gen.apply_gradients(zip(gradients_gen,_gen.trainable_variables))
        gradients_disc = disc_tape.gradient(total_disc_loss, _dis.trainable_variables)
        optimizer_disc.apply_gradients(zip(gradients_disc, _dis.trainable_variables))
        # gradients_ref = ref_tape.gradient(refloss, _ref.trainable_variables)
        # optimizer_ref.apply_gradients(zip(gradients_ref, _ref.trainable_variables))

        # train_loss_ref(refloss)
        
        y_final_lin = tf_utils.bgr2rgb(y_final_lin)
        # pred_hdr = tf_utils.bgr2rgb(pred_hdr)

        # return pred_hdr, y_final_lin
        return y_final_lin

    @tf.function
    def test_step(gt):
        # NO USED, NO TYPED
        pass

    ckpts = [ckpt_sky, ckpt_sun, ckpt_ref]
    ckpt_managers = [ckpt_manager_sky, ckpt_manager_sun, ckpt_manager_ref]

    print("시작")
    
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

        train_loss_ref.reset_states()

        for (ldr, hdr) in tqdm(ds):
            
            y_final_lin = train_step(ldr, hdr)

        with train_summary_writer_ref.as_default():
            
            tf.summary.scalar('gen_total_loss', train_total_loss_gen.result(), step=epoch)
            tf.summary.scalar('gen_l1_loss', train_l1_loss.result(), step=epoch)
            tf.summary.scalar('gen_perceptual_loss', train_perceptual_loss.result(), step=epoch)
            tf.summary.scalar('gen_DoG_loss', train_DoG_loss.result(), step=epoch)
            tf.summary.scalar('gen_adv_loss', train_gen_loss.result(), step=epoch)
            tf.summary.scalar('gen_kl_div', train_kl_divergence.result(), step=epoch)

            tf.summary.scalar('disc_total_loss', train_total_loss_disc.result(), step=epoch)
            tf.summary.scalar('disc_generated_loss', train_generated_loss.result(), step=epoch)
            tf.summary.scalar('disc_real_loss', train_real_loss.result(), step=epoch)
            
            tf.summary.image('y_final_lin', y_final_lin, step=epoch)
            # tf.summary.image('pred_hdr', pred_hdr, step=epoch)

            tf.summary.image('hdr', hdr, step=epoch)
            tf.summary.image('ldr', ldr, step=epoch)
            
        print(f'IN ref, epoch: {epoch}, Train Loss: {train_loss_ref.result()}')

        print(f"Spends time : {time.perf_counter() - start} seconds in Epoch number {epoch}")
        
        for ckpt in ckpts:
            ckpt.epoch.assign_add(1)
        
        for ckpt_manager in ckpt_managers:
            save_path =  ckpt_manager.save()
            print(f"Saved checkpoint for step {epoch}: {save_path}")

    print("끝")