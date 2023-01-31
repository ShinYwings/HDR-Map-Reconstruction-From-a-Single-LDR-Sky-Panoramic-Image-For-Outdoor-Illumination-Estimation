import os
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
import cv2

from train import LEARNING_RATE
import utils
import tf_utils
import grad_cam

import sunpose_net as sun
import generator as gen
import discriminator as dis

from vgg16 import Vgg16

from numpy.random import randint

import glob

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

AUTO = tf.data.AUTOTUNE

"""
BGR input but RGB conversion in dataset.py (due to tf.image.rgb_to_grayscale and other layers)
"""
# Hyper parameters
THRESHOLD = 0.12
IMSHAPE = (32,128,3)
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr
CURRENT_WORKINGDIR = os.getcwd()
input_dirname = f"{INPUT_DATASET_PATH}"
output_dirname = f"{OUTPUT_DATASET_PATH}"
DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, input_dirname)

SKY_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, f"checkpoints/SKY")
SUN_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, f"checkpoints/SUN")

if __name__=="__main__":

    _gen  = gen.model(batch_size=1, im_height=IMSHAPE[0], im_width=IMSHAPE[1])
    _sun = sun.model()
    _dis = dis.model()
    vgg = Vgg16('vgg16.npy')
    vgg2 = Vgg16('vgg16.npy')

    """Model initialization"""
    # TODO dataset path setting
    outputDir = utils.createNewDir(CURRENT_WORKINGDIR, output_dirname)
    
    checkpoint_path = utils.createNewDir(CURRENT_WORKINGDIR, "checkpoints")

    optimizer_ref = tf.keras.optimizers.Adam(LEARNING_RATE)

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

    @tf.function
    def generator_in_step(ldr, training= False):
        
        thr = 0.12

        res_out = _gen.encode(ldr, training=training)
        sky_pred_gamma = _gen.sky_decode(res_out, ldr, training=training)
        sky_pred_lin = tf_utils.hdr_logDecompression(sky_pred_gamma)
        
        sunpose_cmf, sunpose_actvMaps= _sun.sunposeEstimation(ldr, training=training)
        sunpose_pred = tf.reshape(sunpose_cmf, (-1, IMSHAPE[0], IMSHAPE[1], 1))
        
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

        sun_cam = [sun_cam1, sun_cam2, sun_cam3]
        return [y_final_lin, sunpose_pred, alpha_c3, sun_cam, sun_rad_lin, sun_pred_lin, sky_pred_lin]

    @tf.function
    def inference(ldr):
        
        pred = generator_in_step(ldr, training= False)
        
        return pred
    
    ldr_imgs = glob.glob(os.path.join(DATASET_DIR, '*.jpg'))
    ldr_imgs = sorted(ldr_imgs)

    hdr_imgs = glob.glob(os.path.join(DATASET_DIR, '*.hdr'))
    hdr_imgs = sorted(hdr_imgs)

    for ldr_img_path, hdr_img_path in zip(ldr_imgs, hdr_imgs):

        start = time.perf_counter()
        
        # input bgr
        ldr_img = cv2.imread(ldr_img_path)
        h, w, _ = ldr_img.shape

        hdr_img = cv2.imread(hdr_img_path, -1)
        # hdr_img = hdr_img[:int(h/2),:,:]
        # hdr_img = hdr_img / np.max(hdr_img)
        
        # ldr_img = ldr_img[:int(h/2),:,:]
        ldr_val = ldr_img / 255.0

        ldr_val = tf.convert_to_tensor(ldr_val, dtype=tf.float32)
        ldr_val = tf.expand_dims(ldr_val, axis=0)

        # ldr_val =  rgb
        pred, pred_hdr = inference(ldr_val)
        HDR_out_val, sunpose_pred, alpha_c3, sun_cam, sun_rad_lin, sun_pred_lin, sky_pred_lin = pred
        sun_cam1, sun_cam2, sun_cam3 = sun_cam
        
        imgname = os.path.split(ldr_img_path)[-1]
        imgname = str.split(imgname, sep=".")[0]
        outputfile_path= imgname +'.hdr'
        gt_path= imgname +'_sunrad.hdr'
        cv2.imwrite(os.path.join(outputDir, outputfile_path), pred_hdr[0].numpy()) # rgb output [:,:,::-1]
        cv2.imwrite(os.path.join(outputDir, gt_path), sun_rad_lin[0].numpy()) # rgb output [:,:,::-1]
        
        print(f"Spends time : {time.perf_counter() - start} seconds")