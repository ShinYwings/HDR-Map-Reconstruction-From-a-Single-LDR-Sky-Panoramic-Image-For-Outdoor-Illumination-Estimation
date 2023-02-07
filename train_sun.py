"""
Sun Pose Estimation Network (Pretrain module)
"""
import os
import tensorflow as tf
import time
from tqdm import tqdm
import argparse

import utils
import tf_utils

import sunpose_net as sun
import grad_cam

from numpy.random import randint

from utils import StrEnum
from enum import auto


AUTO = tf.data.AUTOTUNE
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr
IMSHAPE = (32,128,3)
AZIMUTH_gt = IMSHAPE[1]*0.5-1
SUNPOSE_BIN = tf.convert_to_tensor([tf_utils.sunpose_init(i,IMSHAPE[0],IMSHAPE[1]) for i in range(IMSHAPE[0]*IMSHAPE[1])])

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    except RuntimeError as e:
        print(e)

class mod(StrEnum):
    SUN = auto()

def vMF(x, y, h, w, kappa=80.0):
    # discrete the sun into (h*w) bins and model the sun probability distirbution. (von Mises-Fisher)
    
    sp_vec = tf_utils.sphere2world((x, y), h, w, skydome=True)
    sp_vec = tf.expand_dims(sp_vec, axis=0)
    sp_vec = tf.tile(sp_vec, (h*w, 1))
    
    batch_dot = tf.einsum("bc, bc-> b", SUNPOSE_BIN, sp_vec)
    batch_dot = tf.scalar_mul(kappa, batch_dot)
    pdf = tf.math.exp(batch_dot)
    
    pdf = pdf / tf.reduce_sum(pdf)
    return pdf
   
def _preprocessing(hdr, crf_src, t_src):

    b, _, _, _, = tf_utils.get_tensor_shape(hdr)

    crf_list = [crf_src[randint(0,len(crf_src))] for _ in range(b)]
    t = [t_src[randint(0,len(t_src))] for _ in range(b)]
    
    crf = tf.convert_to_tensor(crf_list)
    t = tf.convert_to_tensor(t)

    _hdr_t = hdr * tf.reshape(t, [b, 1, 1, 1])

    # Augment Poisson and Gaussian noise
    sigma_s = 0.08 / 6 * tf.random.uniform([b, 1, 1, 3], minval=0.0, maxval=1.0,
                                                     dtype=tf.float32, seed=1)
    sigma_c = 0.005 * tf.random.uniform([b, 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
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

    ## SingleHDR way
    # loss mask to exclude over-/under-exposed regions
    # gray = tf.image.rgb_to_grayscale(jpeg_img)
    # over_exposed = tf.cast(tf.greater_equal(gray, 249), tf.float32)
    # over_exposed = tf.reduce_sum(over_exposed, axis=[1, 2], keepdims=True)
    # over_exposed = tf.greater(over_exposed, 256.0 * 256.0 * 0.5)
    # under_exposed = tf.cast(tf.less_equal(gray, 6), tf.float32)
    # under_exposed = tf.reduce_sum(under_exposed, axis=[1, 2], keepdims=True)
    # under_exposed = tf.greater(under_exposed, 256.0 * 256.0 * 0.5)
    # extreme_cases = tf.logical_or(over_exposed, under_exposed)
    # loss_mask = tf.cast(tf.logical_not(extreme_cases), tf.float32)
    
    return [_hdr_t, jpeg_img_float]
    
def _parse_function(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'azimuth' : tf.io.FixedLenFeature([], tf.float32),
        'elevation' : tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)

    hdr = tf.io.decode_raw(example['image'], tf.float32)
    hdr = tf.reshape(hdr, IMSHAPE)
    hdr = hdr[:,:,::-1] # bgr2rgb due to tf.image.rgb2grayscale in preprocessing
    
    # DrTMO way
    hdr_mean = tf.reduce_mean(hdr)
    hdr = 0.5 * hdr / (hdr_mean + 1e-6) 

    azimuth = AZIMUTH_gt
    elevation = example['elevation'] # top == 0 to bottom
    
    sun_pose = vMF(azimuth ,elevation, IMSHAPE[0], IMSHAPE[1])

    return hdr, sun_pose

def configureDataset(dirpath, batchsize, train= "train"):

    tfrecords_list = list()
    a = tf.data.Dataset.list_files(os.path.join(dirpath, "*.tfrecord"), shuffle=False)
    tfrecords_list.extend(a)

    ds = tf.data.TFRecordDataset(filenames=tfrecords_list, num_parallel_reads=AUTO, compression_type="GZIP")
    ds = ds.map(_parse_function, num_parallel_calls=AUTO)

    if train:
        ds = ds.shuffle(buffer_size = 5000).batch(batch_size=batchsize, drop_remainder=True).prefetch(AUTO)
    else:
        ds = ds.batch(batch_size=batchsize, drop_remainder=True).prefetch(AUTO)
        
    return ds

def run(args):
    
    global IMSHAPE, AZIMUTH_gt, SUNPOSE_BIN
    
    # Hyper parameters
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batchsize

    EPOCHS = args.epochs
    IMSHAPE = (args.imheight,args.imwidth,3)
    TRAIN_SUN = args.train

    # Azimuth angle depends on dataset (fixed)
    AZIMUTH_gt = IMSHAPE[1]*0.5-1
    
    if args.dir == "None":
        DATASET_DIR = os.path.join(CURRENT_WORKINGDIR, "dataset_{}_{}/tfrecord".format(IMSHAPE[1], IMSHAPE[0]))
    else:
        DATASET_DIR = args.dir
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    DoRF_PATH = args.dorfpath

    # Absolute path
    SUN_PRETRAINED_DIR = os.path.join(CURRENT_WORKINGDIR, "checkpoints/SUN")

    SUNPOSE_BIN = tf.convert_to_tensor([tf_utils.sunpose_init(i,IMSHAPE[0],IMSHAPE[1]) for i in range(IMSHAPE[0]*IMSHAPE[1])])
    
    """Init Dataset"""
    train_ds = configureDataset(TRAIN_DIR, BATCH_SIZE, train=True)
    test_ds  = configureDataset(TEST_DIR, BATCH_SIZE, train=False)

    train_crf, test_crf = utils.getDoRF(DoRF_PATH)
    train_t  , test_t   = utils.get_T()

    """CheckPoint Create"""
    checkpoint_path = utils.createNewDir(CURRENT_WORKINGDIR, "checkpoints")
    _sun = sun.model(im_height=IMSHAPE[0], im_width=IMSHAPE[1], da_kernel_size=3, dilation_rate=1)

    """"Create Output Image Directory"""
    train_summary_writer_sun, test_summary_writer_sun, logdir_sun = tf_utils.createDirectories(CURRENT_WORKINGDIR, name=mod.SUN, dir="tensorboard")
    print('tensorboard --logdir={}'.format(logdir_sun))
    train_outImgDir_sun, test_outImgDir_sun = tf_utils.createDirectories(CURRENT_WORKINGDIR, name=mod.SUN, dir="outputImg")

    """Model initialization"""
    optimizer_sun, train_loss_sun, test_loss_sun = tf_utils.metric_initialization(mod.SUN, LEARNING_RATE) 

    ckpt_sun, ckpt_manager_sun = tf_utils.checkpoint_initialization(
                                    model_name=mod.SUN,
                                    pretrained_dir=SUN_PRETRAINED_DIR,
                                    checkpoint_path=checkpoint_path,
                                    model=_sun,
                                    optimizer=optimizer_sun)

    sun_crit = tf.keras.losses.KLDivergence()

    @tf.function
    def inference(jpeg_img_float):
        # jpeg-float (0~1)
        input_jpeg_img_float = tf_utils.rgb2bgr(jpeg_img_float)
        sm, Aks= _sun.sunposeEstimation(input_jpeg_img_float, training= False)

        sunlayer1, sunlayer2, sunlayer3 = Aks # [b, h, w, [64, 32, 16]]

        y_c = tf.math.reduce_max(sm, axis=1)
        
        sun_cam1 = grad_cam.layer(y_c, sunlayer1)
        sun_cam2 = grad_cam.layer(y_c, sunlayer2)
        sun_cam3 = grad_cam.layer(y_c, sunlayer3)
        
        pred = tf.reshape(sm, (-1, IMSHAPE[0], IMSHAPE[1], 1))

        return pred, [sun_cam1, sun_cam2, sun_cam3]

    @tf.function
    def sun_train_step(ds, sunpose_gt):
        # jpeg-float (0~1)
        _, jpeg_img_float= ds
        
        input_jpeg_img_float = tf_utils.rgb2bgr(jpeg_img_float)

        with tf.GradientTape() as sun_tape:

            sm, Aks= _sun.sunposeEstimation(input_jpeg_img_float, training= True)

            with sun_tape.stop_recording():
                sunlayer1, sunlayer2, sunlayer3 = Aks # [b, h, w, [64, 32, 16]]
                
                max_arg = tf.math.argmax(sunpose_gt, axis=1)
                max_arg = tf.expand_dims(max_arg, axis=-1)
                y_c     = tf.gather_nd(indices=max_arg, params=sm, batch_dims=1)

                sun_cam1 = grad_cam.layer(y_c, sunlayer1)
                sun_cam2 = grad_cam.layer(y_c, sunlayer2)
                sun_cam3 = grad_cam.layer(y_c, sunlayer3)
            
            # training sun_rad net is not available cuz it depends on alpha_c3 
            # sun_rad, gamma, beta = _sun.sun_rad_estimation(sun_cam1, sun_cam2, sun_cam3, sunpose_pred, training= True)

            kl_loss = sun_crit(sunpose_gt, sm)
            
            pred = tf.reshape(sm, (-1, IMSHAPE[0], IMSHAPE[1], 1))
            sungt = tf.reshape(sunpose_gt, (-1, IMSHAPE[0], IMSHAPE[1], 1))

            dog_image1, dog_image2, dog_image3, dog_image4 = tf_utils.DoG(pred)
            dog2_image1, dog2_image2, dog2_image3, dog2_image4 = tf_utils.DoG(sungt)

            DoG_loss  = tf.reduce_mean(tf.abs((dog_image1 - dog2_image1)))
            DoG_loss += tf.reduce_mean(tf.abs((dog_image2 - dog2_image2)))
            DoG_loss += tf.reduce_mean(tf.abs((dog_image3 - dog2_image3)))
            DoG_loss += tf.reduce_mean(tf.abs((dog_image4 - dog2_image4)))
            
            sun_loss = tf.reduce_mean(kl_loss + DoG_loss)
            
        gradients_sun = sun_tape.gradient(sun_loss, _sun.trainable_variables)
        optimizer_sun.apply_gradients(zip(gradients_sun, _sun.trainable_variables))
        train_loss_sun(sun_loss)

        return pred, sungt, [sun_cam1, sun_cam2, sun_cam3]

    @tf.function
    def sun_test_step(ds, sunpose_gt):
        # jpeg-float (0~1)
        _, jpeg_img_float = ds
        
        input_jpeg_img_float = tf_utils.rgb2bgr(jpeg_img_float)

        sm, Aks= _sun.sunposeEstimation(input_jpeg_img_float, training= False)

        sunlayer1, sunlayer2, sunlayer3 = Aks # [b, h, w, [64, 32, 16]]
        
        max_arg = tf.math.argmax(sunpose_gt, axis=1)
        max_arg = tf.expand_dims(max_arg, axis=-1)
        y_c     = tf.gather_nd(indices=max_arg, params=sm, batch_dims=1)

        sun_cam1 = grad_cam.layer(y_c, sunlayer1)
        sun_cam2 = grad_cam.layer(y_c, sunlayer2)
        sun_cam3 = grad_cam.layer(y_c, sunlayer3)
        
        kl_loss = sun_crit(sunpose_gt, sm)
        
        pred = tf.reshape(sm, (-1, IMSHAPE[0], IMSHAPE[1], 1))
        sungt = tf.reshape(sunpose_gt, (-1, IMSHAPE[0], IMSHAPE[1], 1))

        dog_image1, dog_image2, dog_image3, dog_image4 = tf_utils.DoG(pred)
        dog2_image1, dog2_image2, dog2_image3, dog2_image4 = tf_utils.DoG(sungt)

        DoG_loss  = tf.reduce_mean(tf.abs((dog_image1 - dog2_image1)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image2 - dog2_image2)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image3 - dog2_image3)))
        DoG_loss += tf.reduce_mean(tf.abs((dog_image4 - dog2_image4)))
        
        sun_loss = tf.reduce_mean(kl_loss + DoG_loss)

        test_loss_sun(sun_loss)

        return pred, sungt, [sun_cam1, sun_cam2, sun_cam3]
    
    def train(module="module",
                train_step="train_step", test_step="test_step",
                train_loss="train_loss", test_loss="test_loss",
                train_summary_writer = "train_summary_writer",
                test_summary_writer = "test_summary_writer",
                test_outImgDir = "test_outImgDir",
                ckpt = "ckpt",
                ckpt_manager = "ckpt_manager"):

        isFirst = True

        for epoch in range(1, EPOCHS+1):

            start = time.perf_counter()

            train_loss.reset_states()
            test_loss.reset_states()

            for (hdr, sunpose) in tqdm(train_ds):

                preprocessed_dataset = tf.py_function(_preprocessing, [hdr, train_crf, train_t], [tf.float32, tf.float32])
                
                pred, sungt, sun_cam= train_step(preprocessed_dataset, sunpose)
                # pred = [pred, y_final, alpha, sun_cams]
            
            sun_cam1_dir = utils.createNewDir(train_outImgDir_sun, "sun_cam1")
            sun_cam2_dir = utils.createNewDir(train_outImgDir_sun, "sun_cam2")
            sun_cam3_dir = utils.createNewDir(train_outImgDir_sun, "sun_cam3")
            outimg_epoch_dir = utils.createNewDir(train_outImgDir_sun, "outImg")

            grad_cam.show(sun_cam[0], sun_cam1_dir, int(ckpt.epoch), show=False, save=True)
            grad_cam.show(sun_cam[1], sun_cam2_dir, int(ckpt.epoch), show=False, save=True)
            grad_cam.show(sun_cam[2], sun_cam3_dir, int(ckpt.epoch), show=False, save=True)
            grad_cam.show(pred, outimg_epoch_dir, int(ckpt.epoch), show=False, save=True)
            
            with train_summary_writer.as_default():
                
                tf.summary.scalar('loss', train_loss.result(), step=epoch)

            for (hdr, sunpose) in tqdm(test_ds):
                
                preprocessed_dataset = tf.py_function(_preprocessing, [hdr, test_crf, test_t], [tf.float32, tf.float32])
                
                pred, sungt, sun_cam = test_step(preprocessed_dataset, sunpose)

            with test_summary_writer.as_default():
                
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
            
            if isFirst:
                isFirst = False
                groundtruth_dir = utils.createNewDir(test_outImgDir, "groundTruth")
                if not os.listdir(groundtruth_dir):
                    for i in range(hdr.get_shape()[0]):
                        # rgb2bgr
                        utils.writeHDR(hdr[i,:,:,::-1].numpy(), "{}/{}_gt.{}".format(groundtruth_dir,i,HDR_EXTENSION), hdr.get_shape()[1:3])

            ckpt.epoch.assign_add(1)

            sun_cam1_dir = utils.createNewDir(test_outImgDir, "sun_cam1")
            sun_cam2_dir = utils.createNewDir(test_outImgDir, "sun_cam2")
            sun_cam3_dir = utils.createNewDir(test_outImgDir, "sun_cam3")
            outimg_epoch_dir = utils.createNewDir(test_outImgDir, "outImg")
            sungt_epoch_dir = utils.createNewDir(test_outImgDir, "sungt")

            grad_cam.show(sun_cam[0], sun_cam1_dir, int(ckpt.epoch), show=False, save=True)
            grad_cam.show(sun_cam[1], sun_cam2_dir, int(ckpt.epoch), show=False, save=True)
            grad_cam.show(sun_cam[2], sun_cam3_dir, int(ckpt.epoch), show=False, save=True)
            grad_cam.show(pred, outimg_epoch_dir, int(ckpt.epoch), show=False, save=True)
            grad_cam.show(sungt, sungt_epoch_dir, int(ckpt.epoch), show=False, save=True)     

            if int(ckpt_sun.epoch) % 10 == 0:
                
                save_path =  ckpt_manager.save()
                print(f"Saved checkpoint for step {int(ckpt.epoch)}: {save_path}")

            print(f'[{module}] Epoch: {int(ckpt.epoch)}, Train Loss: {train_loss.result()}, Test Loss: {test_loss.result()}, Elapsed time : {time.perf_counter() - start} seconds')
    
    print("Trian start")
    if TRAIN_SUN:
        train(module=mod.SUN,
                train_step=sun_train_step, test_step=sun_test_step,  
                train_loss=train_loss_sun, test_loss=test_loss_sun,
                train_summary_writer = train_summary_writer_sun,
                test_summary_writer = test_summary_writer_sun,
                test_outImgDir = test_outImgDir_sun,
                ckpt = ckpt_sun,
                ckpt_manager = ckpt_manager_sun)
    
    else:
        from glob import glob
        import cv2
        import matplotlib.pyplot as plt

        # TODO dir
        inference_img_dir = args.inference_img_dir
        hdr_imgs = [glob(os.path.join(inference_img_dir, '*.hdr'))]
        hdr_imgs = sorted(hdr_imgs)
        
        for _ , hdr_img_path in enumerate(hdr_imgs[0]):
            
            print(hdr_img_path)

            hdr_src = cv2.imread(hdr_img_path, -1)

            hdr = tf.expand_dims(hdr_src, axis=0)

            crf = test_crf[randint(0,len(test_crf))]
            t = test_t[randint(0,len(test_t))]
            
            crf = tf.convert_to_tensor(crf)
            t = tf.convert_to_tensor(t)

            _hdr_t = hdr * tf.reshape(t, [1, 1, 1, 1])

            # Augment Poisson and Gaussian noise
            sigma_s = 0.08 / 6 * tf.random.uniform([1, 1, 1, 3], minval=0.0, maxval=1.0,
                                                            dtype=tf.float32, seed=1)
            sigma_c = 0.005 * tf.random.uniform([1, 1, 1, 3], minval=0.0, maxval=1.0, dtype=tf.float32, seed=1)
            noise_s_map = sigma_s * _hdr_t
            noise_s = tf.random.normal(shape=tf.shape(_hdr_t), seed=1) * noise_s_map
            temp_x = _hdr_t + noise_s
            noise_c = sigma_c * tf.random.normal(shape=tf.shape(_hdr_t), seed=1)
            temp_x = temp_x + noise_c
            _hdr_t = tf.nn.relu(temp_x)

            # Dynamic range clipping
            clipped_hdr_t = tf.clip_by_value(_hdr_t, 0, 1)

            crf = tf.reshape(crf, [1, -1])
            ldr = tf_utils.apply_rf(clipped_hdr_t, crf)
            
            # Quantization and JPEG compression
            quantized_hdr = tf.round(ldr * 255.0)
            quantized_hdr_8bit = tf.cast(quantized_hdr, tf.uint8)
            jpeg_img = tf.image.adjust_jpeg_quality(quantized_hdr_8bit[0], 90)
            jpeg_img_float = tf.cast(jpeg_img, tf.float32) / 255.0
            jpeg_img_float = tf.expand_dims(jpeg_img_float, axis=0)
            pred, sun_cam = inference(jpeg_img_float)

            sun_cam2 = tf.image.resize(sun_cam[1], (IMSHAPE[0],IMSHAPE[1]))
            
            sum_pred = sun_cam[0] * sun_cam2 * pred
            sum_pred = sum_pred / (tf.reduce_max(sum_pred) + 1e-5)

            fig = plt.figure()

            ax = fig.add_subplot(6, 1, 1)
            ax.imshow(sun_cam[0][0])

            ax = fig.add_subplot(6, 1, 2)
            ax.imshow(sun_cam[1][0])

            ax = fig.add_subplot(6, 1, 3)
            ax.imshow(sun_cam[2][0])

            ax = fig.add_subplot(6, 1, 4)
            ax.imshow(pred[0])

            ax = fig.add_subplot(6, 1, 5)
            ax.imshow(sum_pred[0])

            ax = fig.add_subplot(6, 1, 6)
            ax.imshow(hdr_src)

            ax.set_title(hdr_img_path, fontsize=5)

            plt.show()

    print("trian end")
if __name__=="__main__":

    CURRENT_WORKINGDIR = os.getcwd()
    
    parser = argparse.ArgumentParser(description="pretraining sun luminance estimator")
    parser.add_argument('--dir', type=str, default="/media/shin/2nd_m.2/LavalSkyDB")
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--inference_img_dir', type=str, default=f"/home/shin/shinywings/research/challengingImages/{IMSHAPE[0]}_{IMSHAPE[1]}")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--imheight', type=int, default=32)
    parser.add_argument('--imwidth', type=int, default=128)
    parser.add_argument('--dorfpath', type=str, default=os.path.join(CURRENT_WORKINGDIR, 'dorfCurves.txt'))
    
    args = parser.parse_args()

    run(args)
