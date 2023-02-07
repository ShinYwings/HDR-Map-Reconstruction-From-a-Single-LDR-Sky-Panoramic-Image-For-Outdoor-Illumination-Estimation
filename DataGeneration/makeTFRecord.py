import sys
sys.path.append("..")

import os
import tensorflow as tf
import cv2
import pandas as pd

TFRECORD_FILE_NAME= lambda name : '{}.tfrecord'.format(name)
# IMAGE_ENCODING_QUALITY = 100  # default 95
TFRECORD_OPTION = tf.io.TFRecordOptions(compression_type="GZIP")

def _float_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _bytes_feature(values):
  """Returns a bytes_list from a string / byte."""
  if isinstance(values, type(tf.constant(0))):
    values = values.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def serialize_ds(image, azimuth, elevation):
  feature_description = {
    'image': _bytes_feature(image),
    'azimuth': _float_feature(azimuth),
    'elevation': _float_feature(elevation),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature_description))
  return example_proto.SerializeToString()

def convert_image_to_bytes(image="image"):
    # We are gonna make exr file(RGB format) so flip BGR

    # image = image[...,::-1]

    # LDR case
    # convert_image = tf.cast(image, tf.uint8)
    # img_raw = tf.io.encode_jpeg(convert_image, quality=IMAGE_ENCODING_QUALITY)
    
    # HDR case
    # print("type(image) : ", type(image))
    img_raw = image.tostring()
    
    return img_raw

def parse_to_tfrecord(df = "df", envmap_dirname = "envmap_dirname", tfrecord_dir= "tfrecord_dir"):

    for idx, row in df.iterrows():
            
        imgpath = os.path.join(envmap_dirname, "{}.hdr".format(row["image_name"]))
        original_image = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
        image_bytes = convert_image_to_bytes(original_image)
       
        filename = TFRECORD_FILE_NAME(row["image_name"])
        filepath = os.path.join(tfrecord_dir, filename)
        with tf.io.TFRecordWriter(filepath, TFRECORD_OPTION) as writer:
            print("index: ", idx, "  parsing ",row["image_name"])
            example = serialize_ds(image_bytes, row["azimuth"], row["elevation"])
            writer.write(example)
        writer.close()

def _mkdir(dir):
    if not os.path.isdir(dir):
          os.mkdir(dir)

def makeTFRecord(reshaped_size):

    cwd = os.getcwd()

    DS = os.path.join(cwd, "dataset_{}_{}".format(reshaped_size[0], reshaped_size[1]))

    TRAINDIR = os.path.join(DS, "train")
    TRAINHDRDIR = os.path.join(TRAINDIR, "hdr")
    # TRAINRENDERDIR = os.path.join(TRAINDIR, "render")
    TESTDIR = os.path.join(DS, "test")
    TESTHDRDIR = os.path.join(TESTDIR, "hdr")
    # TESTRENDERDIR = os.path.join(TESTDIR, "render")
    NEWDIR = os.path.join(DS,"tfrecord")

    TRAIN_TFREC_DIR = os.path.join(NEWDIR,"train")
    TEST_TFREC_DIR = os.path.join(NEWDIR,"test")

    _mkdir(NEWDIR)
    _mkdir(TRAIN_TFREC_DIR)
    _mkdir(TEST_TFREC_DIR)

    proc_list = ["train", "test"]

    for proc in proc_list:

        if "train" == proc:
            dirname = TRAINDIR
            envmap_dirname = TRAINHDRDIR
            # render_dirname = TRAINRENDERDIR
            tfrecord_dir = TRAIN_TFREC_DIR
        else:
            dirname = TESTDIR
            envmap_dirname = TESTHDRDIR
            # render_dirname = TESTRENDERDIR
            tfrecord_dir = TEST_TFREC_DIR

        csv_path = os.path.join(dirname, proc + "_refine.csv")
        df = pd.read_csv(csv_path)

        parse_to_tfrecord(df = df, envmap_dirname = envmap_dirname, tfrecord_dir= tfrecord_dir)