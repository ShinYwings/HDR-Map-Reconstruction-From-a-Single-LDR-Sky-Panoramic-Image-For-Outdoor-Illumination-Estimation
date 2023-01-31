import tensorflow as tf
import numpy as np
import os
import glob
import cv2

TFRECORD_OPTION = tf.io.TFRecordOptions(compression_type="GZIP")

def _bytes_feature(values):
  """Returns a bytes_list from a string / byte."""
  if isinstance(values, type(tf.constant(0))):
    values = values.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def serialize_ds(ldr, hdr):
  feature_description = {
    'ldr': _bytes_feature(ldr),
    'hdr': _bytes_feature(hdr),
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature_description))
  return example_proto.SerializeToString()

def convert_image_to_bytes(image="image"):
    img_raw = image.tostring()
    
    return img_raw

TFRECORD_FILE_NAME = lambda x : os.path.join(out_dir, f"{x}.tfrecords")

if __name__=="__main__":

    out_dir = 'tf_records/outdoor_tfrecords'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    HDRs = sorted(glob.glob('/media/shin/2nd_m.2/HDR_test_dataset/outdoor_real_gt/*.exr'))
    LDRs = sorted(glob.glob('/media/shin/2nd_m.2/HDR_test_dataset/outdoor_real_input/*.jpg'))

    for i, scene_dir in enumerate(HDRs):
        
        # read images
        # two of them are BGR format
        ref_HDR = cv2.imread(HDRs[i], -1).astype(np.float32)  # read raw values
        ref_LDR = cv2.imread(LDRs[i]).astype(np.float32)   # read jpg

        h, w, c = ref_HDR.shape

        ref_HDR = ref_HDR[:int(h/2),:,:]
        ref_LDR = ref_LDR[:int(h/2),:,:]

        ref_HDR_bytes = convert_image_to_bytes(ref_HDR)
        ref_LDR_bytes = convert_image_to_bytes(ref_LDR)

        imgpath = os.path.split(scene_dir)[-1]
        imgname = str.split(imgpath, sep=".")[0]
        filepath = TFRECORD_FILE_NAME(imgname)
        with tf.io.TFRecordWriter(filepath, TFRECORD_OPTION) as writer:
            print("index: ", i, "  parsing ",imgname)
            example = serialize_ds(ref_LDR_bytes, ref_HDR_bytes)
            writer.write(example)
        writer.close()