from DataGeneration import loadLavalSkyDB as a
from DataGeneration import makeTFRecord as b
import argparse
import time
import os

# DATASET_DIR = 

# reshape_size = [128,32]

# Image bias is the minimum pixel value of HDR images
# If not added, tonemapping operators (Reinhard, drago, ...) will produce synthetic LDR images that are not suitable for use in training.
img_bias = 0.00955794

def make_dataset(reshape_size, dataset_dir):
    start = time.perf_counter()

    print("run loadSUN360 on {}".format(reshape_size))
    a.loadLavalSkyDB(dataset_dir, reshape_size, img_bias)

    print("run makeTFRecord")
    b.makeTFRecord(reshape_size)

    print("Finish.")
    
    print("elapsed Time of dataset generation: {} secs".format(time.perf_counter() - start))

if __name__=="__main__":
    CURRENT_WORKINGDIR = os.getcwd()
    
    parser = argparse.ArgumentParser(description="dataset generation")
    parser.add_argument('--dir', type=str, default=os.path.join(CURRENT_WORKINGDIR))
    parser.add_argument('--imheight', type=int, default=32)
    parser.add_argument('--imwidth', type=int, default=128)
    
    args = parser.parse_args()
    make_dataset([args.imwidth, args.imheight], args.dir)