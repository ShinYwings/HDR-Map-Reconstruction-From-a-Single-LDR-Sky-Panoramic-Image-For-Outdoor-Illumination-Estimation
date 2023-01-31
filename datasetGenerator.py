from DataGeneration import loadLavalSkyDB as a
from DataGeneration import makeTFRecord as b

import time

DATASET_DIR = "/media/shin/2nd_m.2/LavalSkyDB"

# reshape_size = [128,32]

img_bias = 0.00955794

def make_dataset(reshape_size):
    start = time.perf_counter()

    print("run loadSUN360 on {}".format(reshape_size))
    a.loadLavalSkyDB(DATASET_DIR, reshape_size, img_bias)

    print("run makeTFRecord")
    b.makeTFRecord(reshape_size)

    print("Finish.")
    
    print("elapsed Time of dataset generation: {} secs".format(time.perf_counter() - start))

if __name__=="__main__":
    make_dataset([128, 32])