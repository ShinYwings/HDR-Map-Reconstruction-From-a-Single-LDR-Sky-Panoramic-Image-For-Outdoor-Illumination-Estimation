import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from os import environ
environ["OPENCV_IO_ENABLE_OPENEXR"] = "true"

"""
LavalSkyDB to resized hdr img
"""

# Must define the minimum value of the hdr image (issues of tone mapping and image showing)

def alignSunpose(img, sun_azimuth, imshape):
    # ori_img = img[...,::-1]
    img = cv2.resize(img, (imshape[0], int(imshape[1] * 2)))
    h,w,_ = img.shape
    img = img[:int(h/2),:,:]
    newimg = np.zeros_like(img)
    
    for i in range(w):

        new_loc = i- sun_azimuth

        if new_loc < 0:
            new_loc += imshape[0]

        if new_loc >= imshape[0]:
            new_loc -= imshape[0]
        
        newimg[:,new_loc,:] = img[:,i,:]

    return newimg

def _mkdir(dir):

    if not os.path.isdir(dir):
        os.mkdir(dir)

def loadLavalSkyDB(ROOT_DIR, reshape_size, img_bias):

    azimuth_unit = reshape_size[0]/360
    zenith_unit = reshape_size[1]/90

    CWD = os.getcwd()
    NEWDIR = os.path.join(CWD,"dataset_{}_{}".format(reshape_size[0], reshape_size[1]))
    TRAINDIR = os.path.join(NEWDIR, "train")
    TESTDIR = os.path.join(NEWDIR, "test")
    HDRDIR_TRAIN = os.path.join(TRAINDIR, "hdr")
    HDRDIR_TEST = os.path.join(TESTDIR, "hdr")

    envmap_dirname = os.path.join(ROOT_DIR, "envmap")
    csvday_dirname = os.path.join(ROOT_DIR, "csv_day")

    dirs = [NEWDIR, TRAINDIR, TESTDIR, HDRDIR_TRAIN, HDRDIR_TEST]

    for dir in dirs:
        _mkdir(dir)

    print(envmap_dirname)
    print(csvday_dirname)

    dir_list= os.listdir(envmap_dirname)
    dir_list.sort()

    idx = 30000

    new_df = pd.DataFrame(columns=["image_name","azimuth","elevation"])

    hdrdir = HDRDIR_TRAIN

    for date in dir_list:
        
        envmap_date_dirname = os.path.join(envmap_dirname,date)
        csvday_date_path = os.path.join(csvday_dirname,date)

        df = pd.read_csv(csvday_date_path)
        """
        20220418 shin
        Sun zenith angle is described as "Sun elevation" by the author, but it is actually "zenith angle" indeed.
        It has been modified as "zenith" in the latest metadata description.
        """
        df = df.dropna(subset=["Sun elevation","Sun azimuth"])
        
        timeline_list = os.listdir(envmap_date_dirname)
        timeline_list.sort()
        for timeline in timeline_list:
            
            envmap_timeline_dirname = os.path.join(envmap_date_dirname, timeline)

            img_path = os.path.join(envmap_timeline_dirname, "envmap.exr")
            
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            if np.max(img) < img_bias:
                print("Skip the invalid image (all dark)")
                continue
            if idx == 0:
                # Save the records in csv and Switch the img directory to test 
                df_path= os.path.join(TRAINDIR, "train_refine.csv")
                new_df.to_csv(df_path, columns=["image_name","azimuth","elevation"], index=False)

                hdrdir = HDRDIR_TEST
                new_df = pd.DataFrame(columns=["image_name","azimuth","elevation"])
            
            timeline_format = "{}-{}-{}_{}:{}:{}".format(date[:4], date[4:6], date[6:8], timeline[:2],timeline[2:4],timeline[4:6])
            location_format = "{}-{}-{} {}:{}:{}".format(date[:4], date[4:6], date[6:8], timeline[:2],timeline[2:4],timeline[4:6])
            
            desc = df.loc[df['Datetime'] == location_format]

            if desc.empty:
                continue
            
            sun_zenith = float(desc['Sun elevation'].values[0])
            sun_zenith = np.rad2deg(sun_zenith)

            sun_azimuth = float(desc['Sun azimuth'].values[0])
            sun_azimuth = np.rad2deg(sun_azimuth)

            sun_zenith = int(np.round(sun_zenith * zenith_unit))
            sun_azimuth = int(np.round(sun_azimuth * azimuth_unit))
            
            newimg = alignSunpose(img, sun_azimuth, reshape_size)

            # Save HDR images
            newhdrpath = os.path.join(hdrdir, timeline_format+".hdr")
            cv2.imwrite(newhdrpath, newimg)

            # Records img description in csv file
            parse_elevation = reshape_size[1] - sun_zenith   # zenith to elevation 
            parse_azimuth = sun_azimuth + int(reshape_size[1] * 2)
            new_df = new_df.append({"image_name": timeline_format,"azimuth": parse_azimuth,"elevation": parse_elevation}, ignore_index=True)

            idx-=1

            print("Save ",timeline_format, " \t idx : ", idx)

    # Save the testset records in csv
    df_path= os.path.join(TESTDIR, "test_refine.csv")
    new_df.to_csv(df_path, columns=["image_name","azimuth","elevation"], index=False)

    ##########################################