# HDR Map Reconstruction From a Single LDR Sky Panoramic Image For Outdoor Illumination Estimation \[[Paper](https://ieeexplore.ieee.org/document/10045650)]

![result1](figure/rendered.png)
A multi-faceted approach to reconstructing HDR maps from a single LDR sky panoramic image that considers the sun and sky regions separately and accounts for various weather conditions.

(This paper has been published on IEEE Access)
</br></br>

# Results

1. HDR maps reconstructed using the proposed method under outdoor environmental conditions achieved better HDR image scores than maps reconstructed using other existing methods.

    <img src="figure/res1.jpg" width="70%" height="70%">

2. The proposed method directly estimates the lighting of the sun and its surrounding area, overcoming the performance limit of existing HDR reconstruction methods. This limit is caused by overexposure areas of input LDR images under various outdoor weather conditions.

    <img src="figure/res2.jpg" width="90%" height="90%">

# Architecture

![arch](figure/arch.jpg)

# Requirements

- tensorflow >= 2.4
- tensorflow_adds_on
- opencv >= 4 (conda install -c conda-forge opencv)
- pandas
- tqdm
- matplotlib
</br></br>

# DataGeneration

- Generate input traing & test data from the Laval HDR dataset. \
(Redistribution of the Laval-HDR-dataset is not permitted. Please contact Jean-Francois Lalonde at jflalonde at gel dot ulaval dot ca to obtain the dataset.)

    ```
    python datasetGenerator.py

    (Optional)
    --dir :
        Absolute path of your dataset directory to save. (default : current working directory)
     
    --imheight :
        Vertical size of the output panoramic image (default : 32)

    --imwidth :
        Horizontal size of the output panoramic image (default : 128)
    ```

- To use your own input data,
    1. Make sure your input images conform to the sky-dome image format.  
        > Only sky-dome images converted from fisheye-lens images are available as input images.
(The sky-dome image is a panoramic image that captures the sky of $0\degree$ - $90\degree$ for elevation and $0\degree$ - $360\degree$ for azimuth.)
    2. Your dataset must conform to the TFRecord format described below.
        > The input data is encoded as TFRecord to improve training latency due to loading overhead.

        ```
        feature_description = {
            'image': _bytes_feature(image),
            'azimuth': _float_feature(azimuth),
            'elevation': _float_feature(elevation),
        }
        ```

# Training

1. Download the pre-trained weights of [vgg16](https://github.com/alex04072000/SingleHDR/tree/master/training_code#:~:text=trained%20weights%20of-,vgg16,-and%20vgg16_places365_weights).
    (This is a property of [SingleHDR](https://github.com/alex04072000/SingleHDR/tree/master/training_code))

2. Pretrain a sun luminance estimator.

    ```
    python pretrain_sun.py --dir "your/dir/path" --train True --inference_img_dir "your/dir/path"
    

    --dir :
        Absolute path of your dataset directory to train.
    
    --train :
        If "False", you can evaluate the sun luminance estimator that you trained.
    
    --inference_img_dir :
        Absolute path of your input LDR directory to evaluate your sun luminance estimator. (Enable only if --train False) 
    
    (Optional)
        --dorf :
            Absolute path of DoRF file. (provided in git repository)
        
        --lr : 
            learning rate of training model (default : 1e-4)

        --batchsize :
            batch size of training model (default : 32)

        --epochs :
            epochs number of training model (default : 1000)
        
        --imheight :
            Vertical size of the input panoramic image (default : 32)

        --imwidth :
            Horizontal size of the input panoramic image (default : 128)
    ```

    > Please make sure your dataset conform to our input format \
    (See item 2 in DataGenration).

3. Train a main model

    ```
    python train.py --dir "/your/dir/path" --sky "/sky/preweight/path" --sun "sun/preweight/path" --dorf="/txt/path" --vgg "/npy/path"

        --dir :
            Absolute path of your dataset directory to train.

        --sun :
            Absolute path of your sun luminance estimator weights file that pretrained on previous step (step 2).

        --vgg :
            Absolute path of pretrained weights file of vgg16. (see step 1)

    (Optional)
        --sky :
            Absolute path of your sky luminance estimator weights file that you pretrained.

        --dorf :
            Absolute path of DoRF file. (provided in git repository)
        
        --lr : 
           learning rate of training model (default : 1e-4)

        --batchsize :
            batch size of training model (default : 32)

        --epochs :
            epochs number of training model (default : 1000)
        
        --imheight :
            Vertical size of the input panoramic image (default : 32)

        --imwidth :
            Horizontal size of the input panoramic image (default : 128)
    ```

~~The pre-trained weight file can be downloaded from here.~~
</br></br>

# Inference

We have evaluated our model using two dataset (Laval-dataset, CAU dataset) respectively.

```
python inference.py --indir "abs/path" --outdir "name"

--indir :
    Absolute path of your dataset directory to inference.

--outdir :
    Specifies the directory name of the output inference image.

(Optional)
    --sky :
        Absolute path of your sky luminance estimator weights file that you pretrained.
        (default : os.path.join(CURRENT_WORKINGDIR, "checkpoints/SKY")))

    --sun :
        Absolute path of your sun luminance estimator weights file that you pretrained.
        (default : os.path.join(CURRENT_WORKINGDIR, "checkpoints/SKY")))
```

The CAU dataset can be downloaded from [here](https://drive.google.com/drive/folders/1-EujEiQdLnBVUENRKUOU56_g0PgdWYVI?usp=sharing).

# Citation

```
@article{shin2023hdr,
  title={HDR Map Reconstruction From a Single LDR Sky Panoramic Image For Outdoor Illumination Estimation},
  author={Shin, Gyeongik and Yu, Kyeongmin and Mark, Mpabulungi and Hong, Hyunki},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```
