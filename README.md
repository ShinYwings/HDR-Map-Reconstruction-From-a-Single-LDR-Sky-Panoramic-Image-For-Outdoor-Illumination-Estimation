# HDR Map Reconstruction From a Single LDR Sky Panoramic Image For Outdoor Illumination Estimation

![result1](figure/rendered.png)
A multi-faceted approach to reconstructing HDR maps from a single LDR sky panoramic image that considers the sun and sky regions separately.
(This paper is under review)
</br></br>

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
    ```

- The input data is encoded as TFRecord to improve training latency due to loading overhead.

- To use your own input data, your data must conform to the TFRecord format described below.

    ```
    feature_description = {
        'image': _bytes_feature(image),
        'azimuth': _float_feature(azimuth),
        'elevation': _float_feature(elevation),
    }
    ```

# Train

1. Pretrain a sun luminance estimator.

    ```
    python pretrain_sun.py --dir="path" --train=True --inference_img_dir=absolute/path/of/your/ldr/directory/path/to/evaluate/the/sun/luminance/estimator
    

    --dir :
        absolute directory path if you would like to train your own dataset.
    
    --train :
        if "False", you can evaluate the sun luminance estimator 
        you trained.
    
    --inference_img_dir :
        When --train=False, You must insert the absolute path of 
        your LDR directory to evaluate your sunluminance estimator.

    ```

    If you don't use datasetGenerator.py, you should conform to our TFRecord format \
    (See item 3 in DataGenration).

2. Train a main model (until around 450 epochs)

    ```
    python train.py
    ```

~~The pre-trained weight file can be downloaded from here.~~
</br></br>

# Inference

We have evaluated our model using two dataset (Laval-dataset, CAU dataset) respectively.

```
python test_real_refinement.py
```

The CAU dataset can be downloaded from [here](https://drive.google.com/drive/folders/1-EujEiQdLnBVUENRKUOU56_g0PgdWYVI?usp=sharing).
