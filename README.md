# HDR-Map-Reconstruction-From-a-Single-LDR-Sky-Panoramic-Image-For-Outdoor-Illumination-Estimation
사진사진
a multi-faceted approach to reconstructing HDR maps from a single LDR sky panoramic image that considers the sun and sky regions separately.
(This paper is under review)


# Requirements
opencv >= 4 (conda install -ㅊ conda-forge opencv)
tensorflow >= 2.4
tensorflow_adds_on

# Dataset Generation
```
python datasetGenerator.py
```
The CAU dataset can be downloaded from here([link](https://drive.google.com/drive/folders/1-EujEiQdLnBVUENRKUOU56_g0PgdWYVI?usp=sharing)).
(Sythetic dataset는 Lalonde의 창작물이므로 제공이 불가하다.)

# Pretraining
```
python pretrain_sun.py
```

# Train
```
python train.py
```
The pre-trained model can be downloaded from here(link).

# Inference
```
python test_real_refinement.py
```

# Results
사진사진
