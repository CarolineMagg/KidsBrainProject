# KidsBrainProject

## Citation

Caroline Magg, Laura Toussaint, Ludvig Paul Muren, Danny Indelicato, Renata Raidou
Visual Assessment of Growth Prediction in Brain Structures after Pediatric Radiotherapy
In Eurographics Workshop on Visual Computing for Biology and Medicine (VCBM2021)., pages 31-35. September 2021.
https://doi.org/10.2312/vcbm.20211343 

## Overview

In this project, we solved three different tasks:
* segmentation predictions with annotated pre-treatment data and unannotated post-treatment data
* accuracy quantification without GT labels
* visualization of the brain growth for multiple time points

![three tasks](https://github.com/CarolineMagg/KidsBrainProject/blob/master/graphics/fig_5.PNG)

## Details

The data available includes pre-treatment CT with corresponding segmentation masks and post-treatment MRI data at different time points.
We employ an Active Contour Model to generate segmentation masks on the post-treatment data and trained a SVR to quantify the accuracy of the predictions.
The predictions and the estimated accuracy are then combined in a brain growth visualization. The visualization is superimposed on top of the MRI or CT scans to provide anatomical context.

![](https://github.com/CarolineMagg/KidsBrainProject/blob/master/graphics/fig_1.PNG)

For more details, check out the conference paper.

## Results

The visualization supports different color maps.

![](https://github.com/CarolineMagg/KidsBrainProject/blob/master/graphics/fig_4.PNG)

The segmentation predictions works good on larger brain structures, e.g., the brain or the temporal lobe, and needs improvement for smaller brain structures, e.g., the Cingulum or the Thalamus. However, the correct accuracy behavior is learned by the SVR.

![](https://github.com/CarolineMagg/KidsBrainProject/blob/master/graphics/acm_svr.png)
