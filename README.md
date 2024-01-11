# UBC-OCEAN_5th_solution

## Competition:

UBC Ovarian Cancer Subtype Classification and Outlier Detection (UBC-OCEAN)

[Solution summary](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/466017)

## Prepare data

* UBC data:
    * WSI: Crop 1536 x 1536 pactches of tissue part without overlapping
    * TMA: Do nothing
    * Collect all WSI patches and TMA data to make a csv file for training
        * The csv file we used for final submission is provided as sample, the columns not used in the training script
          are irrelevant.
* Minimum external data used as 'Other' class:
    * Hubmap
      data: [HuBMAP + HPA - Hacking the Human Body](https://www.kaggle.com/competitions/hubmap-organ-segmentation/overview)

## Train

### Segmentation model

Training script: train_segmentation.py

* With segmentation model for selecting top 5 valuable patches:  Private LB=0.61
* Without segmentation model, just use first 5 patches with tissue: Private LB=0.58

For the details of training segmentation model, please refer to

### Classification model

Training script: train_classification.py

1. Train 5 models using 5 fold data to make pseudo labels for all the patches
2. Using pseudo labels to train a model with full dataset for submission

## Inference

Script: [Inference notebook](https://www.kaggle.com/code/liushuzhi/5thplacesolutionsubnotebook?scriptVersionId=158430537)

For details of model training and inference, please refer to
the [Solution summary](https://www.kaggle.com/competitions/UBC-OCEAN/discussion/466017)