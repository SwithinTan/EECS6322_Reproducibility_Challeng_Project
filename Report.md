EECS6322 Reproducibility Challenge Final Report

## Main Contribution

In the paper, the main contributions are:

1.  They introduced a new approach to detect CNN-generated images, by training forensics models on CNN-generated images. This approach shows a surprising amount of generalization compared with other CNN systhesis methods. 
2. They introduced a new dataset and evaluation metrics for detecting CNN-generated images.
3. They experimentally analyzed the factors that account for cross-model generalization.





## About My Reproducibility Attempt

There are two main attempts in my reproduction project that correspond to the two main contributions of the original paper:

1. The first attempt is to verify if the data augmentations improve generalization.
2. The second attemp is to verify if more diverse datasets improve generalization on unseen architectures.

First, we will try to verify the first and then the second if time allows.

### Data Augmentations Improve Generalization

1. Dataset
2. Model
3. Methodology (Augmentations)
4. 











### Diverse Datasets Improve Generalization





## Difficulties

1. Dataset downloader provided by the author has been deprecated, and the size of dataset is relatively large (70GB), which causes some trouble when I was trying to ingest dataset. This problem has been solved by downloading original datasets and save to another drive. 