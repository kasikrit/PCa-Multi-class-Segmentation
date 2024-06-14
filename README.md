# PCa-Multi-class-Segmentation
Code for Enhancing Prostate Cancer Detection and Grading with the DARUN Model for Multi-class Semantic Segmentation

Software: Python 3.9.16 and TensorFlow 2.11.0

Dataset is avaiable at DOI: [10.21227/jy12-2c41](https://dx.doi.org/10.21227/jy12-2c41)

Abstract:
Prostate cancer is a major global health challenge. In this study, we present an approach for the early detection of prostate cancer through the semantic segmentation of adenocarcinoma tissues, specifically focusing on distinguishing Gleason patterns 3 and 4. Our method leverages deep learning techniques to improve diagnostic accuracy and enhance patient treatment strategies. We developed a new dataset comprising 100 digitized whole-slide images of prostate needle core biopsy specimens, publicly available for research purposes. Our proposed model, DARUN, integrates dilated attention mechanisms and a residual convolutional U-Net architecture to enhance the richness of feature representations. We addressed class imbalance using pixel expansion and computed class weights and implemented a five-fold cross-validation method for robust training and validation. The DARUN model achieved an average accuracy of 0.82 on unseen test data. The segmentation and grading results were validated by a team of expert pathologists. Additionally, our ablation study demonstrated the model's generalization and robustness, with high Jaccard and Dice coefficients (0.91 and 0.95, respectively) on a separate testing set. Based on a limited dataset, this study demonstrates the potential of our proposed methodologies and the DARUN model as a promising tool for the early detection of prostate cancer in clinical settings.

Keywords:
Prostate cancer, semantic segmentation, multi-class segmentation, Gleason patterns, whole-slide image, dilation rate, residual convolution, attention gate, adenocarcinoma segmentation, early cancer detection.
