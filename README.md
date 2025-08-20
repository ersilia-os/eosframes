# data-frames
Template repository for an Ersilia Python package.

This package provides a foundation for developing and distributing Python tools within the Ersilia ecosystem. It is designed to help researchers and developers quickly set up, share, and maintain reproducible code for AI/ML models, particularly in the context of antimicrobial drug discovery.
data-frames is a Python package for fitting data transformation pipelines and applying them to new datasets. It is built for reproducible preprocessing of numeric data—including median imputation, robust scaling, Yeo-Johnson power transforms, and quantization.

---

## Features
 - **Fit & Save Pipelines**: Train a transformation pipeline on one dataset, then save and reuse it on new data.
 - **Robust Scaling**: Reduce the influence of outliers using 'RobustScaler'.
 - **Power Transformation**: Stabilize variance and make data more Gaussian-like with Yeo-Johnson transforms.
 - **Quantization**: Convert continuous features into evenly-spcaed bins.
 - **Integration with pandas**: Works directly on 'DataFrame' objects

 ___

 ## Installation
 ''' bash
 git clone https://github.com/patrickwalsh26/data-frames.git
 cd data-frames
 pip install -e

