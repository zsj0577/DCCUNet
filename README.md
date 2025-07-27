# MSAGHNet
This is the repository for 'DCCUNet: A Double Cross-Shaped Network for Pathology Image Segmentation'

# Architecture
<p align="center">
<img src="img/Architecture.png">
</p>

# Dataset Structure
The dataset is organized as follows:

 - `data/`
    - `dataset_name/`: Name of the dataset used, such as CoNIC, DSB, PanNuke, and MoNuSeg
        - `train/`: Contains training dataset
          - `img/`: Training images
          - `mask/`: Corresponding segmentation masks for training images
        - `test/`: Contains training dataset
          - `img/`: Test images
          - `mask/`: Corresponding segmentation masks for test images          
        - `val/`: Contains validation dataset
          - `img/`: Validation images
          - `mask/`: Corresponding segmentation masks for validation images
            
    - `dataset_name/`: Name of the dataset used, such as CoNIC, DSB, PanNuke, and MoNuSeg
       - .......

# Train and Test
Please use Train.py and Test.py for model training and prediction. 

# Datasets
The following datasets are used in this experiment:
<ol>
  <li><a href="https://conic-challenge.grand-challenge.org/">CoNIC</a></li>
  <li><a href="https://www.kaggle.com/c/data-science-bowl-2018">DSB</a></li>
  <li><a href="https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke">PanNuke</a></li>
  <li><a href="https://monuseg.grand-challenge.org/">MoNuSeg</a></li>
 </ol>
