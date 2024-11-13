# Green Sweet Pepper Detection Using Mask R-CNN in Greenhouses Documentation

## Paper & Citation

Link: https://www.mdpi.com/2076-3417/13/10/6296

```
@Article{app13106296,
  AUTHOR = {López-Barrios, Jesús Dassaef and Escobedo Cabello, Jesús Arturo and Gómez-Espinosa, Alfonso and Montoya-Cavero, Luis-Enrique},
  TITLE = {Green Sweet Pepper Fruit and Peduncle Detection Using Mask R-CNN in Greenhouses},
  JOURNAL = {Applied Sciences},
  YEAR = {2023},
  URL = {https://www.mdpi.com/2076-3417/13/10/6296},
  DOI = {10.3390/app13106296}
}
```

This is an implementation of Mask R-CNN in Python 3.6, Keras 2.1.2, and TensorFlow-GPU 1.4.0 developed by the Matterport team (https://github.com/matterport/Mask_RCNN) for the detection of sweet pepper/bell pepper fruit and peduncle in its most common color varieties (green, red, yellow, and orange), focusing on green sweet pepper, as this color variety is the most challenging due to its similarity to the background in its actual production environment.

![b (4)](https://user-images.githubusercontent.com/107544707/205514983-d2b25fe5-1ae3-400f-bee9-ce1a12cda57e.png)

## Requirements

- Windows 10 is used in this specific implementation.
- Anaconda (this implementation uses a virtual environment in Anaconda).
  - https://www.anaconda.com
- CUDA Toolkit 8.0.61/8.0 GA2 and cuDNN v6.0 for CUDA 8.0 (if you have an NVIDIA GPU for hardware acceleration).
  - CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
    - Direct product version: https://developer.nvidia.com/cuda-80-ga2-download-archive
    - Documentation: https://docs.nvidia.com/cuda/archive/8.0/
  - cuDNN Archive: https://developer.nvidia.com/rdp/cudnn-archive
    - [Search] Download cuDNN v6.0 (April 27, 2017), for CUDA 8.0
- Visual Studio 2015 or Visual Studio Community 2015.

## Overview

- Step 1: Clone this repository.
- Step 2: Create the conda virtual environment based on the YAML file that contains all necessary dependencies.
- Step 3: Install pycocotools.
- Step 4: Install Mask R-CNN (model-code, main scripts).
- Step 5: Verify correct installation of dependencies.
- Step 6: Test it.

## Process

Run these commands in an Anaconda Prompt.

### Step 1 - Clone this repository

Within a specific folder for this environment (recommended), run this command:

`https://github.com/dassdinho/green_sweet_pepper_detection_using_mask_rcnn.git`

### Step 2 - Create the conda virtual environment based on the YAML file with all dependencies

- Enter the **Mask R-CNN** folder

`cd <DIRECTORY PATH>`

Example: `cd F:\mask_rcnn_models\green_sweet_pepper_detection_using_mask_rcnn\Mask_RCNN`

- Create the environment based on the YAML file

`conda env create -f <FILE PATH>`

Example: `conda env create -f F:\mask_rcnn_models\green_sweet_pepper_detection_using_mask_rcnn\MASK_RCNN_env_v1.yaml`

- Activate the environment

Each time we want to use the environment in a terminal/Anaconda prompt, we need to activate it (already within the Mask R-CNN folder).

`conda activate MASK_RCNN_env_v1`

Similarly, we can deactivate the environment so it is not available from that terminal.

`conda deactivate`
 
### Step 3 - Install pycocotools

Once inside the Mask R-CNN folder and with the environment activated...

- **NOTE**: pycocotools requires Visual C++ 2015 Build Tools.
  - Download if necessary: https://go.microsoft.com/fwlink/?LinkId=691126
    - You can also follow the instructions from the original repository: https://github.com/philferriere/cocoapi
- With the environment active, use pip to install pycocotools

`pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

### Step 4 - Install the Mask R-CNN model

Once inside the Mask R-CNN folder and with the environment activated...

Run the following command to install the Mask R-CNN model in our environment (***mask-rcnn***):

`python setup.py clean --all install`

Similarly, this command should be executed whenever we want to see the changes reflected in the scripts contained
in the ***mask-rcnn*** folder (main scripts defining the model and its tools).

### Step 5 - Check dependencies

Before discovering what this Mask R-CNN implementation can do, we need to check that everything we installed is in place.
To do this, use the following command and look for the names of all the dependencies we just installed (***pycocotools*** and ***mask-rcnn***):

`pip list`

![h23](https://user-images.githubusercontent.com/107544707/205516823-58813193-75d4-490b-8419-d4e416e75aa9.png)

### Step 6 - Let's try it out!

At this point, we will briefly explain the relevant files for training and visualizing our custom Mask R-CNN model. For a better understanding of the recommended codes, please refer to the original repository (https://github.com/matterport/Mask_RCNN):

### Info in the original repository

![cc](https://user-images.githubusercontent.com/107544707/205518250-03d400f3-792e-48a4-9425-ec1f7e9e51ba.JPG)

#### Folder *annotation_tool*

In the main directory (*Mask R-CNN*), there is the *annotation_tool* folder, which contains the tool (in a specific version) used to annotate our instance segmentation masks.
The tool in question is the VIA VGG Image Annotator (https://www.robots.ox.ac.uk/~vgg/software/via/).

#### Folder *mrcnn*

Within the *mrcnn* folder in the main directory, we can modify the model configuration and tools for visualizing the results.

![bb](https://user-images.githubusercontent.com/107544707/205518263-e66c7ca3-bf59-46b1-a088-bbd5f5390784.JPG)

#### Folder *dataset*

Within the *dataset* folder in the main directory, we can add our images and JSON files with the annotations we generate for training the model.

The information about the dataset used can be found at the following link: https://drive.google.com/drive/folders/1EcMkmG8q7ZoA0WM5FOjpXURkvLe2UoGr
 
#### Folder *logs*

Inside the *logs* folder, in the main directory, we can find the .H5 files that contain the weights of our models at each epoch (file that defines our trained model ready to perform inference/predictions).

The .H5 files generated in this implementation for sweet pepper can be found at the following link: https://drive.google.com/drive/folders/1EcMkmG8q7ZoA0WM5FOjpXURkvLe2UoGr

#### Folder *samples*

Inside the *samples* folder, in the main directory, we find all the Jupyter Notebooks used to understand the model, the databases used, and the results obtained through various examples described in the original repository.

In this specific case, our implementation for sweet pepper fruits and peduncles is located in the *pepper* folder, where we find the following:
- ***pepper_dass.py***: Essentially, the script needed to train the Mask R-CNN implementation with custom databases, where we can specify hyperparameters and configure all aspects of our model for the required objective.

- ***inspect_pepper_data_dass.ipynb***: This notebook allows us to visualize step-by-step the process that the Mask R-CNN model follows to obtain the instance segmentation masks on our own databases.

- ***inspect_pepper_model_dass.ipynb***: This notebook lets us view the final results of metrics, masks, detected object, defined class, and bounding box of our customized Mask R-CNN model.

![cc (2)](https://user-images.githubusercontent.com/107544707/205518299-f0a9c71a-927b-49ee-bace-76551d30e6bc.JPG)

*In this way, any emerging error or doubt is very likely to be found in the Issues section of the original repository.*

### ***Just start with the Jupyter Notebook demo.ipynb and run it.***

## Ready? Let's go!
