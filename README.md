# GAARC
Final Masters' Project: Generalist Approach to Abstraction and Reasoning Challenge

This repository contains the code used on the development ambit of my Final Masters' Degree Project, since as I started to have multiple processes that made use of the same code, having all instances of the code aligned became more problematic and time-consuming than just taking the tame to generate a centralized code base.

## Introduction

Generative Approach to Abstraction and Reasoning Challenge (whose beautiful acronym, GAARC, will be used from now on to refer to it) aims to be a proposal to the (Abstracted And Reasoning Challenge)[https://www.kaggle.com/c/abstraction-and-reasoning-challenge], introduced by François Chollet, that aims to properly follows a Generalistic approach. The complete extent of the approach is explained within the Final Master's Degree Project's document, which will be, if possible, linked here upon finalization.

As a brief overlook, the approach of this project is to try to achieve, through multi-task learning, a rich hidden space that captures a complex enough interpretation of the data so that the tasks can be confronted through few-shot learning over the required changes in that hidden space interpretation.

## Overview
TBD

## Repository Contents
```
gaarc/
├── data/
│   ├── arc_data_models.py
│   ├── augmentation.py
│   ├── data_interface.py
│   ├── preprocessing.py
│   └── transformations.py
├── model/
│   ├── autoencoder.py
│   └── unet.py
└── visualization/
    └── arc_visualization.py
```
### Gaarc modules:
- **data**: Data management, from preprocessing and transformations for augmentation to data interface classes.
- **model**: Everything related to model architecture definition and management automation.
- **visualization**: General visualization tools for the rest of the components.

### Notebooks

The original idea was to bring the notebooks used during the development of this work into this repository, under the directory notebooks/. 

Nevertheless, at the time of writting, given that the notebooks have been used to iterate over multiple configurations of the models and haven´t been brought to a proper clean and explicative state to serve as a standalone walkthrough of the developed functionality. Thus, instead of keeping a single copy here within the repository, and until I have time to provide something of proper quality, the notebooks presence in the repository will be relegated to links to the published versions in Kaggle, which also holds alongside them the different versions each notebook have iterated over.


- [U-Net Gaarc base autoencoder](https://www.kaggle.com/code/varnez/u-net-gaarc-base-autoencoder)
- [Task learning with U-Net Gaarc base autoencoder](https://www.kaggle.com/code/varnez/task-learning-with-u-net-gaarc-base-autoencoder)
- [Task data exploration](https://www.kaggle.com/code/varnez/task-data-exploration)


## ToDo


### AutoEncoder

- In the Autoencoder and subsequent modules, manage a way to be able to explicitly select a device, and properly propagate to all the submodules, keeping a reference as a property for the subtask modules. That is, so far, the biggest painpoin in terms of automated alignment.
- Store the ARCSample cache within the autoencoder so that multiple STMs only need to cache them once.


### STMs
- Allow submodule tasks to select an specific subset of entities (or similar) to work with per step.
- Add other metrics to the tasks, to improve understanding.
