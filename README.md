## Overview
This repository contains the source code for the models used for the submission Semantic Coherence Markers: the Contribution of Perplexity Metrics.

## Prerequisites
#### 1 - Install Requirements
```
conda env create -f environment.yml
```
NB. It requires [Anaconda](https://www.anaconda.com/distribution/)

#### 2 - Download pre-trained models and fine-tuned models
Download data from [this link](https://drive.google.com/file/d/1YYMmFlwrNnuQSgUnDlteSrLopm82VyCo/view?usp=sharing) (295GB) and unpack the archive in the resource folder.

## Execution
In order to reproduce the submission results run the scripts in 'src/run/' package. In particular, from the src folder run the command:
```
python -m run.experimentN
```
where N refers to the number of the experiment [1,2,3].