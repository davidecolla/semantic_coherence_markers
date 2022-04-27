## Overview
This repository contains the source code for the models used for the submission Semantic Coherence Markers: the Contribution of Perplexity Metrics.

## Prerequisites
#### 1 - Install Requirements
```
conda env create -f environment.yml
```
NB. It requires [Anaconda](https://www.anaconda.com/distribution/)

#### 2 - Download the transcript from the URL list and the documents from the Pitt Corpus

#### 3 - Download pre-trained models and fine-tuned models
Download GPT2 data from [this link](https://drive.google.com/file/d/1YYMmFlwrNnuQSgUnDlteSrLopm82VyCo/view?usp=sharing) (295GB) and unpack the archive in the resource folder.

## Execution
In order to reproduce the submission results run the scripts in 'src/run/' package. In particular, from the src folder run the command:
```
python -m run.experiment1
```
to run the first experiment. The script will fine-tune the GPT2 model for [0,5,10,20,30] epochs and acquire N-grams ranging N from 2 to 5.

In order to run the second experiment run the following command:
```
python -m run.experiment2
```
as for the first experiment, the script will go through all the experimental settings presented in the article and will compute the ICC score.

Finally, to run the third experiment run the following command:
```
python -m run.experiment3
```
as for the previous settings, the script will run all the conditions presented in the paper. The evaluation metrics will be computed after all the LMs have been acquired.

In order to avoid long training times, all the experimental conditions have been optimized, if the GPT2 models have been downloaded and unpacked in the resource folder, the scripts skip the fine-tuning phase; they start the training otherwise.
