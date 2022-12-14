![](UTA-DataScience-Logo.png)

# NLP - FastAI Transfer Learning with Disaster Tweets

* Classify tweets as pertaining to real or otherwise disasters with FastAI and AWD-LSTM.
 **Natural Language Processing with Disaster Tweets**:  https://www.kaggle.com/competitions/nlp-getting-started/overview/

## Overview

The goal of this Kaggle challenge is to use Natural Language Processing (NLP) to determine whether specific tweets are talking about disasters that have actually happened, give a prefined set of tweets categorized as having actually happened or not, (0,1). This challenge is overcome using transfer learning with FastAI, a deep learning focused python library built ontop of pytorch. The score this bot received, when turing in the sample submission csv on kaggle, was 79.9%.

## Summary of Workdone

Include only the sections that are relevant an appropriate.

### Data

* Data:
  * Type: input: CSV containing text data, output: CSV containing boolean data
  * Size: train.csv: 987kB, test.csv: 421kB
  * Instances: Training dataset: 6003 unique entries, Validation dataset: 1500 unique entries (20% split), Test dataset: 3243 Unique entries

#### Preprocessing / Clean up

* Given that the text data is in the form of tweets, the data is going to contain a lot of symbols and links that won't directly affect whether something is or is not an actual disaster. Therefore, using a function contained in the 'utils.py' called 'tweet_cleaning' cleans up the tweets following these instructions:
	* Strip the '#' character
	* Remove '@' and handles following
	* Remove http, www, and other various url headings and their following links
	* Set text to lowercase
- Next, the data needs to be Vectorized in some fashion, in order to be more appropriately interpreted by the model. In the model, there is a function that takes the text data from a pandas dataframe and vectorizes it using FastAIs TextDataLoader object.

### Problem Formulation

The Input of this function is text, so I wanted to find a package that focuses on and has functions created for dealing with text, leading to FastAI as a framework. The method of attack this project uses is transfered learning, limiting the direct control over certain hyperparameters. However, many choices still had to be made:
- Data is vectorized and segmented using FastAIs TextDataLoader, with the validation percent set to 20% and a specific seed chosen of '1337'
- Setting up the pretrained model with the new dataset, an architecture AWD-LSTM(1), Dropout Multiplier of 0.5 to scale the dropout weights for nodes, and the metrics set to accuracy.
- Finally, the learning rate used is approximated by emualting fitting with different learning rates from 10e-7 to 10e0 to find a valley. The valley is found the 0.002, so thats what was chosen for the model.(2)
![[lr_finder.png]]

### Training
Through various trails in find tuning, I found that training with 3 epochs yielded the best split between training and validation losses. Below is a chart graphing the loss over trained data (split-up epochs).
![[Pasted image 20221213193421.png]]

Overall, the training took 13 minutes and 37 seconds on my hardware using the cpu build of pytorch on my computer. 

### Conclusions

Transfer learning FastAI, using AWD-LSTM and little in the way of tweaking hyperparameters, yielded results of 79.9, roughly 80% accuracy for a binary classification of text. This is a model that can absolutely be tweaked to a much higher percent given more time library. I worked to try and get ROC curves up for the conclusion/performance section, but I was unable to pull the original values that matched up with the predicted labels.

### Future Work

This classifier can be refined into a more robust binary classifier, or be mutated to be able to take on multiple classes and rank how well a string fits into that category relatively simply. While there is already a layer of abstraction between data, preprocessing, and training/fitting the model, more can be done to allow for different types of classifiers, or preprocess for different types of text data.

## How to reproduce results

Given a train.csv and test.csv, running the inference notebook will create the model and save it after training, then run the test.csv versious the model trained on train.csv, and output some submission.csv if not already provided, that contains the results of the prediction done on test.

### Overview of files in repository

|-- DATA-3402-FINAL-NLP
|    |--models
|    |    |--model_1.pth: saved encoder fo the classifier
|    | --images: folder for readme related image
|    |--project-initialization.ipynb: downloads kaggle challenge related data and preprocesses
|    |--model-training-1.ipynb: model is created and saved here
|    |--inference.ipynb: previous two notebooks run, submission is generated and saved
|    |--utils.py: holds the preprocessing done to the text to remove extraneous features
|    |--README.md

### Software Setup
* List all of the required packages.
* If not standard, provide or point to instruction for installing the packages.
* Describe how to install your package.

All of the following packages are installing using the default PyPi package manager, PIP.
- numpy 1.23.5
- pandas 1.5.2
- matplotlib 3.6.2
- fastai 2.7.10

fastai requires PyTorch to be installed before installing through pip, so: additionally the pytorch package needs to be isntalled.

### Data
In the project-initialization.ipynb file, the kaggle cli is used to download data relevant to the NLP Disaster tweets challenge. Alternatively, the data can be downloaded from [here.](https://www.kaggle.com/competitions/nlp-getting-started/data) The preprocessing is done in the same notebook. Each entry of the text column in a csv is run through the 'tweet_cleaning' function from the utils.py module.

### Training

In the model-training-1.ipynb, the model is constructed and fit according to data provided in some 'train.csv'. Running the notebook will create the model based on the parameters chosen, and save the model into the 'models' folder.


## Citations

(1): Stephen Merity, Nitish Shirish Keskar, & Richard Socher. (2017). Regularizing and Optimizing LSTM Language Models. _ArXiv: Computation and Language_.

(2): Smith, L. N. (2017). Cyclical Learning Rates for Training Neural Networks. _2017 IEEE Winter Conference on Applications of Computer Vision (WACV)_. https://doi.org/10.1109/wacv.2017.58





