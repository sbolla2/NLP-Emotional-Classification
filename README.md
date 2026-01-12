# NLP Empathy Detection and Emotional Classification

### Our Technical Report on our findings and process: [Technical Report](Technical_Report.pdf)

NLP Project modeling, investigating, and comparing the capturing of empathy, emotional polarity, and emotional intensity within text

This project is a solution to the second track of WASSA 2024 Shared Task on Empathy Detection and Emotion Classification and
Personality Detection in Interactions. Specifically, this project provides three different predictive models
for identifying emotional intensity, emotional polarity, and empathy in conversations between individuals discussing
topics in which harm has befallen another individual or group. The targets are defined as follows: 

- Emotion: How intense is the emotion the person speaking is feeling. 
- Polarity: How positive or negative is the emotion being felt. 
- Empathy: To what extent does the person
speaking feel empathy towards the subject of
conversation.

The models we have developed to predict these targets are a deep averaging network using n-gram features, a convolutional neural network, 
and a BERT model. The goal of this project is to compare the effectiveness of each of these models and determine if any one model is particularly
suited to any one target. 

## Instructions

The entry point of the project is evaluation.py and can be run using the command
`python evaluation.py`. the evaluation file accepts a number of arguments.

`--model` Determines which of the models to run. Can be one of NGRAM, CNN, BERT, or BASELINE. Default is CNN

`--target` Which target the model you are running should be predicting. Can be one of EMPATHY, INTENSITY, or POLARITY. 
Default is EMPATHY.

`--lr` Defines the learning rate to use in training. Default is 0.001.

`--num_epochs` Defines the number of training epochs to run. Default is 10. 

`--hidden_dim` Defines the hidden dimensions for N-Gram or filter size for CNN model. Default is 100.

`--dropout` Defines the dropout rate to use in training. Default is 0.5.

`--batch_size` Determines the batch size to use in training. Default is 50.

`--n-grams` Defines the number of words to include in an n-gram in the n-gram model. Default is 3. 

`--weight_decay` Defines weight decay to use in training. Default is 1e-4. 

`--train_path` Defines the file path to the training data to use in training. Defaults to data/trac2_CONVT_train.csv

`--dev_path` Defines the file path to the development data set to use in evaluation. Defaults to data/trac2_CONVT_dev.csv

`--test_path` Defines the file path to the test data set. Defaults to data/goldstandard_CONVT.csv

`--embeddings_path` Defines the file path to GloVe word embeddings. Defults to data/glove.6B.300d-relativized.txt

## Dependencies 
- pytorch
- numpy
- matplotlib
- transformers
- argparse
- scipy
- spacy
- pandas
