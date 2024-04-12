# Model Card

## Model Details
- Model created on April 12th, 2024 by Gabriel Santoyo for Udactity Machine Learning DevOps course.
- The Random Forest Classifier was used to train the model

## Intended Use
- To determine whether a person makes under or over 50k a year based on their demographic information

## Training Data
- The data from this model is census data retrieved from UC Irvine's machine learning repository 
- More info about this dataset can be found at https://archive.ics.uci.edu/dataset/20/census+income
- Data has been preprocessed using one hot encoding for the categorical features and a label binarizer for the labels
- 80% of the data is used for training, while 20% of the data is used for testing

## Evaluation Data
- The data from this model is census data retrieved from UC Irvine's machine learning repository 
- More info about this dataset can be found at https://archive.ics.uci.edu/dataset/20/census+income
- Data has been preprocessed using one hot encoding for the categorical features and a label binarizer for the labels

## Metrics
- Metrics from the most recent run are as follows: Precision: 0.7052 | Recall: 0.6297 | F1: 0.6653

## Ethical Considerations
- This census data is from 1994 and may reflect societal biases and inequalities both in general and due to its time period. Machine learning models trained on such data can accidentally continue or worsen these biases.
- There is always a possibility of inaccuracy of census data, as well as the fact that this dataset contains missing values in some columns.

## Caveats and Recommendations
- The model has limitations, such as its inability to capture all factors that may influence income.
- This model should not be used to gather insights for the present day, as the data is extremely outdated.