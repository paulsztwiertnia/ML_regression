# ML_regression
introduction to pandas, sklearn, linear and logistic regression, multi-class classification.

PART 1: basic linear regression
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant is
located. The chain already has several restaurants in different cities. The goal is to model the relationship
between the profit and the populations from the cities where they are located.

Load the data from the file RegressionData.csv in a pandas dataframe, then to plot
the data using a scatter plot to visualize the data. Using the training data, train a linear regression
model, and eventually, apply it to predict the profit of a restaurant located in a city of 18 habitants.

PART 2: logistic regression
Goal is to predict whether an applicant is likely to get hired or rejected. Your task is to use logistic
regression to build a model that predicts whether an applicant is likely to be hired or not, based on the
results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.

PART 3: multi-class classification using logistic regression
Not all classification algorithms can support multi-class classification (classification tasks with more
than two classes). Logistic Regression was designed for binary classification.
One approach to alleviate this shortcoming, is to split the dataset into multiple binary classification
datasets and fit a binary classification model on each. 
