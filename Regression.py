"""
Goals: introduction to pandas, sklearn, linear and logistic regression, multi-class classification.
"""

import pandas
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
The goal is to predict the profit of a restaurant, based on the number of habitants where the restaurant 
is located. The chain already has several restaurants in different cities. Your goal is to model 
the relationship between the profit and the populations from the cities where they are located.
"""

# Loading the data from the file RegressionData.csv in a pandas dataframe. Name the first feature 'X' and the second feature 'y' (these are the labels)

data = pandas.read_csv('/Users/pawelsztwiertnia/Desktop/2023fall/cps803/a2/RegressionData.csv', header=2, names=['X', 'y']) # 5 points

# Reshape the data so that it can be processed properly

X = data['X'].values.reshape(-1,1) 
y = data['y'] 
# Plot the data using a scatter plot to visualize the data
plt.scatter(x=data['X'], y=data['y']) 
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.show()

# Linear regression using least squares optimization
reg = linear_model.LinearRegression() 
reg.fit(X,y) 


# Plot the linear fit
fig = plt.figure()
y_pred = reg.predict(X)
plt.scatter(X, y, c='b', label='Data points')
plt.plot(X, y_pred, 'r', label='Linear fit')
plt.xlabel('Cdata 1')
plt.ylabel('Cdata 2')
plt.title('Linear Fit of the Data')
plt.legend()
plt.show()

# Complete the following print statement 
b_0 = reg.intercept_
b_1 = reg.coef_[0]
print("The linear relationship between X and y was modeled according to the equation: y = b_0 + X*b_1, \
where the bias parameter b_0 is equal to ", b_0, " and the weight b_1 is equal to ", b_1)



# Predict the profit of a restaurant, if this restaurant is located in a city of 18 habitants 
city_pop = 18
predicted_profit = reg.predict([[city_pop]])
print("The profit/loss in a city with", city_pop ,"habitants is ",predicted_profit[0],".")



"""
Goal is to predict whether an applicant is likely to get hired or rejected. 
Using the gathered data over the years to use as a training set. 
Task is to use logistic regression to build a model that predicts whether an applicant is likely to
be hired or not, based on the results of a first round of interview (which consisted of two technical questions).
The training instances consist of the two exam scores of each applicant, as well as the hiring decision.
"""

# Load the data from the file 'LogisticRegressionData.csv' in a pandas dataframe. Make sure all the instances 
# are imported properly. Name the first feature 'Score1', the second feature 'Score2', and the class 'y'
data = pandas.read_csv('/Users/pawelsztwiertnia/Desktop/2023fall/cps803/a2/LogisticRegressionData.csv', header = 2, names=['Score1', 'Score2', 'y']) # 2 points

# Seperate the data features (score1 and Score2) from the class attribute 
X = data[['Score1','Score2']] 
y = data['y'] 

# Plot the data using a scatter plot to visualize the data. 
# Represent the instances with different markers of different colors based on the class labels.
m = ['o', 'x']
c = ['hotpink', '#88c999']
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker=m[data['y'][i]], color = c[data['y'][i]]) 
plt.xlabel('Score1')
plt.ylabel('Score2')
plt.title('Scatter plot of raw data')
fig.canvas.draw()
plt.show()

# Train a logistic regression classifier to predict the class labels y using the features X
regS = linear_model.LogisticRegression(solver='liblinear') 
regS.fit(X,y) 

# Now, we visualize how well does the trained classifier perform on the training data
# Use the trained classifier on the training data to predict the class labels
y_pred = regS.predict(X) 
# To visualize the classification error on the training instances, we will plot again the data. However, this time,
# the markers and colors selected will be determined using the predicted class labels
m = ['o', 'x']
c = ['red', 'blue'] # this time in red and blue
fig = plt.figure()
for i in range(len(data)):
    plt.scatter(data['Score1'][i], data['Score2'][i], marker= m[y_pred[i]], color = c[y_pred[i]]) 
plt.xlabel('Score1')
plt.ylabel('Score2')
plt.title('Scatter plot of predicted data')
fig.canvas.draw()
plt.show()

