#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#Create your df here:
df = pd.read_csv("profiles.csv")
#df_states = pd.read_csv("states.csv")


# Exploring the data
explore_education = df.education.value_counts()
print("------------------------------------\n")
print("Education:")
print(explore_education)

job_responses = df.job.value_counts()
print("\n------------------------------------\n")
print("Jobs:")
print(job_responses)


# Visualizing some data
# Height
plt.hist(df.height.dropna(), bins = 60, range = [50,97])
plt.xlabel("Height (inches)")
plt.ylabel("Frequency")
plt.title("Frequency of Height")
print("\n------------------------------------\n")
plt.show()

#Income
plt.hist(df.income.dropna(), bins = 30, range = [0,500000])
#plt.hist(df.income.dropna())
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Frequency of Income (First Half)")
print("\n------------------------------------\n")
plt.show()

#Income
plt.hist(df.income.dropna(), bins = 30, range = [500000,1000000])
#plt.hist(df.income.dropna())
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Frequency of Income (Second Half)")
print("\n------------------------------------\n")
plt.show()

#Age
plt.hist(df.age.dropna(), bins = 52, range = [17,70])
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Frequency of Age")
print("\n------------------------------------\n")
plt.show()


# Formulate a Question

education_mapping = {'graduated from college/university': 0,
                    'graduated from masters program':1,
                    'working on college/university': 2,
                    'working on college/university': 3,
                    'working on masters program': 4,
                    'graduated from two-year college': 5,
                    'graduated from high school': 6,
                    'graduated from ph.d program': 7,
                    'graduated from law school': 8,
                    'working on two-year college': 9,
                    'dropped out of college/university': 10,
                    'working on ph.d program': 11,
                    'college/university': 12,
                    'graduated from space camp': 13,
                    'dropped out of space camp': 14,
                    'graduated from med school': 15,
                    'working on space camp': 16,
                    'working on law school': 17,
                    'two-year college': 18,
                    'working on med school': 19,
                    'dropped out of two-year college': 20,
                    'dropped out of masters program': 21,
                    'masters program': 22,
                    'dropped out of ph.d program': 23,
                    'dropped out of high school': 24,
                    'high school': 6,
                    'working on high school': 25,
                    'space camp': 26,
                    'ph.d program': 27,
                    'law school': 28,
                    'dropped out of law school': 29,
                    'dropped out of med school': 30,
                    'med school': 31}
df['education_code'] = df.education.map(education_mapping)


#location_mapping = {}
#df["location_code"] = df.location.map(location_mapping)

job_mapping = {'student': 0,
               'science / tech / engineering':1,
               'computer / hardware / software': 2,
               'artistic / musical / writer': 3,
               'sales / marketing / biz dev': 4,
               'medicine / health': 5,
               'education / academia': 6,
               'executive / management': 7,
               'banking / financial / real estate': 8,
               'entertainment / media': 9,
               'law / legal services': 10,
               'hospitality / travel': 11,
               'construction / craftsmanship': 12,
               'clerical / administrative': 13,
               'political / government': 14,
               'rather not say': 15,
               'transportation': 16,
               'unemployed': 17,
               'retired': 18,
               'military': 19,
               'other': 20}
df['job_code'] = df.job.map(job_mapping)


df = df.dropna(subset = ['education_code',
                         'job_code',
                         'age',
                         'height'])

#QUESTION 1 & QUESTION 2 DATA:
data = df[['education_code','job_code']]

x_val = data.values
scaler_min_max = preprocessing.MinMaxScaler()
x_val_scaled = scaler_min_max.fit_transform(x_val)
data = pd.DataFrame(x_val_scaled, columns = data.columns)

#QUESTION 1 Y-VALUE:
y_val = df[['age']]

#QUESTION 2 Y-VALUE:
#y_val = df[['height']]

train_x, test_x, train_y, test_y = train_test_split(data,y_val,test_size = 0.20)

start_time_1 = time.time()

# CLASSIFICATION - NAIVE BAYES CLASSIFIER
class_nb = MultinomialNB()
class_nb.fit(train_x, train_y.values.ravel())

nb_prediction = class_nb.predict(test_x)
print("\n------------------------------------\n")
print("Prediction of using Naive Bayes Classifier:")
print(nb_prediction)

print("\n------------------------------------\n")
print("Time to run above model (ms):")
time2 = time.time() - start_time_1
print(time2)

pred_probability = class_nb.predict_proba(test_x)
print("\n------------------------------------\n")
print("Prediction of Probability using Naive Bayes Classifier:")
print(pred_probability)

print("\n------------------------------------\n")
print("Accuracy of Naive Bayes Classifier:")
print(accuracy_score(test_y,nb_prediction))

print("\n------------------------------------\n")
print("Precision of Naive Bayes Classifier:")
print(precision_score(test_y,nb_prediction, average = None))

print("\n------------------------------------\n")
print("Recall of Naive Bayes Classifier:")
print(recall_score(test_y,nb_prediction, average = None))

start_time_4 = time.time()

# CLASSIFICATION - K-Nearest
classifier_k = KNeighborsClassifier(n_neighbors = 40)
classifier_k.fit(train_x, train_y.values.ravel())

k_class_pred = classifier_k.predict(test_x)
print("\n------------------------------------\n")
print("Prediction using K-Nearest Classifier:")
print(k_class_pred)

print("\n------------------------------------\n")
print("Time to run above model (ms):")
#time4 = time.time() - time3
print((time.time() - start_time_4))

pred_k_probability = classifier_k.predict_proba(test_x)
print("\n------------------------------------\n")
print("Prediction of Probability using K-Nearest Classifier:")
print(pred_k_probability)

print("\n------------------------------------\n")
print("Accuracy of K-Nearest Classifier:")
print(accuracy_score(test_y,k_class_pred))

print("\n------------------------------------\n")
print("Precision of K-Nearest Classifier:")
print(precision_score(test_y, k_class_pred, average = None))

print("\n------------------------------------\n")
print("Recall of K-Nearest Classifier:")
print(recall_score(test_y, k_class_pred, average = None))

start_time_2 = time.time()

# REGRESSION APPROACHES: MULTIPLE LINEAR REGRESSION
mlr = LinearRegression()
model = mlr.fit(train_x, train_y.values.ravel())
mlr_predict = mlr.predict(test_x)

print("\n------------------------------------\n")
print("Prediction using MULTIPLE LINEAR REGRESSION:")
print(mlr_predict)

print("\n------------------------------------\n")
print("Time to run above model (ms):")
print((time.time() - start_time_2))

print("\n------------------------------------\n")
print("R^2 score of MULTIPLE LINEAR REGRESSION:")
#rms = mean_squared_error(test_y,mlr_predict)
print(r2_score(test_y,mlr_predict))

print("\n------------------------------------\n")
print("Variance Score of MULTIPLE LINEAR REGRESSION:")
print(explained_variance_score(test_y,mlr_predict))

print("\n------------------------------------\n")
print("Mean Absolute Error of MULTIPLE LINEAR REGRESSION:")
print(mean_absolute_error(test_y,mlr_predict))

#REGRESSION - K-NEAREST (OPTION 2)
start_time_3 = time.time()

regressor = KNeighborsRegressor(n_neighbors = 40, weights = "distance")
regressor_fit = regressor.fit(train_x, train_y.values.ravel())
k_predict = regressor.predict(test_x)

print("\n------------------------------------\n")
print("Prediction using K-NEAREST REGRESSION:")
print(k_predict)

print("\n------------------------------------\n")
print("Time to run above model (ms):")
print((time.time()-start_time_3))

print("\n------------------------------------\n")
print("R^2 score of K-NEAREST REGRESSION:")
rms_k = mean_squared_error(test_y,k_predict)
print(r2_score(test_y,k_predict))

print("\n------------------------------------\n")
print("Variance Score of K-NEAREST REGRESSION:")
print(explained_variance_score(test_y,k_predict))

print("\n------------------------------------\n")
print("Mean Absolute Error of K-NEAREST REGRESSION:")
print(mean_absolute_error(test_y,k_predict))

print("\n------------------------------------\n")
accuracies = []
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(train_x, train_y)
    accuracies.append(classifier.score(test_x, test_y))
k_list = range(1,101)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Classifier Accuracy")
plt.show()






