print ('Begin LogisticRegression...')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

#load dataset
fields = ['tweet_time', 'Sentiment', 'Score']
dataset = pd.read_csv("bitcoin_tweets_dataset_2_sentiment.csv",skipinitialspace=True,usecols = fields)

dataset['tweet_time'] = pd.to_datetime(dataset['tweet_time'])
#print (dataset['tweet_time'])
#print (dataset.keys())
#print (dataset.tweet_time, dataset.Sentiment, dataset.Score)
#dataset['Score'] = dataset['Score'].round().astype(int)
dataset['Sentiment'] = dataset['Score'].round().astype(int)

print ("here")
#2D array of X,y coordinates to feed the Logistic Regression model
#X = np.array([dataset['tweet_time'].values], dtype='float64')
#y = np.array(dataset['Score'].values)
X = np.array([dataset['Score'].values], dtype='float64')
y = np.array(dataset['Sentiment'].values)

#X = X.reshape(X.shape[1:])
X = X.transpose()


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.70)

X_train = np.array(X_train, dtype='float64')
#X_train = X_train.reshape(X_train.shape[1:])

#X_train = X_train.transpose()
X_train = X_train.reshape((-1,1))
y_train = np.array(y_train)

#test arrays
X_test = np.array(X_test,dtype='float64')
X_test = X_test.reshape((-1,1))
y_test = np.array(y_test)


# instantiate a logistic regression model, and fit with X and y
lr_model = LogisticRegression()
lr_model = lr_model.fit(X_train, y_train)
print(lr_model.predict_proba(X)[:,1])

#Accuracy
y_pred = lr_model.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_model.score(X_test, y_test)))

#k-fold validation
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

#confusion matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print ("Confusion Matrix: ")
print (confusion_matrix)

#matplotlib scatter funcion w/ logistic regression

plt.figure(1, figsize=(5, 4))

#plt.scatter(X,lr_model.predict(X))
plt.scatter(X_train,lr_model.predict_proba(X_train)[:,1])
plt.xlabel("Scores")
plt.ylabel("Prediction")
plt.show()

