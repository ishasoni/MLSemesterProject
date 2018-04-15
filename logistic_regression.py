## References:
## https://www.quantinsti.com/blog/machine-learning-logistic-regression-python/
##


print ('Begin LogisticRegression...')
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
#from patsy import dmatrices
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score


#load tweetData
tweet_fields = ['Date', 'Sentiment', 'Score']
btc_fields = ['Date','Price']
tweetData = pd.read_csv("bitcoins_for_plotting.csv",header=None)
btcData = pd.read_csv("bitcoin_price.csv", header=None)
tweetData.columns = tweet_fields
btcData.columns = btc_fields

tweetData['Date'] = pd.to_datetime(tweetData['Date'])#.dt.strftime("%Y-%m-%d")
btcData['Date'] = pd.to_datetime(btcData['Date'])#.dt.strftime("%Y-%m-%d")

#Dictionary for holding EOD BTC price for each Date
bitcoinPrices = dict(zip(btcData['Date'],btcData['Price']))

#Dictionary for holding sentiments and its weights
sentimentWeights = {'negative': 0, 'positive': 1, 'neutral': 2}

#add new column Price to twitter dataset - assign bitcoin prices to each date
tweetData['Price'] = tweetData['Date'].map(bitcoinPrices)

#add new column Weight to twitter dataset - assign weight to each sentiment
tweetData['Weight'] = tweetData['Sentiment'].map(sentimentWeights)



# #2D array of X,y coordinates to feed the Logistic Regression model
X = np.array([tweetData.Date.values, tweetData.Weight.values, tweetData.Score.values])
print (X)

#the next day
#nextDay = tweetData['Date'] + datetime.timedelta(days=1)
#currentDay = tweetData.Date
#print ("Next Day: " + nextDay.dt.strftime("%Y-%m-%d") + " , Current Day: " + currentDay.dt.strftime("%Y-%m-%d"))


#If the next day's price is higher than current day's price, then btc price goes up (1), else price goes down (-1).
y = np.where ( tweetData.Price.shift(1) > tweetData.Price, 1, -1)



# #X = X.reshape(X.shape[1:])
X = np.transpose(np.matrix(X))
#print(X.shape)
#print(y.shape)

## split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state=25)

X_train = np.array(X_train)
# #X_train = X_train.reshape(X_train.shape[1:])

# #X_train = X_train.transpose()
#X_train = X_train.reshape((-1,1))
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)

#test arrays
X_test = np.array(X_test)
#X_test = X_test.reshape((-1,1))
y_test = np.array(y_test)
print(X_test.shape)
print(y_test.shape)


##instantiate a logistic regression model, and fit with X and y
lr_model = LogisticRegression()
lr_model = lr_model.fit(X_train, y_train)
#(lr_model.predict_proba(X)[:,1])

##examine coefficients
print("Coefficiens: -------\n", lr_model.coef_)

##class probabilities
probability = lr_model.predict_proba(X_test) #predicting class labels - whether it will be 1 or -1
print ("Probabilties: -----\n", probability)


##Accuracy
predicted = lr_model.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_model.score(X_test, y_test)))

##k-fold validation
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

##confusion matrix
confusion_matrix = confusion_matrix(y_test, predicted)
print ("Confusion Matrix: ")
print (confusion_matrix)

##classification report
print (classification_report (y_test, predicted))


##matplotlib scatter funcion w/ logistic regression

plt.figure(1, figsize=(5, 4))

#plt.scatter(y_test,predicted)
plt.scatter(X_train,lr_model.predict_proba(X_train)[:,1])
# plt.xlabel("Scores")
# plt.ylabel("Prediction")
plt.show()

