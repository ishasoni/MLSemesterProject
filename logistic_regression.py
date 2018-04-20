## References:
## https://www.quantinsti.com/blog/machine-learning-logistic-regression-python/
##

print ('Begin LogisticRegression...')
import numpy as np
import pandas as pd
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
#from patsy import dmatrices
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score


def loadData(filename1, filename2):
    tweet_fields = ['Date', 'Sentiment', 'Score']
    btc_fields = ['Date','Price']
    tweetData = pd.read_csv(filename1,header=None)
    btcData = pd.read_csv(filename2, header=None)
    tweetData.columns = tweet_fields
    btcData.columns = btc_fields

    tweetData['Date'] = pd.to_datetime(tweetData['Date'])#.dt.strftime("%Y-%m-%d")
    btcData['Date'] = pd.to_datetime(btcData['Date']).dt.strftime("%Y-%m-%d")

    #Dictionary for holding EOD BTC price for each Date - used from Sabreen's code
    bitcoinPrices = dict(zip(btcData['Date'],btcData['Price']))
    print(bitcoinPrices.get("2018-01-05"))

    #Dictionary for holding sentiments and its weights
    sentimentWeights = {'negative': 0, 'positive': 1, 'neutral': 2}

    #add new column Price to twitter dataset - assign bitcoin prices to each date
    tweetData['Price'] = tweetData['Date'].map(bitcoinPrices)

    #add new column Weight to twitter dataset - assign weight to each sentiment
    tweetData['Weight'] = tweetData['Sentiment'].map(sentimentWeights)


    ## If the current day's price is higher than previous day's price, then btc price goes up (1), else price goes down (-1).
    class_values = [] #temp array that holds indicators of whether each price goes up or down

    for index, row in tweetData.iterrows():
        previousDay = (row['Date'] - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        currentDay = (row.Date).strftime("%Y-%m-%d")
        if (bitcoinPrices.get(currentDay,0) > bitcoinPrices.get(previousDay,0)):
            class_values.append(1)
        else:
            class_values.append(-1)
        

    ## 2D array of X,y coordinates to feed the Logistic Regression model

    #convert tweet dates to numeric to fit the model
    tweetData['Date'] = tweetData['Date'].apply(lambda x: x.toordinal()) 
    X = np.array([tweetData.Date.values, tweetData.Weight.values, tweetData.Score.values])
    # #X = X.reshape(X.shape[1:])
    X = np.transpose(np.matrix(X))

    y = np.array([i for i in class_values])
    
    return X, y

def getTrainingandTestingData(X,y):
    ## split dataset
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state=5)

    #X[0] = np.array(X[0], dtype ='datetime64[D]') 

    print ("x -->",X.shape)
    print ("Y -->",y.shape)
    print ("x_train -->",X_train.shape)
    print ("y_train -->",y_train.shape)
    print ("x_test -->",X_test.shape)
    print ("y_test -->",y_test.shape)

    return X_train, X_test, y_train, y_test

def logisticRegression(X,y):

    X_train, X_test, y_train, y_test = getTrainingandTestingData(X,y)
    ## instantiate a logistic regression model, and fit with X and y
    lr_model = LogisticRegression()
    lr_model = lr_model.fit(X_train, y_train)
    #(lr_model.predict_proba(X)[:,1])

    # X_train[:,0] = np.array(X_train[:,0], dtype ='datetime64[D]')
    # X_test[:,0] = np.array(X_test[:,0], dtype ='datetime64[D]')

    
    ## examine coefficients
    print("Coefficients: -------\n", lr_model.coef_)

    ## class probabilities
    probability = lr_model.predict_proba(X_test) #predicting class labels - whether it will be 1 or -1
    np.savetxt('probabilities.txt', probability) #temporarily to see results

    ## predict class labels
    predicted = lr_model.predict(X_test)
    np.savetxt('predicted.txt', predicted) #temporarily to see results
    #classif_rate = np.mean(predicted.ravel() == y.ravel()) * 100

    pred_log = lr_model.predict_log_proba(X_test)
    np.savetxt('pred_log.txt',pred_log)
	
    ## confusion matrix
    confusionMatrix = confusion_matrix(y_test, predicted)
    print ("Confusion Matrix: ", confusionMatrix)


    ## classification report
    print (classification_report (y_test, predicted))

    ## Accuracy
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr_model.score(X_test, y_test)))

    ## k-fold validation
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    modelCV = LogisticRegression()
    scoring = 'accuracy'
    results = model_selection.cross_val_score(modelCV, X, y, cv=kfold, scoring=scoring)
    print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

    return lr_model
    
def convert_ordinal_to_datetime(X):
    ####    converts the values of an array back to datetime for plotting ( this was needed to adjust with matplotlib handling and plotting dates)

    temp = np.array([datetime.datetime.fromordinal(x) for x in X])
    return temp
    
def get_x_y_axes(a, b):
    ### params: a and b are 1-D arrays of datetime and fluctuation value, respectively
    ## returns values for X and Y axis for plotting, given the array
    
    df = pd.DataFrame({'date':a,'upOrDown':b})
    df = df.drop_duplicates('date')
    df = df.sort_values(by='date')
    xaxis = np.array(df.date.values)
    yaxis = np.array(df.upOrDown.values)

    return xaxis,yaxis

def set_plot(x_axis):
    ###
    ## plot tweet dates vs fluctuation (upOrDown values)

    #set limits for x-axis
    plt.xlim(x_axis[0],x_axis[-1])

    ax = plt.gca()
    xaxis = mdates.date2num(x_axis)
    hfmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_formatter(hfmt)

    plt.xticks(x_axis, visible=True,rotation=50,ha="right",rotation_mode='anchor')

    return plt

    
    ##plt.figure(1, figsize=(7,5))
    #plt.scatter(y_test,predicted)
    #plt.scatter(X_train,lr_model.predict_proba(X_train)[:,1], c='b')
    ##plt.xticks(X_train[:,0], y_train, c = 'b')
    #plt.autoscale(enable=True, axis='x', tight=True)
    ##plt.scatter(X_test[:,0], y_test, c = 'r')
    # plt.xlabel("Scores")
    # plt.ylabel("Prediction")
    ##plt.show()

def plotDatesVsFluctuation(X_train, X_test, y_train, y_test, model):

    ##matplotlib scatter funcion w/ logistic regression
    #mpl.rcParams['agg.path.chunksize'] =  1000000000

    x_train_dates = convert_ordinal_to_datetime(X_train[:,0])
    
    x_test_dates = convert_ordinal_to_datetime(X_test[:,0])

    xaxis, yaxis = get_x_y_axes(x_train_dates,y_train)
    
    xtest, ytest = get_x_y_axes(x_test_dates,y_test)
    
    plt = set_plot(xaxis)
    plt = set_plot(xtest)
    plt.plot(xaxis, yaxis, 'b-', label='Train')
    plt.plot(xtest, ytest, 'r:', label='Test')

   # plt.scatter(xaxis, yaxis, c = 'b', label='Train')
   # plt.scatter(xtest, ytest, c ='r', label='Test')
    plt.xlabel("Tweet Dates")
    plt.ylabel("Fluctuation")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plotPrediction(x_train, x_test, model):

    ##matplotlib scatter funcion w/ logistic regression
    mpl.rcParams['agg.path.chunksize'] =  1000000000

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    
    x_train_dates = convert_ordinal_to_datetime(x_train[:,0])   
    xtrain,ytrain = get_x_y_axes(x_train_dates,train_pred)
    
    x_test_dates = convert_ordinal_to_datetime(x_test[:,0])
    xtest,ytest = get_x_y_axes(x_test_dates,test_pred)
   
    np.savetxt('train_pred.txt', train_pred) #temporarily to see results
    np.savetxt('test_pred.txt', test_pred)
    print("train_pred --->",train_pred)
    print("test_pred ---->",test_pred)
    
    plt = set_plot(xtrain)
    plt = set_plot(xtest)

    
    #plotting weights and class probabilities
    plt.plot(xtrain, ytrain, 'b-', label='Train')
    plt.plot(xtest, ytest, 'r:', label='Test')
   # plt.plot(a[:,1], probability[:,0], 'y-', label='Price Decrease')
   # plt.scatter(xaxis, yaxis, c = 'b', label='Train')
   # plt.scatter(xtest, ytest, c ='r', label='Test')
    plt.xlabel("Tweet Dates")
    plt.ylabel("Prediction")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def main():
    
    filename1 = "bitcoins_for_plotting_new.csv"
    filename2 = "bitcoin_price_new.csv"
    
    ## load data and preprocessing
    X,y = loadData(filename1,filename2)
    
    ## split data
    X_train, X_test, y_train, y_test = getTrainingandTestingData(X,y)
    
    ## implement logistic regression and evaluate the model
    lr_model = logisticRegression(X,y)

    ## plotting probability (up or down)
    plotPrediction(X_train, X_test, lr_model)
    
    
    ## plotting fluctuation
   # plotDatesVsFluctuation(X_train, X_test, y_train, y_test, lr_model)


if __name__ == "__main__": 
    main()
