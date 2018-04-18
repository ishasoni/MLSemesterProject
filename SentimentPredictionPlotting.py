import csv
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

#class object for tweets sentiment data
class Tweet:
    #def __init__(self, time, sentiment, score, price, upOrDown):
    def __init__(self, time, sentiment, score, price, upOrDown):
        #self.Timestamp = time
        self.Timestamp = datetime.strptime(time, "%Y-%m-%d")
        #self.Sentiment = sentiment
        self.Score = score
        self.Price = price
        self.sentimentWeight = self.sentimentWeight(sentiment)
        self.upOrDownPrice = upOrDown

    ## Giving weights to the sentiment because scikit learn doesn't read non numerical data for sentiment
    def sentimentWeight(self, sentiment):
	    if sentiment == "negative":
		    weight = 0
	    elif sentiment == "positive":
		    weight = 1
	    else:
		    weight = 2 # neutral == 2

	    return weight

class Tweet_cluster:
    def __init__(self, time, sentiment, count):
        self.Timestamp = datetime.strptime(time, "%Y-%m-%d")
        self.Sentiment = sentiment
        self.Count = count
        self.Color = self.color(sentiment)

    def color(self, sentiment):
	    if sentiment == "negative":
		    Color = 'r'
	    elif sentiment == "positive":
		    Color = 'b'
	    else:
		    Color = 'y'

	    return Color

## Function responsible for opening file and reading file content
def getDataFromFile(tweet_file_name, bitcoin_file_name):
    
    #list of tweet objects 
    tweet_list = []
    tweetC_list = [] # for plotting

    tweetData = pd.read_csv(tweet_file_name, sep = ",", names=['Date', 'Sentiment', 'Perc'])
    btcData = pd.read_csv(bitcoin_file_name, sep = ",", names=['Date', 'Price'])
    btcData['Date'] = pd.to_datetime(btcData['Date']).dt.strftime("%Y-%m-%d")

    ## sort dataframes by date and sentiment
    tweetData = tweetData.sort_values(by=['Date', 'Sentiment'])	
    btcData = btcData.sort_values(by=['Date'])

    ## Dictionary for holding EOD BTC price for each Date
    bitcoinPrices = dict(zip(btcData['Date'],btcData['Price']))

    ## group the data by date and sentiment to get count of occurrances
    group_data = tweetData.groupby(['Date', 'Sentiment']).size()

    ##build the "cluster"/grouped by object
    for index, val in group_data.iteritems():
    	tweetC = Tweet_cluster(index[0], index[1], val)
    	tweetC_list.append(tweetC)

    ##iterate through the rows to create list of tweet objects
    for index, row in tweetData.iterrows():
	    tweet = Tweet(row[0], row[1], row[2], getPriceValue(row[0], bitcoinPrices), getUpOrDownValue(row[0], bitcoinPrices))
	    tweet_list.append(tweet)
        
    return tweet_list, tweetC_list, bitcoinPrices


## Method for retrieving the BTC price value for the tweets date 
def getPriceValue(tweet_date, bitcoinPrices):

    return bitcoinPrices.get(tweet_date, 0) # default of 0


## Method for retrieving the BTC price value for the tweets date 
def getUpOrDownValue(tweet_date, bitcoinPrices):
    #Get previous day
    prevDay = (datetime.strptime(tweet_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # return 1 if current price increased from previous day 
    # return -1 if current price decreased from previous day
    return np.where(bitcoinPrices.get(tweet_date, 0) > bitcoinPrices.get(prevDay, 0), 1, -1)


def splitTestTrainingData(tweet_list):

    ## #############################################################################
    ## X : numpy array of shape [n_samples, n_features]
    ##    the features that are contributing -- in the date and sentiment perc.
    ## y : numpy array  of shape [n_samples]
    ##    Target values -- this is what we are trying to evaluate/predict 
    ##    the Bitcoin price or the 1/-1 value to indicate fluctuation in price

    y = [x.Price for x in tweet_list] ## Use Price variable when trying to predict the BTC price
    #y = [x.upOrDownPrice for x in tweet_list] ## Use upOrDownPrice variable when trying to predict the BTC fluctuation
    y = np.array(y)
    y = np.sort(y)

    #X = [[x.Timestamp.day for x in tweet_list], [x.Score for x in tweet_list], [x.sentimentWeight for x in tweet_list]]  
    X = [[x.Timestamp.day for x in tweet_list], [x.Score for x in tweet_list]]
    X = np.transpose(np.matrix(X))
    X = np.sort(X)

    ## #############################################################################
    ## Split training/test data (70% training data)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.33)

    #print("X train shape, ", X_train.shape)
    #print("X test shape, ", X_test.shape)
    #print("y train shape, ", Y_train.shape)
    #print("y test shape, ", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


def implementLinearBaysianRegression(X_train, X_test, Y_train, Y_test):

    ## #############################################################################
    ## Fit the Bayesian Ridge Regression and an Ordinary least squares linear regression

    print("#############################################################################")
    print("Fitting Linear Basysian Regression")
    
    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, Y_train)

    ## “Least Squares” means that we’re trying to fit a regression line that would minimize the square of distance from the regression line
    ols = LinearRegression(copy_X = True, fit_intercept = True, normalize = False)
    ols.fit(X_train, Y_train)

    ## --> Testing
    confidence = ols.score(X_test, Y_test)
    print("Testing confidence: ", confidence)
    
    ## --> based off COEF value (mean of distribution) it looks like sentiment has a better chance of predicting price, here i'm comparing the 
    ##     percentage of sentiment versus the "weights of the sentiment"
    print("COEF is --> ", clf.coef_)
    print("OLS COEF is --> ", ols.coef_)

    print("INTERCEPT is --> ", clf.intercept_)
    print("OLS INTERCEPT is --> ", ols.intercept_)


    ## --> the coefficient of determination (R^2)
    ##    A measure of how well observed outcomes are replicated by the model, as the proportion of total variation of outcomes explained by the model. 
    ##    Which is basically the accuracy measure for the Linear Regression Model   
    clf_score = clf.score(X_test, Y_test)
    ols_score = ols.score(X_test, Y_test)
    print("SCORE is -->", clf_score)
    print("OLS SCORE is -->", ols_score)

    ## predit 
    y_pred_train = ols.predict(X_train)
    y_pred_test = ols.predict(X_test) 

    ## --> see the mean squared error -- since we have only one feature -- we shouldn't be too concerned with this  
    print("Fit model X_train, and calculate MSE with Y_train: ", np.mean((Y_train - ols.predict(X_train)) ** 2) )
    print("Fit model X_train, and calculate MSE with X_test, Y_test: ", np.mean((Y_test - ols.predict(X_test)) ** 2) )

    print("using mean squared error function: ")
    regression_model_mse = mean_squared_error(y_pred_test, Y_test)
    print(math.sqrt(regression_model_mse)) #getting the sqrt gives us how likely the prediction is far from the true value


    print("this is the prediction for test dataset: ")
    print(y_pred_test)
    #print("prediction size --> ", y_pred_test.shape)


    ## --> Creating a risidual graph of the training and test data to see the distro
    #plt.scatter(ols.predict(X_train), ols.predict(X_train) - Y_train, c='b')
    #plt.scatter(ols.predict(X_test), ols.predict(X_test) - Y_test, c='g')
    ##plt.hlines(y = 0, xmin = 9000, xmax = 17000)
    #plt.title('BLR: Residual Plot using training(blue) and test(green) data')
    #plt.ylabel('Residuals')
    #plt.show()

    return ols, y_pred_train, y_pred_test

def implementPolyBaysianRegression(X_train, X_test, Y_train, Y_test):
   
    print("#############################################################################")
    print("Fitting Polynomial Basysian Regression")

    # PolynomialFeatures (prepreprocessing)
    poly = PolynomialFeatures(degree=7)
    X_ = poly.fit_transform(X_train)
    X_test_ = poly.fit_transform(X_test)

    # Instantiate
    lg = LinearRegression()

    # Fit
    lg.fit(X_, Y_train)

    # Obtain coefficients
    lg.coef_
    
    print("COEF is --> ", lg.coef_)

    ## predit 
    y_pred_train = lg.predict(X_test_)

    ## --> see the mean squared error -- since we have only one feature -- we shouldn't be too concerned with this  
   #print("Fit model X_train, and calculate MSE with Y_train: ", np.mean((Y_train - lg.predict(X_train)) ** 2) )
   #print("Fit model X_train, and calculate MSE with X_test, Y_test: ", np.mean((Y_test - lg.predict(X_test)) ** 2) )
   
   #print("using mean squared error function: ")
   #regression_model_mse = mean_squared_error(y_pred_test, Y_test)
   #print(math.sqrt(regression_model_mse)) #getting the sqrt gives us how likely the predictio is far from the true value


    print("this is the prediction")
    print(y_pred_train)
    print("prediction size --> ", y_pred_train.shape)


    ## --> Creating a risidual graph of the training and test data to see the distro
    #plt.scatter(y_pred_train, y_pred_train)
    ##plt.hlines(y = 0, xmin = 9000, xmax = 17000)
    #plt.title('BLR: Residual Plot using training(blue) and test(green) data')
    #plt.ylabel('Residuals')
    #plt.show()

    return y_pred_train

    
def implementSVM(X_train, X_test, Y_train, Y_test, tweet_list):

    ## #############################################################################
    ## Fit SVM regression models
    print("#############################################################################")
    print("Fitting SVM Regressions")    

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    
    svr_rbf.fit(X_train, Y_train)
    svr_lin.fit(X_train, Y_train)
    svr_poly.fit(X_train, Y_train)

    print("RBF = " + str(svr_rbf.predict(X_train)))
    print("Lin = " + str(svr_lin.predict(X_train)))
    print("Poly = " + str(svr_poly.predict(X_train)))

    ##Draw black dots representing prices
    #plt.plot(X_train, svr_lin.predict(X_train), color = 'blue', linewidth = 3, label = 'Linear Model')
    #plt.plot(X_train, svr_poly.predict(X_train), color = 'green', linewidth = 3, label = 'Polynomial Model')
    #plt.plot(X_train, svr_rbf.predict(X_train), color = 'red', linewidth = 4, label = 'RBF Model')
    #plt.show()

    ## return required variable here once ready 
    return ""

def plotSentimentandPrice(tweetC_list, bitcoinPrices):
    ## for plotting sentiment
    x_values = [x.Timestamp.day for x in tweetC_list]
    y_values = [x.Count for x in tweetC_list]
    c_value = [x.Color for x in tweetC_list]

    # Two subplots
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(list(bitcoinPrices.keys()),list(bitcoinPrices.values()), label='BTC Price')
    axarr[0].set_title('Bitcoin Prices and Sentiment')
    axarr[1].scatter(x_values, y_values, c = c_value)
    plt.legend(loc='January 2018')
    plt.show()

def plotPriceLinearBaysian(X_train, X_test, lb_pred_train, lb_pred_test, bitcoinPrices):

    training_dates = np.squeeze(np.asarray(X_train[:, 1]))
    testing_dates = np.squeeze(np.asarray(X_test[:, 1]))

    ##plt.plot(training_dates, lb_pred_train, 'r')
    plt.plot(testing_dates, lb_pred_test, 'g')
    plt.plot(list(bitcoinPrices.keys()),list(bitcoinPrices.values()))
    plt.legend(loc='Date') 
    plt.xlabel('January 2018')
    plt.show()

def plotPricePolynomialBaysian(X_train, X_test, pb_pred_test, bitcoinPrices):

    training_dates = np.squeeze(np.asarray(X_train[:, 1]))
    testing_dates = np.squeeze(np.asarray(X_test[:, 1]))

    ##plt.plot(training_dates, lb_pred_train, 'r')
    plt.plot(testing_dates, pb_pred_test, 'g')
    plt.plot(list(bitcoinPrices.keys()),list(bitcoinPrices.values()))
    plt.legend(loc='Date') 
    plt.show()
    
def main():
    ## Enter the location of the CSV Datasets 
    tweet_file_name = "D:\Twitter_Dataset\\bitcoins_for_plotting.csv"
    btc_file_name = "D:\Twitter_Dataset\\bitcoin_price.csv"
    
    ##----------------------------------------------------------------------------------
    ## Gather the data from the files into different object types
    tweet_list, tweetC_list, bitcoinPrices = getDataFromFile(tweet_file_name, btc_file_name)

    ##----------------------------------------------------------------------------------
    ## Split the data into testing and training data 
    X_train, X_test, Y_train, Y_test = splitTestTrainingData(tweet_list)

    ##----------------------------------------------------------------------------------
    ## Implement the various algorithms for price predictions
    
    ## Linear Baysian Algorithm
    #linearBaysian, lb_pred_train, lb_pred_test = implementLinearBaysianRegression(X_train, X_test, Y_train, Y_test)

    ## Polynomial Baysian Algorithm
    pb_pred_test = implementPolyBaysianRegression(X_train, X_test, Y_train, Y_test)

    ## SVM
    #SVM = implementSVM(X_train, X_test, Y_train, Y_test, tweet_list)    
    
    # Logistic regression goes here 
   
    ##----------------------------------------------------------------------------------
    ## Plot various data and regressions here
    
    ##plotSentimentandPrice(tweetC_list, bitcoinPrices)
    
    #plotPriceLinearBaysian(X_train, X_test, lb_pred_train, lb_pred_test, bitcoinPrices)

    #plotPricePolynomialBaysian(X_train, X_test, pb_pred_test, bitcoinPrices)
    

    print("Done!")

main()
