import csv
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR

#class object for tweets sentiment data
class Tweet:
    def __init__(self, time, sentiment, score, price, upOrDown):
        #self.Timestamp = time
        self.Timestamp = datetime.strptime(time, "%Y-%m-%d")
        self.Sentiment = sentiment
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
        self.Timestamp = time        
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
        
    return tweet_list, tweetC_list, tweetData


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


def plotScatterChart(tweetC_list):
	# for plotting the chart
	x_values = [x.Timestamp for x in tweetC_list]
	y_values = [x.Count for x in tweetC_list]
	c_value = [x.Color for x in tweetC_list]

	plt.scatter(x_values, y_values, c=c_value)
	plt.legend(loc='Date') 
	plt.show()


def splitTestTrainingData(tweet_list):

    ## #############################################################################
    ## X : numpy array of shape [n_samples, n_features]
    ##    the features that are contributing -- in the date and sentiment perc.
    ## y : numpy array  of shape [n_samples]
    ##    Target values -- this is what we are trying to evaluate/predict 
    ##    the Bitcoin price or the 1/-1 value to indicate fluctuation in price

    #y = [x.Price for x in tweet_list] ## Use Price variable when trying to predict the BTC price
    y = [x.upOrDownPrice for x in tweet_list] ## Use upOrDownPrice variable when trying to predict the BTC fluctuation
    y = np.array(y)
    y = np.sort(y)

    X = [[x.Timestamp.day for x in tweet_list], [x.sentimentWeight for x in tweet_list],  [x.Score for x in tweet_list]]  
    X = np.transpose(np.matrix(X))
    X = np.sort(X)

    ## #############################################################################
    ## Split training/test data (70% training data)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.70, random_state = 5)

    return X_train, X_test, Y_train, Y_test


def implementBaysianRegression(X_train, X_test, Y_train, Y_test):

    ## #############################################################################
    ## Fit the Bayesian Ridge Regression and an Ordinary least squares linear regression
    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, Y_train)

    #ols = LinearRegression()
    ols = LinearRegression(copy_X = True, fit_intercept = True, normalize = False)
    ols.fit(X_train, Y_train)

    ## predit 
    pred_train = ols.predict(X_train)
    pred_test = ols.predict(X_test) 

    ## --> Testing
    confidence = ols.score(X_test, Y_test)
    print("Testing confidence: ", confidence)

    ## --> see the mean squared error -- since we have only one feature -- we shouldn't be too concerned with this
    print("Fit a model X_train, and calculate MSE with Y_train: ", np.mean((Y_train - ols.predict(X_train)) ** 2) )
    print("Fit a model X_train, and calculate MSE with X_test, Y_test: ", np.mean((Y_test - ols.predict(X_test)) ** 2) )

    ## --> based off COEF value it looks like sentiment has a better chance of predicting price, here i'm comparing the 
    ##     percentage of sentiment versus the "weights of the sentiment"
    print("COEF IS --> ", clf.coef_)
    print("OLS COEF IS --> ", ols.coef_)

    ## --> Creating a risidual graph of the training and test data to see the distro
    plt.scatter(ols.predict(X_train), ols.predict(X_train) - Y_train, c='b', s=40, alpha=0.5)
    plt.scatter(ols.predict(X_test), ols.predict(X_test) - Y_test, c='g', s=40)
    plt.title('BLR: Residual Plot using training(blue) and test(green) data')
    plt.ylabel('Residuals')
    plt.show()

    return ols, pred_train, pred_test
    
def implementSVM(X_train, X_test, Y_train, Y_test, tweet_list):

    ## #############################################################################
    ## Fit SVM regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    
    svr_rbf.fit(X_train, Y_train)
    svr_lin.fit(X_train, Y_train)
    svr_poly.fit(X_train, Y_train)

    predict_num = '02' ##int(dates[0]) + 1

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

def main():
    # Enter the location of the CSV Datasets 
    tweet_file_name = "D:\Twitter_Dataset\\bitcoins_for_plotting.csv"
    btc_file_name = "D:\Twitter_Dataset\\bitcoin_price.csv"
    
    ##----------------------------------------------------------------------------------
    ## Gather the data from the files into different object types
    tweet_list, tweetC_list, df = getDataFromFile(tweet_file_name, btc_file_name)

    ##----------------------------------------------------------------------------------
    ## Split the data into testing and training data 
    X_train, X_test, Y_train, Y_test = splitTestTrainingData(tweet_list)

    ##----------------------------------------------------------------------------------
    ## Implement the various algorithms for price predictions
    
    #Linear Baysian Algorithm
    linearBaysian, lb_pred_train, lb_pred_test = implementBaysianRegression(X_train, X_test, Y_train, Y_test)
    
    # Logistic regression goes here 
   
    # SVM
    # TODO: Still working on SVM
    #SVM = implementSVM(X_train, X_test, Y_train, Y_test, tweet_list)

    ##----------------------------------------------------------------------------------
    ## Sabreen To-Do: Find a generic way to plot all the regressions here 
    
    ## first plot the chart of sentiment distribution 
    #plotScatterChart(tweetC_list)

main()
