import csv
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression
import sklearn

#class object for tweets sentiment data
class Tweet:
	#def __init__(self, id, phrase, time, sentiment, perc):
         #self.ID = id	
        #self.Phrase = phrase
    def __init__(self, time, sentiment, perc, price):
        #self.Timestamp = time
        self.Timestamp = datetime.strptime(time, "%Y-%m-%d")
        self.Sentiment = sentiment
        self.Perc = perc
        self.Price = price

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

# function responsible for opening file and reading file content
def getDataFromFile(tweet_file_name, bitcoin_file_name):
    
    #list of tweet objects 
    tweet_list = []
    tweetC_list = [] # for plotting

    #data = pd.read_csv(tweet_file_name, sep = ",", header=None)
    tweetData = pd.read_csv(tweet_file_name, sep = ",", names=['Date', 'Sentiment', 'Perc'])
    btcData = pd.read_csv(bitcoin_file_name, sep = ",", names=['Date', 'Price'])
 
    ## sort by date and sentiment
    tweetData = tweetData.sort_values(by=['Date', 'Sentiment'])	
    btcData = btcData.sort_values(by=['Date'])	

    ## group the data by date and sentiment to get count of occurrances
    group_data = tweetData.groupby(['Date', 'Sentiment']).size()

    #build the "cluster"/grouped by object
    for index, val in group_data.iteritems():
    	tweetC = Tweet_cluster(index[0], index[1], val)
    	tweetC_list.append(tweetC)

    #iterate through the rows to list of tweet objects
    for index, row in tweetData.iterrows():
	    #print( row[0], row[1], row[2])

        ##TO-DO: Determine a better way to assign price to tweet object, currently slow
        for index2, row2 in btcData.iterrows():
            if datetime.strptime(row2[0], "%Y-%m-%d") == datetime.strptime(row[0], "%Y-%m-%d"):
	            tweet = Tweet(row[0], row[1], row[2], row2[1])
	            tweet_list.append(tweet)

    return tweet_list, tweetC_list, tweetData

def plotScatterChart(tweetC_list):
	# for plotting the chart
	x_values = [x.Timestamp for x in tweetC_list]
	y_values = [x.Count for x in tweetC_list]
	c_value = [x.Color for x in tweetC_list]

	plt.scatter(x_values, y_values, c=c_value)
	plt.legend(loc='Date') 
	plt.show()


def implementBaysianRegression(tweet_list):

    ## #############################################################################
    ## X : numpy array of shape [n_samples, n_features]
    ##    the features that are contributing -- in the date and sentiment perc.
    ## y : numpy array  of shape [n_samples]
    ##    Target values -- this is what we are trying to evaluate/predict 
    ##    the Bitcoin price

    y = [x.Price for x in tweet_list]
    y = np.array(y)
    y = np.sort(y)

    X = [[x.Timestamp.day for x in tweet_list], [x.Perc for x in tweet_list]]  
    #X = [x for i,x in enumerate(tweet_list) if i !=3 ]
    X = np.transpose(np.matrix(X))
    X = np.sort(X)

    ## #############################################################################
    ## Split training/test data (70% training data)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.70, random_state = 5)

    ## #############################################################################
    ## Fit the Bayesian Ridge Regression and an Ordinary least squares linear regression
    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, Y_train)

    #ols = LinearRegression()
    ols = LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    ols.fit(X_train, Y_train)

    #predit 
    pred_train = ols.predict(X_train)
    pred_test = ols.predict(X_test) 

    ## --> see the mean squared error -- since we have only one feature -- we shouldn't be too concerned with this
    #print("Fit a model X_train, and calculate MSE with Y_train: ", np.mean((Y_train - ols_train.predict(X_train)) ** 2) )
    #print("Fit a model X_train, and calculate MSE with X_test, Y_test: ", np.mean((Y_test - ols_train.predict(X_test)) ** 2) )

    ## --> based off COEF value it looks like sentiment has a better chance of predicting price, then date
    ## which is the feature we're concerned with
    #print("COEF IS --> ", clf.coef_)
    #print("OLS COEF IS --> ", ols.coef_)

    ## --> Creating a risidual graph of the training and test data to see the distro
    #plt.scatter(ols.predict(X_train), ols.predict(X_train) - Y_train, c='b', s=40, alpha=0.5)
    #plt.scatter(ols.predict(X_test), ols.predict(X_test) - Y_test, c='g', s=40)
    #plt.title('Residual Plot using training(blue) and test(green) data')
    #plt.ylabel('Residuals')
    #plt.show()

    return ols
    

def main():
    # Enter the location of the CSV Datasets 
    tweet_file_name = "D:\Twitter_Dataset\\bitcoins_for_plotting.csv"
    btc_file_name = "D:\Twitter_Dataset\\bitcoin_price.csv"
    
    ## Gather the data from the files into different object types
    tweet_list, tweetC_list, df = getDataFromFile(tweet_file_name, btc_file_name)

    # Implement the various algorithms
    linearBaysian = implementBaysianRegression(tweet_list)
    
    # Logistic regression goes here 
   
    # SVM Goes here 

    # Sabreen To-Do: Find a generic way to plot all the regressions here 
    
    ## first plot the chart of sentiment distribution 
    #plotScatterChart(tweetC_list)

main()
