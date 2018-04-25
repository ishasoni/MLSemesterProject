import csv
import codecs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import sklearn
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
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

    #y = [x.Price for x in tweet_list] ## Use Price variable when trying to predict the BTC price
    y = [x.upOrDownPrice for x in tweet_list] ## Use upOrDownPrice variable when trying to predict the BTC fluctuation
    y = np.array(y)
    y = np.sort(y)

    #X = [[x.Timestamp.day for x in tweet_list], [x.Score for x in tweet_list], [x.sentimentWeight for x in tweet_list]]  
    #X = [[x.Timestamp.day for x in tweet_list], [x.Score for x in tweet_list]]
    X = [[x.Score for x in tweet_list]]
    X = np.transpose(np.matrix(X))
    X = np.sort(X)

    ## #############################################################################
    ## Split training/test data (70% training data)
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.95)

    #print("X train shape, ", X_train.shape)
    #print("X test shape, ", X_test.shape)
    #print("y train shape, ", Y_train.shape)
    #print("y test shape, ", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


def implementLinearBaysianRegression(X_train, X_test, Y_train, Y_test):

    ## #############################################################################
    ## Fit the Bayesian Ridge Regression and an Ordinary least squares linear regression

    print("#############################################################################")
    print("Fitting Linear Basysian Regression\n\n")
    
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


    score = np.squeeze(np.asarray(X_test[:, 0]))

    #print("this is the prediction for test dataset: ")
    #print(y_pred_test)

    ## classification report -- Doesn't work due to 
    #print("The training classification report:")
    #print(classification_report(Y_test, y_pred_test))

    #print("The testing classification report:")
    #print (classification_report (Y_test, y_pred_test))

    ## --> Creating a risidual graph of the training and test data to see the distro
    #plt.scatter(ols.predict(X_train), ols.predict(X_train) - Y_train, c='b')
    #plt.scatter(ols.predict(X_test), ols.predict(X_test) - Y_test, c='g')
    #plt.hlines(y = 0, xmin = 9000, xmax = 17000)
    plt.scatter(score, y_pred_test, c='g')
    plt.title('BLR: Residual Plot using training(blue) and test(green) data')
    plt.ylabel('Fluctuation')
    plt.xlabel('Sentiment Score')
    plt.show()


    return ols, y_pred_train, y_pred_test


def implementLogisticRegression(X_train, X_test, Y_train, Y_test):

    ## #############################################################################
    ##logistic regression testing

    ols = LogisticRegression()
    ols = ols.fit(X_train, Y_train)

    ## --> Testing
    confidence = ols.score(X_test, Y_test)
    print("Testing confidence: ", confidence)
    
    ## --> based off COEF value (mean of distribution) it looks like sentiment has a better chance of predicting price, here i'm comparing the 
    ##     percentage of sentiment versus the "weights of the sentiment"
    print("OLS COEF is --> ", ols.coef_)

    print("OLS INTERCEPT is --> ", ols.intercept_)

     ## class probabilities
    probability = ols.predict_proba(X_test) 
    print("OLS Probability is --> ", probability)

    ## predict class labels
    predicted = ols.predict(X_test)


    print("The testing classification report:")
    print (classification_report (Y_test, predicted))
    
    confusionMatrix = confusion_matrix(Y_test, predicted)
    print ("Confusion Matrix: ", confusionMatrix)

    score = np.squeeze(np.asarray(X_test[:, 0]))
    probPriceDecrease = np.squeeze(np.asarray(probability[:, 0]))
    probPriceIncrease = np.squeeze(np.asarray(probability[:, 1]))

    ## --> Creating a risidual graph of the training and test data to see the distro
    #plt.plot(score, probPriceIncrease, c='g', label="prob Decrease")
    #plt.plot(score, probPriceDecrease, c='b', label="prob Increase")
   
    # Two subplots, unpack the axes array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(score, probPriceDecrease, c='b', label="prob Increase")
    ax1.plot(score, probPriceIncrease, c='g', label="prob Decrease")
    #ax1.ylabel('Probability')
    #ax1.xlabel('January 2018')
    ax1.set_title("Logistic Regression: Probablity of Fluctuation Vs Time")
    ax2.scatter(score, probPriceDecrease, c='b', label="prob Increase")
    ax2.scatter(score, probPriceIncrease, c='g', label="prob Decrease")
    #ax2.xlabel('January 2018')
    #ax2.ylabel('Probability')
    
    plt.legend(loc='best')
    plt.show()

def implementPolyBaysianRegression(X_train, X_test, Y_train, Y_test):
   
    print("#############################################################################")
    print("Fitting Polynomial Basysian Regression\n\n")

    # PolynomialFeatures (prepreprocessing)
    poly = PolynomialFeatures(degree=4)
    X_ = poly.fit_transform(X_train)
    X_test_ = poly.fit_transform(X_test)

    # Instantiate
    lg = LinearRegression()

    # Fit
    lg.fit(X_, Y_train)

    # Obtain coefficients
    lg.coef_
    
    print("COEF is --> ", lg.coef_)
    #print("Score is --> ", lg.score)

    ## --> Testing
    confidence = lg.score(X_, Y_train)
    print("Testing confidence: ", confidence)


    ## predit 
    y_pred_test = lg.predict(X_test_)

    ## --> see the mean squared error -- since we have only one feature -- we shouldn't be too concerned with this  
   #print("Fit model X_train, and calculate MSE with Y_train: ", np.mean((Y_train - lg.predict(X_train)) ** 2) )
   #print("Fit model X_train, and calculate MSE with X_test, Y_test: ", np.mean((Y_test - lg.predict(X_test)) ** 2) )
   
    print("using mean squared error function: ")
    regression_model_mse = mean_squared_error(y_pred_test, Y_test)
    print(math.sqrt(regression_model_mse)) #getting the sqrt gives us how likely the predictio is far from the true value


    print("this is the prediction")
    print(y_pred_test)
    print("prediction size --> ", y_pred_test.shape)


    ## --> Creating a risidual graph of the training and test data to see the distro
    #plt.scatter(y_pred_test, y_pred_test - Y_test, c='b')
    #plt.title('BPR: Residual Plot using training(blue)')
    #plt.ylabel('Residuals')
    #plt.show()

    return y_pred_test

    
def implementSVM(X_train, X_test, Y_train, Y_test, tweet_list):

    ## #############################################################################
    ## Fit SVM regression models
    print("#############################################################################")
    print("Fitting SVM Regressions\n\n")    

    #svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1, cache_size = 8000)
    #svr_lin = SVR(kernel='linear', C=1e3, cache_size = 8000)
    svr_poly = SVR(kernel='poly', C=1e3, degree=4, cache_size = 10000)
    
    #svr_rbf.fit(X_train, Y_train)
    #svr_lin.fit(X_train, Y_train)
    svr_poly.fit(X_train, Y_train)

    #rbf_prediction = svr_rbf.predict(X_train)
    #print("rbf score is --> ", svr_rbf.score(X_train, Y_train))
    #print("RBF prediction = " + str(rbf_prediction))
    
    #lin_prediction = svr_lin.predict(X_train)
    #print("Lin score is --> ", svr_lin.score(X_train, Y_train))
    #print("Lin prediction = " + str(lin_prediction))
    
    poly_prediction = svr_poly.predict(X_train)
    print("Poly score is --> ", svr_poly.score(X_train, Y_train))
    print("Poly prediction = " + str(poly_prediction))

    score = np.squeeze(np.asarray(X_train[:, 0]))
    

    ##Draw black dots representing prices
    #plt.plot(X_train, lin_prediction, color = 'blue', linewidth = 3, label = 'Linear Model')
    
    plt.plot(X_train, poly_prediction, color = 'green', linewidth = 3, label = 'Polynomial Model')
    plt.title('SVM: Polynomial Regression Plot')

    #plt.plot(X_train, rbf_prediction, color = 'red', linewidth = 4, label = 'RBF Model')
    #plt.title('SVM: RBF Regression Plot')
    plt.xlabel("Sentiment Score")
    plt.ylabel("Fluctuation")
    plt.show()


    ## classification report
    print("The classification report:")
    #print (classification_report (Y_train, rbf_prediction))
    #print (classification_report (Y_train, lin_prediction))
    print (classification_report (Y_train, poly_prediction))

    ## return required variable here once ready 
    return ""

def plotSentimentandPrice(tweetC_list, bitcoinPrices):
    ## for plotting sentiment
    x_values = [str(x.Timestamp.day) for x in tweetC_list]
    #x_values = [str(x.Timestamp) for x in tweetC_list]
    y_values = [x.Count for x in tweetC_list]
    c_value = [x.Color for x in tweetC_list]

    # Two subplots
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(list(bitcoinPrices.keys()),list(bitcoinPrices.values()), label='BTC Price')
    axarr[0].set_title('Bitcoin Prices')
    axarr[1].scatter(x_values, y_values, c = c_value, label= 'Sentiment Distro')
    axarr[1].set_title('Sentiment Distro')
    
    plt.xticks(rotation=90) 
    plt.show()


def plotPriceLinearBaysian(X_train, X_test, Y_train, lb_pred_train, lb_pred_test, bitcoinPrices, tweetC_list):

    training_dates = np.squeeze(np.asarray(X_train[:, 1]))
    #testing_dates = np.squeeze(np.asarray(X_test[:, 0 ]))

    ## for plotting sentiment
    x_values = [str(x.Timestamp.day) for x in tweetC_list]
    y_values = [x.Count for x in tweetC_list]
    c_value = [x.Color for x in tweetC_list]

    # Two subplots
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(list(bitcoinPrices.keys()),list(bitcoinPrices.values()), label='BTC Price')
    axarr[0].set_title('Bitcoin Prices and Linear Baysian ')
    axarr[1].scatter(x_values, y_values, c = c_value)
    axarr[2].scatter(training_dates, lb_pred_train - Y_train, c='g')
    plt.legend(loc='January 2018')
    plt.show()

    #Initial plotting here...
    ##plt.plot(training_dates, lb_pred_train, 'r')
    #plt.plot(testing_dates, lb_pred_test, 'g')
    #plt.scatter(lb_pred_train, lb_pred_train - Y_train, c='b')
    #plt.scatter(training_dates, lb_pred_train - Y_train, c='b')
    #plt.plot(list(bitcoinPrices.keys()),list(bitcoinPrices.values()))
    #plt.legend(loc='Date') 
    #plt.xlabel('January 2018')
    #plt.show()

def plotPricePolynomialBaysian(X_train, X_test, Y_test, y_pred_test, bitcoinPrices, tweetC_list):

    training_dates = np.squeeze(np.asarray(X_train[:, 1]))
    testing_dates = np.squeeze(np.asarray(X_test[:, 1 ]))

    ## for plotting sentiment
    x_values = [x.Timestamp.day for x in tweetC_list]
    y_values = [x.Count for x in tweetC_list]
    c_value = [x.Color for x in tweetC_list]

    # Two subplots
    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(list(bitcoinPrices.keys()),list(bitcoinPrices.values()), label='BTC Price')
    axarr[0].set_title('Bitcoin Prices and Linear Baysian ')
    axarr[1].scatter(x_values, y_values, c = c_value)
    axarr[2].plot(testing_dates, y_pred_test, 'g')
    #plt.xaxis.set_major_formatter(myFmt)
    plt.legend(loc='January 2018')
    plt.show()

## method responsible for writing the list of Tweet Objects to CSV file 
## this is to transport the dataset for testing in WEKA
def writeWekaCSVFile(tweet_list):

    with open("D:\Twitter_Dataset\\bitcoins_for_plotting_weka.csv", "w") as f:
        f.write('Timestamp, Score, upOrDownPrice\n')
        for tweet in tweet_list:
            f.write(str(tweet.Timestamp) + ', ' + str(tweet.Score) + ',' + str(tweet.upOrDownPrice) + '\n')


## main function
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

    # write to external file
    #writeWekaCSVFile(tweet_list)

    ##----------------------------------------------------------------------------------
    ## Implement the various algorithms for price predictions
    
    ## Linear Baysian Algorithm
    #linearBaysian, lb_pred_train, lb_pred_test = implementLinearBaysianRegression(X_train, X_test, Y_train, Y_test)

    ## Polynomial Baysian Algorithm
    #y_pred_test = implementPolyBaysianRegression(X_train, X_test, Y_train, Y_test)

    ## SVM
    SVM = implementSVM(X_train, X_test, Y_train, Y_test, tweet_list)    
    
    # Logistic Regression
    #implementLogisticRegression(X_train, X_test, Y_train, Y_test)
    

    ##----------------------------------------------------------------------------------
    ## Plot various data and regressions here
    
    #plotSentimentandPrice(tweetC_list, bitcoinPrices)
    #plotPriceLinearBaysian(X_train, X_test, Y_train, lb_pred_train, lb_pred_test, bitcoinPrices, tweetC_list)
    ##plotPricePolynomialBaysian(X_train, X_test, Y_test, y_pred_test, bitcoinPrices, tweetC_list)
    
    print("Done!")

main()
