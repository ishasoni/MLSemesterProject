# MLSemesterProject
Spring 2018
Machine Learning 
NET ID HERE

The following code was constructed to implement the "Detection of public sentiment and its influence on the daily Bitcoin price fluctuations" machine learning project. 

We built code in in OOP manner to collect the twitter data, pre-process the data, train the data, and implement the learning algorithms. 
External libraries where used to compute mathematical equations, construct the learning algorithms, and train the data.

Executing the code will require all code that is not currently commented out needs to stay as it is necessary for execution. 
The only modification that are required to is to uncomment the specific learning algorithm that needs to be tested when needing to execute. 
For example 

   ## Linear Baysian Algorithm
    linearBaysian, lb_pred_train, lb_pred_test = implementLinearBaysianRegression(X_train, X_test, Y_train, Y_test)
    plotPriceLinearBaysian(X_train, X_test, Y_train, lb_pred_train, lb_pred_test, bitcoinPrices, tweetC_list)

this code would need to be uncommented so that the fitting, prediction, and plotting can be executed 
however the remainder of the methods that handle implementing the other learning algorithms will need to be commented out


The two datasets that are to be used for fitting the libraries can be found in the zipped file titled "Datasets_for_Code_Execution"
the two lines of code that indicate the location of the test data will need to point to the local test dateset files:
   ## Enter the location of the CSV Datasets 
    tweet_file_name = "bitcoin_tweets.csv"
    btc_file_name = "bitcoin_price.csv"

