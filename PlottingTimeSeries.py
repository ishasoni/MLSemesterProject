import csv
import codecs
import pandas as pd
import matplotlib.pyplot as plt

#class object for tweets sentiment data
class Tweet:
	#def __init__(self, id, phrase, time, sentiment, perc):
	def __init__(self, time, sentiment, perc):
			#self.ID = id	
			#self.Phrase = phrase
			self.Timestamp = time
			self.Sentiment = sentiment
			self.Perc = perc

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
def getDataFromFile(file_name):
    
	#list of tweet objects 
	tweet_list = []
	tweetC_list = [] # for plotting

	data = pd.read_csv(file_name, sep = ",", header=None)

	#sort by date and sentiment
	data = data.sort_values(by=[0, 1])	

	#group the data by date and sentiment to get count of occurrances
	group_data = data.groupby([0, 1]).size()

	#build the "cluster"/grouped by object
	for index, val in group_data.iteritems():
		tweetC = Tweet_cluster(index[0], index[1], val)
		tweetC_list.append(tweetC)


	#iterate through the rows to list of tweet objects
	for index, row in data.iterrows():
		#print( row[0], row[1], row[2])
		tweet = Tweet(row[0], row[1], row[2])
		tweet_list.append(tweet)

	return tweet_list, tweetC_list

def plotScatterChart(tweetC_list):
	# for plotting the chart
	y_values = [x.Timestamp for x in tweetC_list]
	x_values = [x.Count for x in tweetC_list]
	c_value = [x.Color for x in tweetC_list]

	plt.scatter(y_values, x_values, c=c_value)
	plt.legend(loc='Date') 
	plt.show()

def main():
	file_name = "/bitcoins_for_plotting.csv"
	
	tweet_list, tweetC_list = getDataFromFile(file_name)

	#first plot the chart
	plotScatterChart(tweetC_list)


main()
