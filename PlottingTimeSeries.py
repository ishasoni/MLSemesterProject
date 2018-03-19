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

# function responsible for opening file and reading file content
def getDataFromFile(file_name):
    
	#list of tweet objects 
	tweet_list = []

	data = pd.read_csv(file_name, sep = ",", header=None)

	#sort by date and sentiment
	data = data.sort_values(by=[0, 1])	

	#group the data by date and sentiment to get count of occurrances
	group_data = data.groupby([0, 1]).size()
	print(group_data) 

	group_data.plot(x=[0], label='model')
	plt.legend(loc='Date')    
	plt.show()

	#for key, grp in data.groupby([0, 1]):
	#	print(grp)
	#	plt.plot(grp[0], label=key)
	#	grp[3] = pd.rolling_mean(grp[1], window=5)    
	#	plt.plot(grp[3], label='rolling ({k})'.format(k=key))
	#plt.legend(loc='best')    
	#plt.show()

	#iterate through the rows to create objects
	for index, row in data.iterrows():
		#print( row[0], row[1], row[2])
		tweet = Tweet(row[0], row[1], row[2])
		tweet_list.append(tweet)

	return tweet_list

def main():
	file_name = "/bitcoins_for_plotting.csv"
	
	tweet_list = getDataFromFile(file_name)



main()
