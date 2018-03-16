import json
import sys
from csv import writer
import codecs
import re 


def uniquelines(lineslist):
    unique = {}
    result = []
    for item in lineslist:
        if item.strip() in unique: continue
        unique[item.strip()] = 1
        result.append(item)
    return result


def main():

	#buildMainFile()
	file1 = codecs.open('/NEW_TW_DATA/tweets_final.csv','r+','utf-8')
	filelines = file1.readlines()
	file1.close()

	#removing duplicates in the file
	with codecs.open("/NEW_TW_DATA/tweets_final_unique.csv", "w", 'utf-8') as output:
		output.writelines(uniquelines(filelines))

def buildMainFile():

	#TwitterScraper created multiple json files for the twitter mining
	#this function handles gathering all the files and joining one unique file
	# also handles removing URL's from tweets

	read_file_name = "/NEW_TW_DATA/bitcoin_33.json"
	write_file_name = "/tweets_final.csv"

	with codecs.open(read_file_name, 'r', 'utf-8') as f:
		tweets = json.load(f, encoding='utf-8')

	with open(write_file_name, 'a', encoding="utf-8") as out_file:
		out_file.write('tweet_id, tweet_time, tweet_author, tweet_text')
		csv = writer(out_file)

		for tweet in tweets:
			## Remove URL's 
			changeText = tweet['text'].replace("\n", " ")
			URLless_string = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', changeText)

			test = '\n{},{},{},"{}"'.format(tweet['id'], tweet['timestamp'], tweet['user'], URLless_string )
			#print(test)
			out_file.write(test)

main()
