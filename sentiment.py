import tweepy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def load_creds(filename = 'creds'):
    # File Format
    # CONSUMER_KEY
    # CONSUMER_SECRET
    # ACCESS_TOKEN
    # ACCESS_TOKEN_SECRET

    with open(filename, 'r') as open_file:
        lines = open_file.readlines()
        assert(len(lines) == 4) , f"Requires 4 credential arguements! Given {len(lines)}"
        lines = [line.rstrip('\n') for line in lines] #remove trailing \n
        return lines
        
        
def csv_write(data, filename = "output.csv"):
    pd.DataFrame(data).to_csv(filename, index=False)

def main():
    creds = load_creds()
    # Step 1 - Authenticate

    consumer_key= creds[0]
    consumer_secret= creds[1]

    access_token=creds[2]        
    access_token_secret=creds[3]    

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)
    
    #Step 3 - Retrieve Tweets
    trump_tweets = api.user_timeline('realDonaldTrump', full_text=True, count=100, tweet_mode='extended')

    #CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
    #and label each one as either 'positive' or 'negative', depending on the sentiment 
    #You can decide the sentiment polarity threshold yourself
    data = []
    cnt = 0
    for tweet in trump_tweets:
        cnt += 1
        print(f'[{cnt}]Current tweet: {tweet.full_text[:75]+"..." if len(tweet.full_text) > 75 else tweet.full_text}')
        # print(tweet.full_text)
        #Step 4 Perform Sentiment Analysis on Tweets
        analysis = TextBlob(tweet.full_text, analyzer = NaiveBayesAnalyzer())
        data.append({'text' : tweet.full_text, 'classification': analysis.sentiment.classification})
        
    csv_write(data, 'test2.csv')
    plot_chart(data)

def plot_chart(data):
    clfs = [tweet['classification'] for tweet in data]
    counts = Counter(clfs)
    slices = dict(counts).values()
    # activities = dict(counts).keys()
    activities = ["Positive", "Negative"]
    colors = ['#e75a7c', '#06d6a0']
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(slices, autopct='%.1f%%', textprops=dict(color="w"))
    ax.legend(wedges, activities,
            title="Types",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title(f"Donald Trump's Twitter Sentiment Analysis of N = {len(data)} Tweets")

    plt.show()

def csv_read(filename):
    df = pd.read_csv(filename)
    raw = df.to_dict('split')
    data = []
    for el in raw['data']:
        data.append(dict(zip(raw['columns'], el)))
    return data
if __name__ == "__main__":
    main()
    # data = csv_read("test2.csv")
    # plot_chart(data)
    # creds = load_creds()
    # print(creds)
    # To do:
    # Use this http://www.nltk.org/howto/sentiment.html