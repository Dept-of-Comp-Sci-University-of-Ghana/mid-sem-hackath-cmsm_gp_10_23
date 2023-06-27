import tweepy
import warnings
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import word_tokenize
warnings.filterwarnings('ignore')


# Access the dataset
#dataset = pd.read_csv('./Suicide_Detection.csv', nrows=10000)
dataset = pd.read_csv('~/Documents/Suicide_Detection.csv', nrows=1000)

# check to print first 10 rows of the dataset, used to ensure it is being accessed
print(dataset.head())
# displays , number of unique values for each class, from this we can see the dataset is evenly divided
print("\n")
print(dataset['class'].value_counts())

# train the model on existing data
#train_data, test_data = train_test_split(
    #dataset, test_size=0.2, random_state=10)

'''
# Chart to show distribution of suicide and not suicide
plt.figure(figsize=(12,10))
plt.pie(train_data['class'].value_counts(),startangle=90,colors=['#FF0000','#000fbb'],
        autopct='%0.2f%%',labels=['suicide','Not Suicide'])
plt.title('SUICIDE OR NOT ?',fontdict={'size':20})
plt.show()
'''


# Function to clean and preprocess data for building the sentiment analysis model
def preprocessing(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    cleaned_text = []

    # Perform lowercase and remove special characters
    processed_text = ' '.join(token.lemma_.lower()
                              for token in doc if not token.is_punct)

    # Stopword removal
    stop_words = spacy.lang.en.stop_words.STOP_WORDS
    words = processed_text.split()
    words = [word for word in words if word not in stop_words]

    # Join the words back into a single string
    processed_text = ' '.join(words)

    cleaned_text.append(processed_text)

    return cleaned_text


#cleaned_train_text = train_data['text'].apply(preprocessing)
#cleaned_test_text = test_data['text'].apply(preprocessing)

# Mapping the class column. Suicide maps to 1, non-suicide maps to 0
label_mapping = {'suicide': 1, 'non-suicide': 0}
dataset['class'] = dataset['class'].map(label_mapping)


dataset['cleaned_text'] = dataset['text'].apply(preprocessing)

dataset['text'] = dataset['cleaned_text'].str.join(' ')


vect = CountVectorizer(max_features=100)
#fit the vectorizer to the text column
vect.fit(dataset.text)

#transform the text column
X_text = vect.fit_transform(dataset['text'])

#X_text = vect.fit_transform(dataset['cleaned_text'].str.join(' '))

#Create the BOW (Bag of Words) representation
X_df=pd.DataFrame(X_text.toarray(), columns=vect.get_feature_names_out())
print(X_df.head())

#Convert DataFrame to dictionary for wordcloud creation
words = X_df.sum().to_dict()
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)

#Plotting the wordcloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#split the dataset
#train_data, test_data = train_test_split(
    #dataset, test_size=0.2, random_state=10)

#y = dataset['class']
#X = dataset.drop('class', axis=1)

#log_reg = LogisticRegression().fit(X, y)
#print('Accuracy of logistic regression: ', log_reg.score(X, y))



# the code below this is to be integrated into the evaluation if necessary

"""
#function to put all collected tweet data into a dictionary and print it out
def printtweetdata(n,ith_tweet, scrapped_data):

    d = {"Tweet Scraped":n, "Username":ith_tweet[0], "Description":ith_tweet[1],"Tweet Text":ith_tweet[2], "Hashtags Used":ith_tweet[3],}

    print(f"Tweet Scrapped{n}:")
    print(f"Username:{ith_tweet[0]}")
    print(f"Description:{ith_tweet[1]}")
    print(f"Tweet Text:{ith_tweet[2]}")
    print(f"Hashtags Used:{ith_tweet[3]}")
    scrapped_data.append(d)


#tweet scraping function
def scrape(keywords, num_tweets, scrapped_data):
    print("runs")
    tweets = tweepy.Cursor(api.search_tweets, q=keywords, lang="en",
                           tweet_mode='extended').items(num_tweets)

    list_tweets = [tweet for tweet in tweets]

    i = 1

    #iteration
    for tweet in list_tweets:
        username = tweet.user.screen_name
        description = tweet.user.description
        totaltweets = tweet.user.statuses_count
        hashtags = tweet.entities['hashtags']
        print(hashtags)
        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])

        ith_tweet = [username, description, totaltweets, text, hashtext]

        printtweetdata(i, ith_tweet, scrapped_data)
        i = i+1

#main function
if __name__ == '__main__':
    # Set up your Twitter API credentials
    consumer_key = "WC2RfdKkc3I3jvqW5Fk39ZSLJ"
    consumer_secret = "perfdWp84L3tqcmCkWqKlP4jy3xXIrZtTLiWNgPESy9wPQp4ub"
    access_token = "2459334069-dYcyulBBJC7YZOnlW8SihUWSdWl40SnSJxbdxD8"
    access_token_secret = "HLFh68ySEJVNSjVoQEO7bsGBTBBEfcvxOsxX4Qmh53lOm"

    # Authenticate with Twitter API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Define keywords related to suicide and self-harm
    keywords = "kill myself OR commit suicide OR self-harm"

    # Set the number of tweets you want to collect
    num_tweets = 100

    # Collect tweets based on keywords
    scraped_data = []
    print("Fetching tweets...")
    scrape(keywords, num_tweets, scraped_data)
    print("Scraping completed")
"""
