import tweepy
import warnings
import pandas as pd
import spacy
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)
import nltk
from nltk.tokenize import word_tokenize
warnings.filterwarnings('ignore')


# Access the dataset
dataset = pd.read_csv('~/Documents/Suicide_Detection.csv', nrows=1000)


# check to print first 5 rows of the dataset, used to ensure it is being accessed
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
#plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# split the dataset
#train_data, test_data = train_test_split(
    #dataset, test_size=0.2, random_state=10)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, dataset['class'], test_size=0.2, random_state=10)

# define matrix of features and target variable
#X_train = X_df
#y_train = dataset['class']

# build logistic regression model
log_reg = LogisticRegression().fit(X_train, y_train)

#predict on training data
y_train_pred = log_reg.predict(X_train)

# Calculate evaluation metrics on training data
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1_score = f1_score(y_train, y_train_pred)

# Predict on testing data
y_test_pred = log_reg.predict(X_test)

# Calculate evaluation metrics on testing data
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Create a bar chart for evaluation metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
train_scores = [train_accuracy, train_precision, train_recall, train_f1_score]
test_scores = [test_accuracy, test_precision, test_recall, test_f1_score]


# confusion matrix
print(confusion_matrix(y_test, y_test_pred)/len(y_test))

# Plot a bar chart for the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, train_scores, label='Train')
plt.bar(metrics, test_scores, label='Test')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Evaluation Metrics')
plt.legend()
plt.show()

print(f"train_accuracy: {train_accuracy}")
print(f"train_precision: {train_precision}")
print(f"train_recall: {train_recall}")
print(f"train_f1_score: {train_f1_score}")
print(f"test_accuracy: {test_accuracy}")
print(f"test_precision: {test_precision}")
print(f"test_recall: {test_recall}")
print(f"test_f1_score: {test_f1_score}")

model_filepath = '/Users/user/PycharmProjects/mid-sem-hackath-cmsm_gp_10_23/model.pkl'

# Save the model
with open(model_filepath, 'wb') as file:
    pickle.dump(log_reg, file)

# Load the saved model
with open(model_filepath, 'rb') as file:
    loaded_model = pickle.load(file)

#Text Classification Function
def classify_texts(texts):
    # Preprocess the texts and convert them to strings
    cleaned_texts = [preprocessing(text)[0] for text in texts]

    # Transform the preprocessed texts using the same CountVectorizer used during training
    X_texts = vect.transform(cleaned_texts)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X_texts)

    # Map the predicted labels back to their original class names
    label_mapping = {1: 'suicide', 0: 'non-suicide'}
    predicted_labels = [label_mapping[prediction] for prediction in predictions]

    # Create a DataFrame with the texts and their corresponding predicted labels
    results = pd.DataFrame({'Text': texts, 'Predicted Label': predicted_labels})

    return results


texts = [
    "I'm feeling really down today. Nothing seems to be going right.",
    "I had a great day with my friends. Life is beautiful.",
    "I can't handle the pain anymore. It's too much to bear.",
    "I'm so happy! Everything is going well for me.",
    "I don't see any point in living anymore.",
    "I achieved a major milestone in my career. Feeling grateful!",
    "I'm overwhelmed by sadness. It's consuming me.",
    "Life is full of possibilities. Excited for what's to come!"
]


results = classify_texts(texts)
print(results)







# the code below this is to be integrated into the evaluation if necessary

'''
#function to put all collected tweet data into a dictionary and print it out
def printtweetdata(n,ith_tweet, scrapped_data):
    d = {"Tweet Scraped": n, "Username": ith_tweet[0], "Description": ith_tweet[1], "Tweet Text": ith_tweet[2],
         "Hashtags Used": ith_tweet[3]}
    scrapped_data.append(d)


#Tweet scraping function
def scrape(keywords, num_tweets):
    print("runs")
    tweets = tweepy.Cursor(api.search_tweets, q=keywords, lang="en", tweet_mode='extended').items(num_tweets)

    list_tweets = [tweet for tweet in tweets]

    scrapped_data = []
    i = 1

    # Iteration
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
        i = i + 1

    return scrapped_data


#main function
if __name__ == '__main__':
    # Set up your Twitter API credentials
    consumer_key = "n0mE9SCjtQJUL7r99t4ef9hh4"
    consumer_secret = "rAUbBXvj1vPp5vmbshqJGp3CLC8cumQi5cuNgLfnDeLiXkSjTO"
    access_token = "1673053054931697664-NmJ61iIUmOqKupJ3wo1VKgjAaExRMo"
    access_token_secret = "pYe1q7oZ6DxKasSfkM2KwoBuKgJmolKJJVolwCN8i9Ff5"

    # Authenticate with Twitter API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Define keywords related to suicide and self-harm
    keywords = "kill myself OR commit suicide OR self-harm"

    # Set the number of tweets you want to collect
    num_tweets = 40

    # Collect tweets based on keywords
    print("Fetching tweets...")
    scrapped_data = scrape(keywords, num_tweets)
    print("Scraping completed")

    # Extract tweets and usernames from the collected data
    texts = [data["Tweet Text"] for data in scrapped_data]
    usernames = [data["Username"] for data in scrapped_data]

    # Call the classify_texts function with the extracted texts
    results = classify_texts(texts)

    # Add the usernames to the results DataFrame
    results['Username'] = usernames

    print(results)
'''
