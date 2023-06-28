import warnings
import pandas as pd
import spacy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score)

warnings.filterwarnings('ignore')


# Preload the spaCy model
nlp = spacy.load('en_core_web_sm')

# Preload stop words
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Access the dataset
dataset = pd.read_csv('./Suicide_Detection.csv', nrows=100)


# Function to clean and preprocess data for building the sentiment analysis model
def preprocessing(text):
    doc = nlp(text)
    cleaned_text = []

    # Perform lowercase and remove special characters
    processed_text = ' '.join(token.lemma_.lower()
                              for token in doc if not token.is_punct)

    # Remove stop words
    list_of_processed_texts = processed_text.split()
    words = [word for word in list_of_processed_texts
             if word not in stop_words]

    # Join the words back into a single string
    processed_text = ' '.join(words)

    cleaned_text.append(processed_text)

    return cleaned_text


# Mapping the class column. Suicide maps to 1, non-suicide maps to 0
label_mapping = {'suicide': 1, 'non-suicide': 0}
dataset['class'] = dataset['class'].map(label_mapping)

# Apply text preprocessing to the 'text' column
dataset['cleaned_text'] = dataset['text'].apply(preprocessing)
dataset['text'] = dataset['cleaned_text'].str.join(' ')

# Convert dataset['text'] to a matrix of token counts
# Create a count vectorizer
vectorizer = CountVectorizer(max_features=100)

# Fit the vectorizer to the text column
vectorizer.fit(dataset.text)

# Transform the text column
X_text = vectorizer.fit_transform(dataset['text'])

# Create the BOW (Bag of Words) representation
X_df = pd.DataFrame(X_text.toarray(),
                    columns=vectorizer.get_feature_names_out())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_df, dataset['class'], test_size=0.2, random_state=10)

# Build logistic regression model
log_reg = LogisticRegression().fit(X_train, y_train)

# Predict on training data
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

# Plot a bar chart for the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics, train_scores, label='Train')
plt.bar(metrics, test_scores, label='Test')
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Evaluation Metrics')
plt.legend()
plt.show()
