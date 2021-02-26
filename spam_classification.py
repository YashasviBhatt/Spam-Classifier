# Importing the Important Libraries
import pandas as pd
import nltk

# Importing the Data Preprocessing Libraries
import re as regex
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing Model Building Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Downloading the nltk Package
nltk.download('stopwords')

# Importing the Dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', names=['Label', 'Message'])

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

# Since there're no null values present in the file so we can proceed further and do other steps of data preprocessing
# Using Lemmatization
lemmatizer = WordNetLemmatizer()

# Creating an Empty List
sentences = []

# Removing all the Extra Chaacters and Symbols from the Text
for row_idx in range(len(df)):
    msg = regex.sub('[^a-zA-Z]', ' ', df['Message'][row_idx])
    msg = msg.lower()
    msg = msg.split()
    msg = [lemmatizer.lemmatize(word) for word in msg if word not in set(stopwords.words('English'))]
    msg = ' '.join(msg)
    sentences.append(msg)
    
# Generating Vectors out of Messages in Data Set and Generating Feature Set
tfidf_ve = TfidfVectorizer(max_features=3000)
X = tfidf_ve.fit_transform(sentences).toarray()

# Setting Encoding Type
enc = {
    'Label' : {'ham' : 0, 'spam' : 1}
}

# Encoding the Label Column and Extracting Class Set
y = df.replace(enc).iloc[:, 0].values

#----------------------------------------Data Preprocessing and Data Cleaning----------------------------------------

#----------------------------------------Model Building and Training----------------------------------------

# Diving the Dataset into Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Fitting the Classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)
print(accuracy_score(y_test, y_pred))

#----------------------------------------Model Building and Training----------------------------------------