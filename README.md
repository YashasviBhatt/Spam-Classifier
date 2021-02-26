# Spam Classification

This Project checks whether a mail is a Spam or not. The Model uses several **NLP Concepts** such as `Tokenization`, `Lemmatization`, `Vectorization` etc and is trained on a dataset with more than 5500 records using `Multinomial Naive Bayes Classification Algorithm`.

## To run this Project please follow these steps

1. Open _command prompt_ or _powershell window_.
2. Type this command<br>`git clone https://github.com/YashasviBhatt/Spam-Classifier`<br>and press enter.
3. Go inside the _Cloned Repository_ folder and open _command-prompt_ or _powershell window_.

4. Type<br>`pip install -r requirements.txt`<br> and press enter in either _command_prompt_ or _powershell window_ as _administrator_.
5. After Installing all the required _libraries_ run the python file using<br>`python spam_classifier.py`.

## Working

1. Firstly, _data_ is imported using `pandas library`.
2. Secondly, data is preprocessed using several NLP concepts like `Word Tokenization`, `Lemmatization` to fetch base word, `Vectorization` to convert words and sentences into vectors.
3. Thirdly, we divide the _features_ and _label_ into seperate _dataframes_.
4. Now, after creating the separate dataframes for features and label, we split them into _training_ and _testing_ sets.
5. The _training set_ is used to train the _model_ using **Multinomial Naive Bayes Classification Algorithm**, since it best works with NLP.
6. The **Accuracy Score** for this model is then analyzed and it was found out that the model showed a splendid performance having the accuracy score of **97%**.<br><br><br>

**I have used UCI Machine Learning Repository Dataset on SMS Spam Collection Data Set and you can download the dataset from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip). You can visit their website and check for more info on dataset from [here](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).**