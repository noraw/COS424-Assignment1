# ********************************************
# Author: Ruth Dannenfelser
# Date: Feb 11, 2015
#
# Apply Naive Bayes classifiers to
# the spam / ham classification problem.
#
# ********************************************
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# read a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
# Adapted from email_process.py
def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofemails,-1))
    return bagofwords



num_emails = 100 # number of emails to use.
# features
X = read_bagofwords_dat("trec07p_data/Train/train_emails_bag_of_words_200.dat", 45000) #45,000
#X = read_bagofwords_dat("testData/Train/train_emails_words_bag_of_words_200.dat", 80)
#X_test = read_bagofwords_dat("testData/Test/test_emails_matrix_0.dat", 80)
X_test = read_bagofwords_dat("trec07p_data/Test/test_emails_bag_of_words_0.dat", 5000) # 5,000


# Snippet that will help later with bigrams
#bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
#analyze = bigram_vectorizer.build_analyzer()
#analyze('Bi-grams are cool!') == (['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])

print "Shape of the training set: " + str(X.shape);
print "Shape of the test set: " + str(X_test.shape);

# labels
y = np.loadtxt(fname=open("trec07p_data/Train/train_emails_classes_200.txt"), dtype=str)
y_test = np.loadtxt(fname=open("trec07p_data/Test/test_emails_classes_0.txt"), dtype=str)

print "data read in."

# Multinomial NB
clf = MultinomialNB()
clf.fit(X, y)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) # use defaults
print "trained."

predicted = clf.predict(X_test)
pred_mean = np.mean(predicted == y_test)
print pred_mean;
print(metrics.classification_report(y_test, predicted))
#
print "Accuracy: " + str(clf.score(X_test, y_test)) # classifier accuracy



# # Bernoulli NB
# clf2 = BernoulliNB()
# clf2.fit(X, y)
# BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
# print(clf2.predict(X[2]))
