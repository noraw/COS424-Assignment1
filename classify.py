# ********************************************
# Author: Ruth Dannenfelser
# Date: Feb 17, 2015
#
# Pipeline for applying classifiers to
# the spam / ham classification problem.
#
# ********************************************
import numpy as np
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


# read a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
# Adapted from email_process.py
def read_bagofwords_dat(myfile, numofemails=10000):
    bagofwords = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    bagofwords = np.reshape(bagofwords,(numofemails,-1))
    return bagofwords


# argument parsing.
parser = argparse.ArgumentParser(description='Classify spam/ham emails.')
parser.add_argument("-M", "--MultinomialNB", action="store_true", help="run naive bayes with multinomial model")
parser.add_argument("-B", "--BernoulliNB", action="store_true", help="run naive bayes with bernoulli model")
parser.add_argument("-S", "--SVM", action="store_true", help="run SVM")
parser.add_argument("-N", "--NN", action="store_true", help="run nearest neighbor")
#parser.add_argument('-f', help="training features" type=str)

args = parser.parse_args()


# data.
X = read_bagofwords_dat("trec07p_data/Train/train_emails_bag_of_words_200.dat", 45000)
X_test = read_bagofwords_dat("trec07p_data/Test/test_emails_bag_of_words_0.dat", 5000)
y = np.loadtxt(fname=open("trec07p_data/Train/train_emails_classes_200.txt"), dtype=str)
y_test = np.loadtxt(fname=open("trec07p_data/Test/test_emails_classes_0.txt"), dtype=str)

outname = ""
# CLASSIFY!
if args.MultinomialNB:
    print "Multinomial NB"
    outname = "mNB"
    clf = MultinomialNB();
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) # use defaults

if args.BernoulliNB:
    print "Bernoulli NB"
    outname = "bNB"
    clf = BernoulliNB()
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True) # use defaults

if args.SVM:
    print "SVM"
    outname = "SVM"
    clf = svm.SVC(kernel='rbf', C=1.0)
#    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#    gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
#    random_state=None, shrinking=True, tol=0.001, verbose=False) # use defaults

if args.NN:
    print "NN"
    outname ="NN"
    clf = KNeighborsClassifier(n_neighbors=3);

if args.MultinomialNB or args.BernoulliNB or args.SVM or args.NN:
    clf.fit(X, y)
    predicted = clf.predict(X_test)
    probs_train = clf.predict_proba(X)
    probs_test = clf.predict_proba(X_test)
    print "Accuracy: " + str(clf.score(X_test, y_test)) # classifier accuracy
    print(metrics.classification_report(y_test, predicted))

    # output scores for ROC curve
    outfile = file("predictions_" + outname + ".txt", "w")
    for i in range (len(predicted)):
        outfile.write(str(predicted[i]) + "\t" + str(probs_test[i][1]) + "\n") #prob of spam
    outfile.close();
