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
import os
import subprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack
import timeit


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
parser.add_argument("-s", "--selection", action="store_true", help="apply feature selection")
parser.add_argument("-V", "--var", type=float, nargs=1, help="variance threshold")
parser.add_argument("-c", "--combine", action="store_true", help="combine matrix files")
#parser.add_argument('-f', help="training features" type=str)

args = parser.parse_args()



# feature selection
# print "feature selection started."
# print subprocess.Popen("python feature_selection.py -p trec07p_data -f trec07p_data/Train/train_emails_vocab_200.txt -m trec07p_data/Train/train_emails_bag_of_words_200.dat -t trec07p_data/Test/test_emails_bag_of_words_0.dat -e 45000 -g 5000", shell=True, stdout=subprocess.PIPE).stdout.read();
# print "feature selection finished."
# X = read_bagofwords_dat("trec07p_data/featureTrain_selected_matrix.dat", 45000)
# X_test = read_bagofwords_dat("trec07p_data/featureTest_selected_matrix.dat", 5000)

if args.combine:
    # data.
    use_bigrams = False;

    X_words = read_bagofwords_dat("trec07p_data/Train/train_emails_bag_of_words_200.dat", 45000)
    X_test_words = read_bagofwords_dat("trec07p_data/Test/test_emails_bag_of_words_0.dat", 5000)
    print "words read in."
    print "words: " + str(X_words.shape)
    print "words_test: " + str(X_test_words.shape)

    if use_bigrams:
        X_bigrams = read_bagofwords_dat("trec07p_data/Train/train_emails_bag_of_bigrams_50.dat", 45000)
        X_test_bigrams = read_bagofwords_dat("trec07p_data/Test/test_emails_bag_of_bigrams_0.dat", 5000)
        print "bigrams read in."
        print "bigrams: " + str(X_bigrams.shape)
        print "bigrams_test: " + str(X_test_bigrams.shape)

    # normalize counts using tf-idf
    print "begin normalization step."

    transformer = TfidfTransformer()
    X = transformer.fit_transform(X_words).toarray()
    X_test = transformer.fit_transform(X_test_words).toarray()

    print X.shape;

    # adding normalized bigrams to feature space
    if use_bigrams:
        X = hstack([X, transformer.fit_transform(X_bigrams).toarray()])
        X_test = hstack([X_test, transformer.fit_transform(X_test_bigrams).toarray()])
        print X.shape;

    print "finished normalization step."

    # combine other features.
    # To Do: JUST read them in and CONCAT them to the array!

    X_length = read_bagofwords_dat("trec07p_data/Train/train_emails_length_matrix.dat", 45000)
    X_test_length = read_bagofwords_dat("trec07p_data/Test/test_emails_length_matrix.dat", 5000)

    X_links = read_bagofwords_dat("trec07p_data/Train/train_emails_links_matrix.dat", 45000)
    X_test_links = read_bagofwords_dat("trec07p_data/Test/test_emails_links_matrix.dat", 5000)

    X_sender = read_bagofwords_dat("trec07p_data/Train/train_emails_sender_matrix.dat", 45000)
    X_test_sender = read_bagofwords_dat("trec07p_data/Test/test_emails_sender_matrix.dat", 5000)

    print "X concatenate"
    print "length: " + str(X_length.shape)
    X = hstack([X, X_length])
    print "links: " + str(X_links.shape)
    X = hstack([X, X_links])
    print "sender: " + str(X_sender.shape)
    X = hstack([X, X_sender])
    print "X_test concatenate"
    print "length: " + str(X_test_length.shape)
    X_test = hstack([X_test, X_test_length])
    print "links: " + str(X_test_links.shape)
    X_test = hstack([X_test, X_test_links])
    print "sender: " + str(X_test_sender.shape)
    X_test = hstack([X_test, X_test_sender])

    X.tofile("trec07p_data/Train/combine_features_matrix.dat")
    X_test.tofile("trec07p_data/Test/combine_features_matrix.dat")
    print "Done"
else:
    X = read_bagofwords_dat("trec07p_data/Train/combine_features_matrix.dat", 45000)
    X_test = read_bagofwords_dat("trec07p_data/Test/combine_features_matrix.dat", 5000)


y = np.loadtxt(fname=open("trec07p_data/Train/train_emails_classes_200.txt"), dtype=str)
y_test = np.loadtxt(fname=open("trec07p_data/Test/test_emails_classes_0.txt"), dtype=str)

outname = ""
if args.selection:
    thresh = args.var[0];
    print "Variance Feature Selection Threshold: " + str(args.var[0]);

# CLASSIFY!
if args.MultinomialNB:
    print "Multinomial NB"
    outname = "mNB"
    if args.selection:
        clf = Pipeline([('feature_selection', VarianceThreshold(threshold=thresh)),('classification', MultinomialNB())])
    else:
        clf = MultinomialNB();
    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) # use defaults

if args.BernoulliNB:
    print "Bernoulli NB"
    outname = "bNB"
    if args.selection:
        clf = Pipeline([('feature_selection', VarianceThreshold(threshold=thresh)),('classification', BernoulliNB())])
    else:
        clf = BernoulliNB()
    BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True) # use defaults

if args.SVM:
    print "SVM"
    outname = "SVM"
    if args.selection:
        clf = Pipeline([('feature_selection', VarianceThreshold(threshold=thresh)),('classification', svm.SVC(verbose=True, probability=True))])
    else:
        clf = svm.SVC(verbose=True, probability=True)
    #clf = svm.SVC(kernel='rbf', C=1.0)
#    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#    gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
#    random_state=None, shrinking=True, tol=0.001, verbose=False) # use defaults

if args.NN:
    print "NN"
    outname ="NN"
    if args.selection:
        clf = Pipeline([('feature_selection', VarianceThreshold(threshold=thresh)),('classification', KNeighborsClassifier(n_neighbors=3))])
    else:
        clf = KNeighborsClassifier(n_neighbors=3);

if args.MultinomialNB or args.BernoulliNB or args.SVM or args.NN:
    start = timeit.default_timer()
    clf.fit(X, y)
    predicted = clf.predict(X_test)
    #probs_train = clf.predict_proba(X)    
    probs_test = clf.predict_proba(X_test)
    print "Accuracy: " + str(clf.score(X_test, y_test)) # classifier accuracy
    print(metrics.classification_report(y_test, predicted))
    stop = timeit.default_timer()
    print " runtime: " + str(stop - start)

    # output scores for ROC curve
    outfile = file("predictions_" + outname + ".txt", "w")
    for i in range (len(predicted)):
        outfile.write(str(predicted[i]) + "\t" + str(probs_test[i][1]) + "\n") #prob of spam
    outfile.close();
