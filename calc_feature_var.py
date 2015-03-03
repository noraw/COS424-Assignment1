# ********************************************
# Author: Ruth Dannenfelser
# Date: Feb 17, 2015
#
# Calculate variance for a matrix of
# features. Output variance in a form for
# plotting.
#
# ********************************************
import numpy as np
from scipy import sparse

# read a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
# Adapted from email_process.py
def read_dat(myfile, numofemails):
    mat = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    mat = np.reshape(mat,(numofemails,-1))
    return [np.var(col) for col in mat.transpose()]

#var_x = read_dat("trec07p_data/Train/train_emails_bag_of_words_200.dat", 45000)
#var_x = read_dat("trec07p_data/Train/combine_features_matrix.dat",45000)
#var_x = read_dat("trec07p_data/Train/combine_features_matrix_words.dat",45000)
var_x = read_dat("trec07p_data/Train/combine_features_matrix_bigrams.dat",45000)
#var_x = sparse.csr_matrix(var_x)

# add labels and save to file.
#out = file("feature_variance_words_200.txt", "w")
out = file("feature_variance_bigrams_200.txt", "w")
labels = []
#with open("trec07p_data/Train/train_emails_vocab_200.txt", "r") as f:
with open("trec07p_data/Train/train_emails_bigram_vocab_200.txt", "r") as f:
    for line in f:
        labels.append(line.strip());

# length
with open("trec07p_data/Train/train_emails_length_vocab.txt", "r") as f:
    for line in f:
        labels.append("length_" + line.strip());
# links
with open("trec07p_data/Train/train_emails_links_vocab.txt", "r") as f:
    for line in f:
        labels.append("links_" + line.strip());
# sender
with open("trec07p_data/Train/train_emails_sender_vocab.txt", "r") as f:
    for line in f:
        labels.append("sender_" + line.strip());


for i in range(len(var_x)):
    out.write(str(var_x[i]) + "\t" + labels[i] + "\n")
out.close()
