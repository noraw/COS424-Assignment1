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

# read a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
# Adapted from email_process.py
def read_dat(myfile, numofemails):
    mat = np.fromfile(myfile, dtype=np.uint8, count=-1, sep="")
    mat = np.reshape(mat,(numofemails,-1))
    return [np.var(col) for col in mat.transpose()]

var_x = read_dat("trec07p_data/Train/train_emails_bag_of_words_200.dat", 45000)

# add labels and save to file.
out = file("feature_variance.txt", "w")
labels = []
with open("trec07p_data/Train/train_emails_vocab_200.txt", "r") as f:
    for line in f:
        labels.append(line.strip());


for i in range(len(var_x)):
    out.write(str(var_x[i]) + "\t" + labels[i] + "\n")
out.close()
