
import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
import numpy
import re
import sys
import getopt
import codecs
import time

chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']


def get_files(mypath):
    return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def get_dirs(mypath):
    return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]

# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_emailLengths_dat(myfile, numofemails=10000):
    bagofwords = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    bagofwords=numpy.reshape(bagofwords,(numofemails,-1))
    return bagofwords


def find_emailLengthsMatrix(emailLengths):
    emailLengthsMatrix = numpy.zeros(shape=(len(emailLengths),1), dtype=numpy.uint8)
    vocabIndex={}

    for i in range(len(emailLengths)):
        emailLengthsMatrix[i,0] = int(emailLengths[i])

    print "Finished find_emailLengthsMatrix for : "+str(len(emailLengths))+"  docs"
    return(emailLengthsMatrix)


def get_email_lenghts(path, train=True):
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    classes = []
    samples = []
    docs = ['length']
    dirs = get_dirs(path)
    words = []

    for dir in dirs:
        files = get_files(path+"/"+dir)
        for f in files:
            classes.append(dir)
            samples.append(f)

            inf = open(path+'/'+dir+'/'+f,'r')
            raw = inf.read().decode('latin1') # or ascii or utf8 or utf16
            words.append(str(len(raw)));         

    return(docs, classes, samples, words)
        

# path should have one folder for each class. Class folders should
# contain text documents that are labeled with the class label (folder
# name). Bag of words representation, vocabulary will be output to
# <outputfile>_*.dat files.
def main(argv):
   path = ''
   outputf = ''
   vocabf = ''
   start_time = time.time()

   try:
      opts, args = getopt.getopt(argv,"p:o:v:",["path=","ofile=","vocabfile="])
   except getopt.GetoptError:
      print 'python text_process.py -p <path> -o <outputfile> -v <vocabulary>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'text_process.py -p <path> -o <outputfile> -v <vocabulary>'
         sys.exit()
      elif opt in ("-p", "--path"):
         path = arg
      elif opt in ("-o", "--ofile"):
         outputf = arg
      elif opt in ("-v", "--vocabfile"):
         vocabf = arg
	 
   print 'Path is "', path
   print 'Output file name is "', outputf
   (docs, classes, samples, vocab) = get_email_lenghts(path, train=True)

   bow = find_emailLengthsMatrix(vocab)
   #write to binary file for large data set
   bow.tofile(path+"/"+outputf+"_matrix.dat")

   # print out files
   outfile= codecs.open(path+"/"+outputf+"_vocab.txt", 'w',"utf-8-sig")
   outfile.write("\n".join(docs))
   outfile.close()

   #write to text file for small data set
   #bow.tofile(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".txt", sep=",", format="%s")
   outfile= open(path+"/"+outputf+"_classes.txt", 'w')
   outfile.write("\n".join(classes))
   outfile.close()

   outfile= open(path+"/"+outputf+"_samples_class.txt", 'w')
   outfile.write("\n".join(samples))
   outfile.close()
   print "Time: " + str(time.time() - start_time)

if __name__ == "__main__":
   main(sys.argv[1:])



