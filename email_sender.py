
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

import urlmarker

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


def find_emailSenderMatrix(docs, uniqueSenders):
    emailSenderMatrix = numpy.zeros(shape=(len(docs),len(uniqueSenders)), dtype=numpy.uint8)
    senderIndex={}
    for i in range(len(uniqueSenders)):
       senderIndex[uniqueSenders[i]]=i

    for i in range(len(docs)):
       doc = docs[i]
       index_t=senderIndex.get(doc)
       if index_t>=0:
          emailSenderMatrix[i,index_t] = 1

    print "Finished find_emailSenderMatrix for : "+str(len(docs))+"  docs"
    return(emailSenderMatrix)



def get_email_sender(path, train=True):
    porter = nltk.PorterStemmer() # also lancaster stemmer
    wnl = nltk.WordNetLemmatizer()
    stopWords = stopwords.words("english")
    classes = []
    samples = []
    dirs = get_dirs(path)
    docs = []
    uniqueSenders = []


    for dir in dirs:
        files = get_files(path+"/"+dir)
        for f in files:
            links = {}
            addToDocs = False
            classes.append(dir)
            samples.append(f)

            inf = open(path+'/'+dir+'/'+f,'r')
            raw = inf.read().decode('latin1') # or ascii or utf8 or utf16
            senderBegin = raw.find("From:")
            senderEnd = raw.find("\n", senderBegin)
            senderLine = raw[senderBegin: senderEnd-1]
            if(senderLine.find("To:") == -1):
                atPoint = senderLine.find("@")
                periodPoint = senderLine.rfind(".", atPoint)
                if (periodPoint >= 0):
                    ending = senderLine[periodPoint+1:]
                    ending = re.sub('[%s]' % ''.join(chars), '', ending)
                    ending.strip()
                    if(ending != ""):
                        addToDocs = True
                        if(ending not in uniqueSenders):
                            uniqueSenders.append(ending)
            if(addToDocs):
                docs.append(ending)
            else:
                docs.append('')
    return(docs, classes, samples, uniqueSenders)
        

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
   print 'vocabulary file is "', vocabf
   if (not vocabf):
      (docs, classes, samples, uniqueSenders) = get_email_sender(path, train=True)
   else:
      (docs, classes, samples, uniqueSenders) = get_email_sender(path, train=False)
      vocabfile = codecs.open(path+"/"+vocabf, 'r',"utf-8-sig")
      uniqueSenders = [str(line.rstrip('\n')) for line in vocabfile]
      vocabfile.close()


   bow = find_emailSenderMatrix(docs, uniqueSenders)
   #write to binary file for large data set
   bow.tofile(path+"/"+outputf+"_matrix.dat")

   if (not vocabf):
       # print out files
       outfile= codecs.open(path+"/"+outputf+"_vocab.txt", 'w',"utf-8-sig")
       outfile.write("\n".join(uniqueSenders))
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



