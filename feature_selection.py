from os.path import isfile, isdir, join
import numpy
from sklearn.feature_selection import VarianceThreshold

import sys
import getopt
import codecs


# reading a bag of words file back into python. The number and order
# of emails should be the same as in the *samples_class* file.
def read_matrix_dat_file(myfile, numofemails=10000):
    matrix = numpy.fromfile(myfile, dtype=numpy.uint8, count=-1, sep="")
    matrix = numpy.reshape(matrix,(numofemails,-1))
    return matrix

def read_array_name_file(myfile):
	arrayfile = codecs.open(myfile, encoding="utf-8-sig").read()
	array = numpy.frombuffer(arrayfile.replace("\n", ""), dtype="<U2")
	#array = [str(line.rstrip('\n')) for line in arrayfile]
	return array


def main(argv):
    print argv;
    path = ''
    featuresFile = ''
    matrixFileTrain = ''
    matrixFileTest = ''
    numEmails = ''
    numEmails2 = ''

    try:
      opts, args = getopt.getopt(argv,"p:f:m:t:e:g", ["path=", "features=", "matrix=","test=", "number_of_emails=", "number_of_emails2="])
    except getopt.GetoptError:
      print 'ERROR:   python feature_selection.py -p <path> -f <inputfile> -m <inputfile> -t <matrixTest> -e <numberOfEmailsTrain> -g <numberOfEmailsTest>'
      sys.exit(2)

    numEmails2 = int(args[0])
    for opt, arg in opts:
      if opt == '-h':
         print 'HELP:   feature_selection.py -p <path> -f <vocabfile> -m <matrixTrain> -t <matrixTest> -e <numberOfEmailsTrain> -g <numberOfEmailsTest>'
         sys.exit()
      elif opt in ("-p", "--path"):
         path = arg
      elif opt in ("-f", "--features"):
         featuresFile = arg
      elif opt in ("-m", "--matrix"):
         matrixFileTrain = arg
      elif opt in ("-t", "--test"):
         matrixFileTest = arg
      elif opt in ("-e", "--number_of_emails"):
         numEmails = arg


    featureArray = read_array_name_file(featuresFile)
    featureMatrixTrain = read_matrix_dat_file(matrixFileTrain, numofemails=int(numEmails))
    featureMatrixTest = read_matrix_dat_file(matrixFileTest, numofemails=int(numEmails2))


    selector = VarianceThreshold(threshold=1.0)
    selector.fit(featureMatrixTrain)

    featureMatrixTrain_selected = selector.transform(featureMatrixTrain)
    featureMatrixTrain_zeroed = selector.inverse_transform(featureMatrixTrain_selected)

    featureMatrixTest_selected = selector.transform(featureMatrixTest)
    featureMatrixTest_zeroed = selector.inverse_transform(featureMatrixTest_selected)

    selectedFeatureArray = []
    deletedFeatureArray = []
    for i in range(len(featureArray)):
        col = featureMatrixTrain_zeroed[:,i]
        numNonZeros = numpy.count_nonzero(col)
        if(numNonZeros == 0):
            deletedFeatureArray.append(featureArray[i])
        else:
            selectedFeatureArray.append(featureArray[i])


    outfile= codecs.open(path+"/features_selected_nameArray.txt", 'w',"utf-8-sig")
    outfile.write("\n".join(selectedFeatureArray))
    outfile.close()

    outfile= codecs.open(path+"/features_deleted_nameArray.txt", 'w',"utf-8-sig")
    outfile.write("\n".join(deletedFeatureArray))
    outfile.close()

    featureMatrixTrain_selected.tofile(path+"/featureTrain_selected_matrix.dat")
    featureMatrixTest_selected.tofile(path+"/featureTest_selected_matrix.dat")











if __name__ == "__main__":
   main(sys.argv[1:])
