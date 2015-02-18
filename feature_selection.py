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
    arrayfile = codecs.open(myfile, 'r',"utf-8-sig")
    array = [str(line.rstrip('\n')) for line in arrayfile]
    arrayfile.close()
    return array


def main(argv):
    featuresFile = ''
    matrixFileTrain = ''
    matrixFileTest = ''
    numEmails = ''

    try:
      opts, args = getopt.getopt(argv,"f:m:t:e", ["features=", "matrix=","test=", "number_of_emails="])
    except getopt.GetoptError:
      print 'ERROR:   python feature_selection.py -p <path> -f <inputfile> -m <inputfile> -t <matrixTest> -e <numberOfEmails>'
      sys.exit(2)

    numEmails = int(args[0])
    for opt, arg in opts:
      if opt == '-h':
         print 'HELP:   feature_selection.py -f <inputfile> -m <matrixTrain> -t <matrixTest> -e <numberOfEmails>'
         sys.exit()
      elif opt in ("-f", "--features"):
         featuresFile = arg
      elif opt in ("-m", "--matrix"):
         matrixFileTrain = arg
      elif opt in ("-t", "--test"):
         matrixFileTest = arg
     
    featureArray = read_array_name_file(featuresFile)
    featureMatrixTrain = read_matrix_dat_file(matrixFileTrain, numofemails=numEmails)
    featureMatrixTest = read_matrix_dat_file(matrixFileTest, numofemails=numEmails)

	 
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


    outfile= codecs.open("selected_features_nameArray.txt", 'w',"utf-8-sig")
    outfile.write("\n".join(selectedFeatureArray))
    outfile.close()

    outfile= codecs.open("deleted_features_nameArray.txt", 'w',"utf-8-sig")
    outfile.write("\n".join(deletedFeatureArray))
    outfile.close()

    featureMatrixTrain_selected.tofile("featureTrain_selected_matrix.dat")
    featureMatrixTest_selected.tofile("featureTest_selected_matrix.dat")











if __name__ == "__main__":
   main(sys.argv[1:])

