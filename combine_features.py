from os.path import isfile, isdir, join
import numpy
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
    path = ''
    namesFiles = ''
    matrixFiles = ''
    numEmails = ''

    try:
      opts, args = getopt.getopt(argv,"p:n:m:e", ["path=", "feature_names=", "feature_matrix=", "number_of_emails="])
    except getopt.GetoptError:
      print 'ERROR:   python combine_features.py -p <path> -n <inputfileslist> -m <inputfileslist> -e <numberOfEmails>'
      sys.exit(2)

    numEmails = int(args[0])
    for opt, arg in opts:
      if opt == '-h':
         print 'HELP:   combine_features.py -p <path> -n <inputfileslist> -m <inputfileslist> -e <numberOfEmails>'
         sys.exit()
      elif opt in ("-p", "--path"):
         path = arg
      elif opt in ("-n", "--feature_names"):
         namesFiles = arg
      elif opt in ("-m", "--feature_matrix"):
         matrixFiles = arg
     

    namesFilesList = namesFiles.split(" ");
    matrixFilesList = matrixFiles.split(" ");
    print namesFilesList
    print matrixFilesList
    print numEmails

    for i in range(len(namesFilesList)):
        if (i==0):
            featureArray = read_array_name_file(path+"/"+namesFilesList[i])
            featureMatrix = read_matrix_dat_file(path+"/"+matrixFilesList[i], numofemails=numEmails)
        else:
            tempArray = read_array_name_file(path+"/"+namesFilesList[i])
            for item in tempArray:
                featureArray.append(item)


            tempMatrix = read_matrix_dat_file(path+"/"+matrixFilesList[i], numofemails=numEmails)
            featureMatrix = numpy.concatenate((featureMatrix, tempMatrix), axis=1)
            print featureMatrix;



    outfile= codecs.open(path+"/"+"combine_features_nameArray.txt", 'w',"utf-8-sig")
    outfile.write("\n".join(featureArray))
    outfile.close()
    #write to binary file for large data set
    featureMatrix.tofile(path+"/"+"combine_features_matrix_.dat")














if __name__ == "__main__":
   main(sys.argv[1:])

