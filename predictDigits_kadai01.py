import sklearn.datasets
import sklearn.svm
import PIL.Image
import numpy
import matplotlib.pyplot as plt

def imageToData(filename):

    grayImage = PIL.Image.open(filename).convert("L")
    grayImage = grayImage.resize((8,8),PIL.Image.Resampling.LANCZOS)

    numImage = numpy.asarray(grayImage, dtype = float)
    numImage = 16 - numpy.floor(17 * numImage / 256)
    numImage = numImage.flatten()


    plt.imshow(numImage.reshape((8,8)), cmap='gray', interpolation='nearest')
    plt.title("変換後の画像データ")
    plt.colorbar()
    plt.show()


    print(numImage)
    return numImage

def predictDigits(data):
    digits = sklearn.datasets.load_digits()

    clf = sklearn.svm.SVC(gamma = 0.001)
    clf.fit(digits.data, digits.target)

    n = clf.predict([data])
    print("予測=",n)

data = imageToData("7.png")
predictDigits(data)
