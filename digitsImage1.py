import sklearn.datasets
import matplotlib.pyplot as plt

digits = sklearn.datasets.load_digits()

plt.imshow(digits.images[6], cmap="Greys")
plt.show()
