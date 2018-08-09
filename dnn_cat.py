import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.misc
from dnn import *


def main():
    train_x, train_y, test_x, test_y, classes = load_cat_data()
    num_px = 64
    n_x = num_px * num_px * 3
    n_y = 1
    layers_dims = (n_x, 20, 5, n_y)
    parameters = L_layer_model(train_x, train_y, layers_dims, learning_rate=0.0075, num_iterations=3000, lambd=0.7, keep_prob=0.88, print_cost=True)
    predict(test_x, test_y, parameters)


def test(parameters):
    num_px = 64
    my_image = "dog.jpg"
    my_label_y = [0]
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((num_px * num_px * 3, 1))
    my_predicted_image = predict(my_image, my_label_y, parameters)
    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")


if __name__ == '__main__':
    main()
