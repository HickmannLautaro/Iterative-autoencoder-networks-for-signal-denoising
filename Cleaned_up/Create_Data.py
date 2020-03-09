import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt


def print_imm(imgs):
    n = 20  # how many digits we will display

    plt.figure(figsize=(40, 2 * len(imgs) + 2))
    for i in range(n):
        for j in range(len(imgs)):
            # display original
            ax = plt.subplot(len(imgs), n, i + 1 + j * n)
            plt.imshow(imgs[j][i].reshape(28, 28))
            if i == 0:
                if j == 0:
                    plt.annotate('Original MNIST', xy=(0, 0), xytext=(0, -2))  # , fontsize=20)
                else:
                    plt.annotate('Noise '+str(0.1*float(j))[0:4], xy=(0, 0), xytext=(0, -2))  # , fontsize=20)

            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


def salt_and_pepper(images,prob):
    images = images + 10*np.random.randint(-1,2, size =images.shape)*np.random.choice(2, images.shape, p=[1-prob, prob])
    images = np.clip(images, -1., 1.)
    return(images)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = (x_train.reshape(60000, 784) / 255)*2-1
x_test = (x_test.reshape(10000, 784) / 255)*2-1
targets = [x_test[i] for i in [2899, 7686, 974, 5516, 1815, 9065, 1587, 2666, 8466, 1304]]

x_train_targets = np.array([targets[i] for i in y_train])
x_test_targets = np.array([targets[i] for i in y_test])

noisy_data_train=np.array([salt_and_pepper(x_train,p) for p in np.arange(0,1.1,0.1)])
noisy_data=np.array([salt_and_pepper(x_test,p) for p in np.arange(0,1.1,0.1)])

x_train_noisy=np.array([noisy_data_train[np.random.randint(11)][i] for i in range(len(x_train))])
x_test_noisy=np.array([noisy_data[np.random.randint(11)][i] for i in range(len(x_test))])

print([np.random.randint(11) for i in range(len(x_test))])
num_classes = 10
train_labels_cat = to_categorical(y_train, num_classes)
test_labels_cat = to_categorical(y_test, num_classes)



print_imm([x_test,x_test_targets,x_test_noisy])
print_imm(noisy_data)


np.savez_compressed("data.npz",x_train=x_train,x_test=x_test,x_train_noisy=x_train_noisy,x_test_noisy=x_test_noisy,x_train_targets=x_train_targets,x_test_targets=x_test_targets,noisy_data=noisy_data,train_labels_cat=train_labels_cat,test_labels_cat=test_labels_cat)
