import matplotlib.pyplot as plt
import numpy as np
import os


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


def plot_image(predictions_array, true_label, img):

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[...,0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue' #if it is correct we put blue
    else:
        color = 'red' #rong it is red

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array( predictions_array, true_label, test_images, rows: int=5, cols: int=3):
    print('Going a window with the results :)')
    os.environ['KMP_DUPLICATE_LIB_OK']='True' #this is necessary since the OpenMP bring a issue to run multiple process
    num_rows = rows
    num_cols = cols
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(predictions_array=predictions_array[i], true_label=true_label[i], img=test_images[i])
        plt.subplot(num_rows, 2*num_cols, 2*i+2)

        predictions_array_tmp, true_label_tmp = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array_tmp, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array_tmp)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label_tmp].set_color('blue')


    plt.show()