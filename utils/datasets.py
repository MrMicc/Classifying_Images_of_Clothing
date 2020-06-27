import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import logging


class FashionMnist():
    def __init__(self):
        self._dataset, self._metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

        self.train_dataset, self.test_dataset = self._dataset['train'], self._dataset['test']

        # Normalizing each element via map function
        self.train_dataset = self.train_dataset.map(FashionMnist.normalize)
        self.test_dataset = self.test_dataset.map(FashionMnist.normalize)

        # putting everythin on cache
        self.train_dataset = self.train_dataset.cache()
        self.test_dataset = self.test_dataset.cache()

        self.num_train = self._metadata.splits['train'].num_examples
        self.num_test = self._metadata.splits['test'].num_examples

        log = tf.get_logger()
        log.setLevel(logging.DEBUG)

    @staticmethod
    def normalize(images, labels):
        '''This function normalize the image to has a range [0,1]. Before the normalize it has a range [0,255]'''
        images = tf.cast(images, tf.float32)
        images /= 255

        return images, labels

    def explore_dataset(self, class_names: list, qtd: str = 1):
        '''
        Is to check examples how the image is setup in the dataset
        :param class_names: name of the classes
        :param qtd: number of picture to show
        :return: None
        '''
        plt.figure(figsize=(10, 10))
        i = 0
        for image, label in self.test_dataset.take(qtd):
            image = image.numpy().reshape((28, 28))
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=plt.cm.binary)
            plt.xlabel(class_names[label])
            i += 1
        plt.show()
