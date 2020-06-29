import tensorflow as tf
import math

class NN():

    def __init__(self, height: int=28, width: int=28, dimension: int=1, neurons: int=128, outuput: int=10):

        #building
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(height,width,dimension)), #flatten the image of 28, 28 in a 1d array of 784
            tf.keras.layers.Dense(neurons, activation=tf.nn.relu), #One layer with 128 Neurons
            tf.keras.layers.Dense(outuput, activation=tf.nn.softmax) #output layer with 10 possibilities
        ])

    def compile(self, optimizer: str='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics: list=['accuracy']):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def train(self,datasets, num_examples: int=0, batch_size: int=32, epochs: int=5):

        train_dataset = datasets.cache().repeat().shuffle(num_examples).batch(batch_size)
        self.model.fit(train_dataset, epochs=epochs, steps_per_epoch=math.ceil(num_examples/batch_size))

    def accuracy(self, dataset, num_examples, batch_size: int=32):

        print('\n\nStarting test')
        dataset = dataset.cache().batch(batch_size)
        test_loss, test_accuracy = self.model.evaluate(dataset, steps=math.ceil(num_examples/batch_size))
        print('Accuracy on test dataset: {}'.format(test_accuracy))
        print('Test Loss: {}'.format(test_loss))

    def prediction(self, dataset, batch_size: int=32):
        dataset = dataset.cache().batch(batch_size)
        for test_images, test_labels in dataset.take(1):
            test_images = test_images.numpy()
            test_labels = test_labels.numpy()

            print(test_images.shape)
            predictions = self.model.predict(test_images)

        return predictions, test_labels, test_images



