
from utils import datasets, model

dts = datasets.FashionMnist()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


print('Number of examples that I\'m going to use:\n',
    'Training examples: {}\n'.format(dts.num_train),
    'Test examples: {}'.format(dts.num_test))

#dts.explore_dataset(class_names=class_names,qtd=25)

md = model.NN()

md.compile()

md.train(batch_size=32, epochs=5, datasets=dts.train_dataset, num_examples=dts.num_train )


md.accuracy(dataset=dts.test_dataset, batch_size=32, num_examples=dts.num_test)

#TODO - MAKE PREDICTIONS

#TODO - PLOT PREDICTIONS