
from utils import datasets, model, image_handling

dts = datasets.FashionMnist()

print('Number of examples that I\'m going to use:\n',
    'Training examples: {}\n'.format(dts.num_train),
    'Test examples: {}'.format(dts.num_test))

#dts.explore_dataset(class_names=class_names,qtd=25)

md = model.NN(neurons=1024)

md.compile()

md.train(batch_size=32, epochs=5, datasets=dts.train_dataset, num_examples=dts.num_train )

md.accuracy(dataset=dts.test_dataset, batch_size=32, num_examples=dts.num_test)

prediction, labels, images = md.prediction(dataset=dts.test_dataset, batch_size=32)

image_handling.plot_value_array(predictions_array=prediction, true_label=labels, test_images=images)

