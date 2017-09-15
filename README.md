# Fashion MNIST - Intro to Deep Learning project.

In this project, I performed basic image preprocessing and deep learning using Keras, a Tensorflow based library. This fashion dataset is similar to handwritten digit dataset MNIST, but for clothes items. Credits to Zalando's Fashion-MNIST project for the dataset. (https://github.com/zalandoresearch/fashion-mnist)

"Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits."

Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |



Three deep learning architecture is used for benchmarking and evaluation: LeNet5, a simpler implementation of VGG19 and a simple architecture with 2 convo layers and no FC layer

Suprisingly, all 3 models have similar accuracy on test set: 0.917, or 91.7% accuracy.



