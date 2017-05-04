# Sushi or sandwich?

Gianandrea Minneci, May 3rd 2017

# The model

The model used is a relatively simple 6 layer CNN (2 convolutional layers, 2 max pools and 2 fully connected layers with dropout). It is implemented in Tensorflow.

To train the model, the images are converted to grayscale (given that the number of examples is quite limited), and then rotated by 90, 180 and 270 degrees. They are then rescaled before being split in a small test sample and a training set.

The model is trained using cross entropy, although during training the train / test AUC is also computed. The model is not extremely accurate but it's usable - it achieves a test AUC > 0.8 after 500 training epochs, average cross-entropy < 0.1.

The training is done in the train_model.py script (there is also a notebook version). The scripts expects only one parameter - the number of epochs to train on.

# Improvements

The model would benefit from some improvements before ready for deployment. Namely, more advanced transformation of the images, collecting more samples and fine tuning the parameters, as well as running a longer training time.

From a technical point of view, the model should be reformatted so that it can learn incrementally and the testing script below should be written as an API for ease of use.

# Testing

Another script (+ notebook) called is_sushi.py is provided. The script loads the trained model, then accepts an image as input (from the set provided or otherwise) and plots it alongside the model's prediction. This can be easily adapted into an API, batched for performance and so on.
