# Sushi or sandwich?

Gianandrea Minneci, May 3rd 2017

# The model

The model used is a relatively simple 6 layer CNN (2 convolutional layers, 2 max pools and 2 fully connected layers with dropout). It is implemented in Tensorflow. After a good degree of experimentation, all activation functions are sigmoids - this proved to be better for robust gradient descent methods (in the final version, I use RMSProp). The dropout is set to quite low (keep_prob = 0.9, i.e. not many random nodes are killed), but after a few hundred iterations some overfitting starts creeping in and it can be adjusted down to contrast that.

To train the model, the images are converted to high contrast monochrome; given that the number of examples is quite limited, I had better resuts in this way than by using the full RGB channels. Additional versions of the images are created by rotating them by 90, 180 and 270 degrees. I have experimented with other angles but it didn't seem to increase the performance anymore. They are then rescaled before being split in a small test sample and a training set.

The model is trained using cross entropy, although during training the train / test AUC is also computed. The model is not extremely accurate but it's usable - it achieves a test AUC > 0.8 after 500 training epochs, average cross-entropy < 0.1.

The training is done in the train_model.py script (there is also a notebook version). The same script also downloads and pre-processes the images; it expects only one parameter - the number of epochs to train on. Please ote that it was written for GPUs and may be very slow to train on CPUs. All the CNN parameters are in this script, while some functions and global parameters are in the shared script utils.py.
```sh
$ python train_model.py 100
```
or, if using notebooks:
```sh
$ python-notebook train_model.ipynb
```

# Improvements

The model would benefit from some improvements before deployment. Namely, more advanced transformation of the images (distortions, cropping, rotations at different angles), collecting more samples (from Google Image API) and fine tuning the parameters, as well as running a longer training time.

More specifically, the model currently does not support "resume learning" - it should be refactored so that it can learn incrementally in the background. Loading the exisisting model is done in the testing script below. The testing script itself could be written as an API for ease of use, allowing image upload or batch processing (depending on the use case). 
I also noticed that my best model takes ~1GB of disk space. This can be a hindrance for portability, and is probably due to Tensorflow saving many unnecessary variable (I dump the whole 'session' to disk). This can be refined by carefully selecting the necessary parts of the graph only.
Finally, the Dockerfile is very basic; for instance, although it would benefit from compiling the project on GPUs, it currently runs on CPU only (because of the pip install of Tensorflow). This should be revisited for training (it's OK for out of sample classifying / testing though).

# Testing

Another script (+ notebook) called is_sushi.py is provided. The script loads the model trained previously, and has a function that accepts a jpg image as input (from the set provided or otherwise) and plots it alongside the model's prediction. This can be easily adapted into an API, batched for performance and so on. There is a naive loop at the end where an image is randomly sampled from the set provided and classified interactively. 
```sh
$ python is_sushi.py
```
or, if using notebooks:
```sh
$ python-notebook is_sushi.ipynb
```

