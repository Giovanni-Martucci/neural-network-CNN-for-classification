# neural-network-CNN-for-classification


The task of this project is to "classify the different touches of the various knobs in a domestic oven". Specifically, given an oven, our algorithm, through a custom-trained neural network, must be able to recognize whether no knob is being touched or, if so, which one specifically.

In this project, we had to work with images, used as input for our network. This is why a Convolutional Neural Network (CNN) was used, which allows neural networks to be applied to image processing, scaling large images and large datasets of images.

This problem of touch/non-touch classification in a domestic oven can actually be extended to industrial machinery, thinking of actions that result from certain choices. For example, it is possible to imagine that after touching a certain button, the system starts a specific action or displays instructions related to that button.

PS: If you want to replicate the training of the network, you need to create the dataset again by manually dividing the videos or using any other software( like Roboflow) within the Video test folder and dividing them into the 4 dataset folders (left, right, center, and null).

#Usage
To use the following neural network, you need to run the Progetto.py script with the path of the video to analyze
