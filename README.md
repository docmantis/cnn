Pet vs. Person CNN Classifier

This repository contains a simple Convolutional Neural Network (CNN) model to classify images as either a pet or a person. The model is built using TensorFlow and Keras, and it is trained on a custom dataset.



Overview

The purpose of this project is to demonstrate the creation and usage of a CNN model for image classification. The model is trained on a dataset containing images of pets and people, and it is capable of distinguishing between the two categories.



Installation and Setup

Ensure you have Python 3 and the following dependencies installed:

        TensorFlow
        Keras
        Matplotlib
        scikit-learn
        NumPy
        Pandas

You can install the required packages using pip:

        pip install tensorflow keras matplotlib scikit-learn numpy pandas



Usage

Update the train_data_dir and test_data_dir variables in the script to point to your own training and testing data directories, respectively.

Run the script:

        python pet_person_cnn_classifier.py

The script will load the dataset, perform data augmentation, build and compile the CNN model, and then train the model using the training data.

Once the model is trained, it will evaluate its performance on the test data and print the test accuracy and confusion matrix.



Dataset

The dataset used in this project is not provided in the repository. You should create your own dataset containing images of pets and people. Organize the dataset into two separate folders for training and testing data. Each folder should contain two subfolders, one for pet images and another for person images.



Dependencies

        TensorFlow 2.x
        Keras
        Matplotlib
        scikit-learn
        NumPy
        Pandas



Acknowledgements

This project is built using open-source libraries and tools. A big thank you to the developers and maintainers of these projects.



License

This project is licensed under the MIT License. See the LICENSE file for more information.



Contact Information

For any questions or suggestions, please feel free to contact us at aivocadotech@gmail.com.

