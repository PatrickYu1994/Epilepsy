# Cardiorespiratory function as a non-invasive biomarker for detecting seizures with artificial intelligence: Utilising data from long-term patient recordings from an epilepsy monitoring unit

This is an implementation of the first step of the project on Python 3, Keras and TensorFlow. 

The repository includes:

 	1. Source codes of different models, including basic 1D convolution, 2D convolution and LSTM.
 	2. Feature generation codes for the prepared dataset, which was sampled by Royal Melbourne Hospital.
 	3. Testing codes for the well-trained models.

This is the first stage of the project, the codes are well-documented and designed to be easy to extend. The dataset is not public. If you have any issues about the dataset, please contact shobi.sivathamboo@monash.edu. If you have any problems about the models or project, please contact zongyuan.ge@monash.edu.

## Dataset

This dataset is sampled by patients from Royal Melbourne Hospital and is not public. The dataset has been cleaned, but without artifacts removal. If there are any issues about the dataset, please find the contact information above.

## Getting Started

In each folder, there are 4 python files, which are feature_gen.py, train.py, test.py and utils.py. The details are as follows:

1. feature_gen.py shows how to generate the model required training and testing dataset based on the prepared dataset. All the hyper-parameters are well-listed and easy to change. For the convenience of the research, the generated datasets are stored as .mat (Matlab files). All the functions are modularized and can be extended with additional pre-processing methods.
2. train.py includes two parts. One is the modelling part and another is the training part.
3. test.py is used for the purpose of testing. The best models are stored under the  ./result/best_model directory.
4. utils.py contains the main utilities methods.

Because the feature_gen.py is specified to our prepared dataset, it's not easy to generalize to other public or private datasets. While the train.py, test.py, and utils.py are well-generalize. If you have any problems for feature generation, please contact yliu0098@student.monash.edu.

## Requirements

Python 3.6. platform linux-64

The environment is built by anaconda on massive server. The environment requirements and installation methods can be found in requirement.txt.

conda create --name \<env> --file \<this file>

## Run project

1. Activate the installed environment

2. cd --file \<object model directory>

3. Feature generation with the commands: 

   python feature_gen.py 

4. Model training with the commands: 

   python train.py

5. Model performance testing with the commands:

   python test.py

As mentioned before, all the best models are stored under the ./result/best_model directory. Before running test.py, change the model path with the path of the best model with lowest loss or highest AUC.
