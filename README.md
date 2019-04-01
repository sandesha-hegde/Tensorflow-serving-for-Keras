# Tensorflow-serving for Keras
This repository shows how to host the  keras model using tensorflow serving. This also shows how to write a client for the hosted model.

## Introduction
TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data.

## Requirements
- tensorflow
- tensorflow-model-server
- keras

## Usage
- This script can be used to host the keras model which are saved in `.h5` type file.
- So this script can also used to convert the keras model to tensorflow format(.pb) for serving.
- `tf-client` script is a client for hosted model.

## Insrtuction
- Install the `Requirements`
- Add the keras model path,output folder for tf model storage, output_servable_folder for hosting model in `tf_server.py` script.
- Create a folder called `serving/versions/1` 
- Copy the servable model and variables in the above path
-  Run the tensorflow server: command `tensorflow_model_server --port=<<port number>> --model_name=<<name of the model>> --model_base_path=<<path to this dir serving/versions>>`
- Example: `tensorflow_model_server --port=9000 --model_name=classifier --model_base_path=/home/techvamp/Documents/Project/classifier/tf_serving/serving/versions`
- Testing the hosted model from the client can be done by executing `tf_client.py`






