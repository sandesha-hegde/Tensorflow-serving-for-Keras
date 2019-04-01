#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from PIL import Image
from grpc.beta import implementations
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import requests
import numpy as np
import imutils
import cv2

server = 'localhost:9000'
host, port = server.split(':')

image = cv2.imread('/home/techvamp/Documents/Project/test/dog.png')
save_img = imutils.resize(image, width=200)
test = np.asarray(save_img, dtype=np.float32)

# create the RPC stub
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# create the request object and set the name and signature_name params
request = predict_pb2.PredictRequest()
request.model_spec.name = 'classify'
request.model_spec.signature_name = 'predict'

request.inputs['images'].CopyFrom(
  tf.contrib.util.make_tensor_proto(test.astype(dtype=np.float32), shape=[1, test.shape[0], test.shape[1], 3]))

result_future = stub.Predict(request, 50.)
output = result_future.outputs['scores'].float_val
print(output)
result_future
type(result_future)
