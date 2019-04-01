#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def

K.set_learning_phase(0)
K.set_image_data_format('channels_last')

INPUT_MODEL = '/home/techvamp/Documents/Project/retina/training_code/cross_validation_models/Architecture3/cross_validation_v4_arc_red_4lyr.h5'
NUMBER_OF_OUTPUTS = 1
OUTPUT_NODE_PREFIX = 'output_node'
OUTPUT_FOLDER= 'frozen'
OUTPUT_GRAPH = 'frozen_model.pb'
OUTPUT_SERVABLE_FOLDER = 'servable_folder'
INPUT_TENSOR = 'conv2d_7_input_2:0'


try:
    model = load_model(INPUT_MODEL)
except ValueError as err:
    print('Please check the input saved model file')
    raise err

output = [None]*NUMBER_OF_OUTPUTS
output_node_names = [None]*NUMBER_OF_OUTPUTS
for i in range(NUMBER_OF_OUTPUTS):
    output_node_names[i] = OUTPUT_NODE_PREFIX+str(i)
    output[i] = tf.identity(model.outputs[i], name=output_node_names[i])
print('Output Tensor names: ', output_node_names)


sess = K.get_session()
try:
    frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)    
    graph_io.write_graph(frozen_graph, OUTPUT_FOLDER, OUTPUT_GRAPH, as_text=False)
    print(f'Frozen graph ready for inference/serving at {OUTPUT_FOLDER}/{OUTPUT_GRAPH}')
except:
    print('Error Occured')

#This is to The SavedModelBuilder class provides functionality to build a SavedModel protocol buffer. 
#Specifically, this allows multiple meta graphs to be saved as part of a single language-neutral SavedModel, while sharing variables and assets.
builder = tf.saved_model.builder.SavedModelBuilder(OUTPUT_SERVABLE_FOLDER)

with tf.gfile.GFile(f'{OUTPUT_FOLDER}/{OUTPUT_GRAPH}', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}
OUTPUT_TENSOR = output_node_names
with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    #sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        #tf.saved_model.signature_def_utils.predict_signature_def(inputs={'images': model.input},outputs={'scores': model.output})
    signature = predict_signature_def(inputs={'images': model.input},outputs={'scores': model.output})


with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'predict': signature})
    try:
        builder.save()
        print(f'Model ready for deployment at {OUTPUT_SERVABLE_FOLDER}/saved_model.pb')
        print('Prediction signature : ')
        print(signature)
    except:
        print('Error Occured, please checked frozen graph')
