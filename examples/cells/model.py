#!/usr/bin/env python3

'''
Keras is used to setup a network that performs cell segmentation on the input.
Pre-trained weights are available in `weights.h5`. A ProtoBuf definition of the
frozen model is written to `model.pb`.
'''

from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.layers import concatenate, Reshape, Permute, Lambda
from keras.activations import softmax
from keras.optimizers import *
from keras import backend as K
from keras.utils import np_utils

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def weighted_matrix_loss(y_true, y_pred):
    # in y_true: (None, dimsx, dimsy, 4 channels where three first represent a
    # 1-hot-vector for the class and the next one is the weight)
    y_pred_c = K.clip(y_pred, K.epsilon(), 1)

    elms = y_true[:, :, :, :3] * K.log(y_pred_c)
    elms = K.sum(elms, 3) * y_true[:, :, :, 3]
    loss = -K.mean(elms, (0, 1, 2))

    return loss


n_classes = 3
patch_size = 112

inputs = Input((patch_size, patch_size, 16))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
    2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
    2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
    2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
    2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv10 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv11 = Conv2D(n_classes, (1, 1), activation='linear')(conv10)

out = Lambda(lambda x: softmax(x, axis=3))(conv11)

model = Model(inputs=[inputs], outputs=[out])

# The accuracy metric throws off the model importer in Java.
model.compile(optimizer='adam', loss=weighted_matrix_loss)
model.load_weights('weights.h5')
print(model.summary())

# Fails for unknown reasons.
# print('Saving Tensorflow checkpoint (.ckpt)')
# saver = tf.train.Saver()
# saver.save(K.get_session(), 'model.ckpt')


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    '''
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.

    @param session         The TensorFlow session to be frozen.
    @param keep_var_names  A list of variable names that should not be frozen,
                           or None to freeze all the variables in the graph.
    @param output_names    Names of the relevant graph outputs.
    @param clear_devices   Remove the device directives from the graph for
                           better portability.
    @return                The frozen graph definition.
    '''

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables())
                                .difference(keep_var_names or []))

        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ''

        frozen_graph = convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


# output_names = [out.op.name for out in model.outputs]
# print('Output names: {}'.format(output_names))
# frozen_graph = freeze_session(K.get_session(), output_names=output_names)
# tf.train.write_graph(frozen_graph, '.', 'unet_cells.pb', as_text=False)

# from tensorflow.python.saved_model import builder as saved_model_builder
# from tensorflow.python.saved_model import utils
# from tensorflow.python.saved_model import tag_constants, signature_constants
# from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
# from tensorflow.contrib.session_bundle import exporter

# K.set_learning_phase(0)
# builder = saved_model_builder.SavedModelBuilder('tf_model')
# signature = predict_signature_def(inputs={'in': model.input},
#                                   outputs={'out': model.output})

# with K.get_session() as sess:
#     builder.add_meta_graph_and_variables(
#         sess=sess, tags=[tag_constants.SERVING],
#         signature_def_map={'predict': signature})
#     builder.save()

builder = tf.saved_model.builder.SavedModelBuilder('tf_model')
builder.add_meta_graph_and_variables(K.get_session(), ['unet_cells'])
builder.add_meta_graph(['unet_cells'])
builder.save()
