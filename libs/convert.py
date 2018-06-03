import json
import keras.backend as K
from libs.args import args
from libs.version import __version__
import tensorflow as tf
from tensorflowjs.converters.tf_saved_model_conversion import convert_tf_frozen_model
import shutil
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


def get_tf_folder_name():
    return 'models/ne%ix-%s-%s-%s_tf' % (args.zoom, args.type, args.model, __version__)


def get_tfjs_folder_name():
    return 'models/ne%ix-%s-%s-%s_tfjs' % (args.zoom, args.type, args.model, __version__)


def get_inputs_outputs(model):
    # build inputs
    input_names = model.input_names
    inputs = {}
    for i in range(len(input_names)):
        input_name = '%s_%i' % (input_names[i], (i + 1))
        inputs[input_name] = model.inputs[i]

    # build outputs
    output_names = model.output_names
    outputs = {}
    for i in range(len(output_names)):
        output_name = '%s_%i' % (output_names[i], (i + 1))
        outputs[output_name] = model.outputs[i]

    return inputs, outputs


def save_keras_as_frozen_tf(model):
    print('Saving Keras session as frozen TF model')
    # try to remove previous version
    try:
        shutil.rmtree(get_tf_folder_name())
    except Exception:
        pass

    K.set_learning_phase(0)
    sess = K.get_session()
    # inputs, outputs = get_inputs_outputs(model)

    output_names = model.output_names
    num_output = len(output_names)
    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = '%s_%i' % (model.output_names[i], (i + 1))
        pred[i] = tf.identity(model.outputs[i], name=pred_node_names[i])

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, get_tf_folder_name(), 'model.pb', as_text=False)

    print('TF model saved to %s' % get_tf_folder_name())


def save_keras_as_frozen_tfjs(model):
    # we first need to save keras to tf
    save_keras_as_frozen_tf(model)
    print('Converting TF frozen model to TFJS frozen model')
    # try to remove previous version
    try:
        shutil.rmtree(get_tfjs_folder_name())
    except Exception:
        pass
    output_names = []
    for i in range(len(model.output_names)):
        output_name = '%s_%i' % (model.output_names[i], (i + 1))
        output_names.append(output_name)

    convert_tf_frozen_model(
        '%s/model.pb' % get_tf_folder_name(),
        ','.join(output_names),
        get_tfjs_folder_name())

    print('TFJS model saved to %s' % get_tfjs_folder_name())

    # make config file
    ib, iw, ih, ic = model.input_shape
    ob, ow, oh, oc = model.input_shape

    config = {
        "input": {
            "name": model.input_names[0],
            "shape": [None, iw, ih, ic]
        },
        "output": {
            "name": output_names[0],
            "shape": [None, ow, oh, oc]
        }
    }

    with open('%s/config.json' % get_tfjs_folder_name(), 'w') as outfile:
        json.dump(config, outfile)


if __name__ == "__main__":
    from keras.models import load_model
    from libs.losses import PSNRLoss
    from libs.layers import SubPixelUpscaling

    model = load_model(
        'models/ne2x-photo-enlarge-0.0.1.h5',
        custom_objects={
            'SubPixelUpscaling': SubPixelUpscaling,
            'PSNRLoss': PSNRLoss,
        }
    )

    save_keras_as_frozen_tfjs(model)

