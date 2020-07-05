
import tensorflow as tf
from layers import * 
from utils import *
from PFCNN import PFCNN
import argparse
from HumanSeg import parse_function_seg, decode_record_seg
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='Config File')
args = parser.parse_args()
module = __import__(args.config)
FLAGS = module.FLAGS

def Matching_network(record, G_num, level_num, conv_shape, class_num, reuse, is_point_feature, feature_channel, drop_out=0):
    F_len = 64

    mesh_name, label, shape_list, input_tensor, _, _, conv_matrix_list, _ = decode_record_seg(record, level_num, is_point_feature)
    if(is_point_feature):
        input_tensor = tf.reshape(input_tensor, (-1, feature_channel))
        #print(input_tensor.shape)
        duplicated = FeatureDuplicate(input_tensor, G_num)
        grid_input = False
    else:
        grid_input = True

    #input_tensor : (v.a.m.n) * (3:4)
    level0_feature = Conv_ResBlock(input_tensor, conv_shape, 'level0', conv_matrix_list[0], F_len, G_num, 4, reuse, feature_channel=feature_channel, grid_input=grid_input, drop_out=drop_out)
    
    reduced = FeatureReduce(level0_feature, G_num)
    level0_f = tf.contrib.layers.fully_connected(reduced, F_len * G_num, activation_fn=tf.nn.relu, reuse=reuse, scope="FC1")
    logits = tf.contrib.layers.fully_connected(level0_f, class_num, activation_fn=None, reuse=reuse, scope="FC2")

    ground_truth = tf.one_hot(indices=label, depth=class_num)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ground_truth))
    predicted = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    #F_num * 1
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted, label), tf.float32))
    acc = tf.expand_dims(acc, axis=0)
    normalized = tf.nn.softmax(logits, axis=1)
    max_logits = tf.reduce_max(normalized, axis=1)

    output = {}
    output['accuracy'] = acc
    output['name'] = mesh_name
    output['loss'] = cross_entropy
    output['logits'] = logits
    output['predicted'] = predicted
    output['label'] = label
    return output

if __name__ == "__main__":
    model = PFCNN(parse_function_seg, Matching_network, FLAGS)
    model.build_graph()
    model.build_summary()
    model.train()

