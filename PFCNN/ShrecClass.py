
import tensorflow as tf
from layers import * 
from utils import *
from PFCNN import PFCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='Config File')
args = parser.parse_args()
module = __import__(args.config)
FLAGS = module.FLAGS

def parse_function_class(example_proto):
    feature_discription = {
        'label': tf.FixedLenFeature([], tf.int64),
        'mesh_id': tf.FixedLenFeature([], tf.int64),
        'shape':  tf.FixedLenFeature([], tf.string),
        'feature_channel': tf.FixedLenFeature([], tf.int64),
        'input_feature': tf.FixedLenFeature([], tf.string),
        'maxpool/offset': tf.FixedLenFeature([], tf.string),
        'maxpool/arg': tf.FixedLenFeature([], tf.string),
        'maxpool/indices': tf.FixedLenFeature([], tf.string),
        'conv/offset': tf.FixedLenFeature([], tf.string),
        'conv/indices': tf.FixedLenFeature([], tf.string),
        'conv/weights': tf.FixedLenFeature([], tf.string)
    }
    return tf.parse_single_example(example_proto, feature_discription)


def decode_record_class(record, level_num, is_point_feature):
    maxpooling_matrix_list = []
    maxpooling_arg_list = []
    conv_matrix_list = []
    label = record['label']
    mesh_id = record['mesh_id']
    label = tf.cast(label, tf.int32)
    shape_list = tf.decode_raw(record['shape'], tf.int32)
    shape_list = tf.reshape(shape_list, (-1, 4))
    if(is_point_feature):
        input_feature = tf.decode_raw(record['points_coord'], tf.float32)
    else:
        feature_channel = record['feature_channel']
        feature_channel = tf.cast(feature_channel, tf.int32)
        input_feature = tf.decode_raw(record['input_feature'], tf.float32)
        input_feature = tf.reshape(input_feature, (shape_list[0][0]*shape_list[0][1]*shape_list[0][2]*shape_list[0][3], feature_channel))
    maxpooling_offset = tf.decode_raw(record['maxpool/offset'], tf.int32)
    maxpooling_arg_list = tf.decode_raw(record['maxpool/arg'], tf.int32)
    maxpooling_indices_list = tf.decode_raw(record['maxpool/indices'], tf.int32)
    conv_offset = tf.decode_raw(record['conv/offset'], tf.int32)
    conv_indices_list = tf.decode_raw(record['conv/indices'], tf.int32)
    conv_weights_list = tf.decode_raw(record['conv/weights'], tf.float32)
    conv_begin = 0
    pool_begin = 0
    for level_i in range(level_num):
        conv_indices = tf.slice(conv_indices_list, [2 * conv_begin], [2 * conv_offset[level_i]])
        conv_indices = tf.cast(tf.reshape(conv_indices, (conv_offset[level_i], 2)), tf.int64)
        conv_weights = tf.slice(conv_weights_list, [conv_begin], [conv_offset[level_i]])
        conv_matrix_list.append(Conv_Matrix(shape_list[level_i][0], shape_list[level_i][1], shape_list[level_i][2], shape_list[level_i][3], conv_indices, conv_weights))
        conv_begin = conv_begin + conv_offset[level_i]
        
    for level_i in range(level_num - 1):
        maxpooling_indices = tf.slice(maxpooling_indices_list, [2 * pool_begin], [2 * maxpooling_offset[level_i]])
        maxpooling_indices = tf.cast(tf.reshape(maxpooling_indices, (maxpooling_offset[level_i], 2)), tf.int64)
        #maxpooling_values = tf.ones(shape=[maxpooling_offset[level_i]], dtype=tf.float64)
        maxpooling_values = tf.ones(shape=[maxpooling_offset[level_i]], dtype=tf.float32)

        pool_begin = pool_begin + maxpooling_offset[level_i]

        maxpooling_matrix_list.append(MaxPooling_Matrix(shape_list[level_i][0], shape_list[level_i + 1][0], shape_list[level_i][1], maxpooling_indices, maxpooling_values, maxpooling_arg_list[level_i]))

    return mesh_id, label, shape_list, input_feature, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list


def Classification_network(record, G_num, level_num, conv_shape, class_num, reuse, is_point_feature, feature_channel, drop_out=0):
    F_len = 64
    mesh_id, label, shape_list, input_tensor, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list = decode_record_class(record, level_num, is_point_feature)
    
    if(is_point_feature):
        input_tensor = tf.reshape(input_tensor, (-1, feature_channel))
        input_tensor = FeatureDuplicate(input_tensor, G_num)
        grid_input = False
    else:
        grid_input = True

    level0_feature = Conv_ResBlock(input_tensor, conv_shape, 'level0', conv_matrix_list[0], F_len, G_num, 1, reuse, feature_channel=feature_channel, grid_input=grid_input, drop_out=drop_out)

    level1_feature = MaxPooling(level0_feature, maxpooling_matrix_list[0], maxpooling_arg_list[0])

    level1_feature = Conv_ResBlock(level1_feature, conv_shape, 'level1', conv_matrix_list[1], F_len, G_num, 1, reuse, drop_out=drop_out)

    level2_feature = MaxPooling(level1_feature, maxpooling_matrix_list[1], maxpooling_arg_list[1])
    
    level2_feature = Conv_ResBlock(level2_feature, conv_shape, 'level2', conv_matrix_list[2], F_len, G_num, 1, reuse, drop_out=drop_out)

    level0_reduced = FeatureReduce(level0_feature, G_num)
    level1_reduced = FeatureReduce(level1_feature, G_num)
    level2_reduced = FeatureReduce(level2_feature, G_num)


    
    level0_mean = tf.reduce_mean(level0_reduced, axis=0, keepdims=False)
    level1_mean = tf.reduce_mean(level1_reduced, axis=0, keepdims=False)  
    level2_mean = tf.reduce_mean(level2_reduced, axis=0, keepdims=False)
    concated = tf.concat([level0_mean, level1_mean, level2_mean], axis=0)

    concated = tf.expand_dims(concated, 0)

    logits = tf.contrib.layers.fully_connected(concated, class_num, activation_fn=None, reuse=reuse, scope="FC")
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ground_truth))
    #batch size = 1
    label = tf.expand_dims(label, 0)
    #acc = tf.equal(tf.cast(tf.argmax(logits, axis=1), tf.int64), label)
    predicted = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    acc = tf.equal(predicted, label)
    output = {}
    output['accuracy'] = acc
    output['name'] = mesh_id
    output['loss'] = cross_entropy
    output['logits'] = logits
    output['predicted'] = predicted
    output['label'] = label
    return output

if __name__ == "__main__":
    model = PFCNN(parse_function_class, Classification_network, FLAGS)
    model.build_graph()
    model.build_summary()
    model.train()

