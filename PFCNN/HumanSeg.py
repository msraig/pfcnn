
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


def parse_function_seg(example_proto):
    feature_discription = {
        'mesh_name': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string),
        'feature_channel': tf.FixedLenFeature([], tf.int64),
        'input_feature': tf.FixedLenFeature([], tf.string),
        'maxpool/offset': tf.FixedLenFeature([], tf.string),
        'maxpool/arg': tf.FixedLenFeature([], tf.string),
        'maxpool/indices': tf.FixedLenFeature([], tf.string),
        'unpooling/indices': tf.FixedLenFeature([], tf.string),
        'conv/offset': tf.FixedLenFeature([], tf.string),
        'conv/indices': tf.FixedLenFeature([], tf.string),
        'conv/weights': tf.FixedLenFeature([], tf.string),
    }
    return tf.parse_single_example(example_proto, feature_discription)

def decode_record_seg(record, level_num, is_point_feature):
    mesh_name = record['mesh_name']
    label = tf.decode_raw(record['label'], tf.int32)
    label = tf.cast(label, tf.int32)
    shape_list = tf.decode_raw(record['shape'], tf.int32)
    shape_list = tf.reshape(shape_list, (-1, 4))

    if(is_point_feature):
        input_feature = tf.decode_raw(record['input_feature'], tf.float32)
        feature_channel = record['feature_channel']
        feature_channel = tf.cast(feature_channel, tf.int32)
        input_feature = tf.reshape(input_feature, (shape_list[0][0], feature_channel))
    else:
        feature_channel = record['feature_channel']
        feature_channel = tf.cast(feature_channel, tf.int32)
        input_feature = tf.decode_raw(record['input_feature'], tf.float32)
        input_feature = tf.reshape(input_feature, (shape_list[0][0]*shape_list[0][1]*shape_list[0][2]*shape_list[0][3], feature_channel))

    conv_offset = tf.decode_raw(record['conv/offset'], tf.int32)
    conv_indices_list = tf.decode_raw(record['conv/indices'], tf.int32)
    conv_weights_list = tf.decode_raw(record['conv/weights'], tf.float32)

    conv_begin = 0
    conv_matrix_list = []
    for level_i in range(level_num):
        conv_indices = tf.slice(conv_indices_list, [2 * conv_begin], [2 * conv_offset[level_i]])
        conv_indices = tf.cast(tf.reshape(conv_indices, (conv_offset[level_i], 2)), tf.int64)
        conv_weights = tf.slice(conv_weights_list, [conv_begin], [conv_offset[level_i]])
        conv_matrix_list.append(Conv_Matrix(shape_list[level_i][0], shape_list[level_i][1], shape_list[level_i][2], shape_list[level_i][3], conv_indices, conv_weights))
        conv_begin = conv_begin + conv_offset[level_i]
    
    maxpooling_offset = tf.decode_raw(record['maxpool/offset'], tf.int32)
    maxpooling_arg_list = tf.decode_raw(record['maxpool/arg'], tf.int32)
    maxpooling_indices_list = tf.decode_raw(record['maxpool/indices'], tf.int32)
    unpooling_indices_list = tf.decode_raw(record['unpooling/indices'], tf.int32)

    maxpooling_matrix_list = []
    unpooling_list = []

    pool_begin = 0
    for level_i in range(level_num - 1):
        maxpooling_indices = tf.slice(maxpooling_indices_list, [2 * pool_begin], [2 * maxpooling_offset[level_i]])
        maxpooling_indices = tf.cast(tf.reshape(maxpooling_indices, (maxpooling_offset[level_i], 2)), tf.int64)
        maxpooling_values = tf.ones(shape=[maxpooling_offset[level_i]], dtype=tf.float32)

        maxpooling_matrix_list.append(MaxPooling_Matrix(shape_list[level_i][0], shape_list[level_i + 1][0], shape_list[level_i][1], maxpooling_indices, maxpooling_values, maxpooling_arg_list[level_i]))

        unpooling_indices = tf.slice(unpooling_indices_list, [2 * pool_begin], [2 * maxpooling_offset[level_i]])
        unpooling_values = tf.ones(shape=[maxpooling_offset[level_i]], dtype=tf.float32)
        unpooling_indices = tf.cast(tf.reshape(unpooling_indices, (maxpooling_offset[level_i], 2)), tf.int64)
        unpooling_list.append(UnPooling_Matrix(shape_list[level_i][0], shape_list[level_i + 1][0], shape_list[level_i][1], unpooling_indices, unpooling_values))

        pool_begin = pool_begin + maxpooling_offset[level_i]
    return mesh_name, label, shape_list, input_feature, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list, unpooling_list

def Segmentation_network(record, G_num, level_num, conv_shape, class_num, reuse, is_point_feature, feature_channel, drop_out=0):
    F_len = 32
    mesh_name, label, shape_list, input_tensor, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list, unpooling_matrix_list = decode_record_seg(record, level_num, is_point_feature)
    if(is_point_feature):
        input_tensor = tf.reshape(input_tensor, (-1, feature_channel))
        input_tensor = FeatureDuplicate(input_tensor, G_num)
        grid_input = False
    else:
        grid_input = True
    
    label = label - 1

    level0_feature = Conv_ResBlock(input_tensor, conv_shape, 'level0', conv_matrix_list[0], F_len, G_num, 2, reuse, feature_channel=feature_channel, grid_input=grid_input, drop_out=drop_out)
    level1_feature = MaxPooling(level0_feature, maxpooling_matrix_list[0], maxpooling_arg_list[0])
    level1_feature = Conv_ResBlock(level1_feature, conv_shape, 'level1', conv_matrix_list[1], F_len, G_num, 2, reuse, drop_out=drop_out)

    level2_feature = MaxPooling(level1_feature, maxpooling_matrix_list[1], maxpooling_arg_list[1])
    level2_feature = Conv_ResBlock(level2_feature, conv_shape, 'level2', conv_matrix_list[2], F_len, G_num, 2, reuse, drop_out=drop_out)

    level2_unpooling = AverageUnPooling(level2_feature, unpooling_matrix_list[1])
    level1_concated = tf.concat([level1_feature, level2_unpooling], axis=1)
    level1_f = Conv_ResBlock(level1_concated, conv_shape, 'level1_concated', conv_matrix_list[1], F_len, G_num, 2, reuse, drop_out=drop_out)

    level0_unpooling = AverageUnPooling(level1_f, unpooling_matrix_list[0])
    concated = tf.concat([level0_feature, level0_unpooling], axis=1)
    level0_f = Conv_ResBlock(concated, conv_shape, 'level0_concated', conv_matrix_list[0], F_len, G_num, 2, reuse, drop_out=drop_out)

    level0_f = tf.reshape(level0_f, (-1, G_num * F_len))
    #F_num x 4 * F_len
    level0_f = tf.contrib.layers.fully_connected(level0_f, 128, activation_fn=tf.nn.relu, reuse=reuse, scope="FC1")
    logits = tf.contrib.layers.fully_connected(level0_f, class_num, activation_fn=None, reuse=reuse, scope="FC2")
    #F_num * 8
    ground_truth = tf.one_hot(indices=label, depth=class_num)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ground_truth))
    predicted = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    #F_num * 1
    acc = tf.reduce_mean(tf.cast(tf.equal(predicted, label), tf.float32))
    acc = tf.expand_dims(acc, axis=0)
    output = {}
    output['accuracy'] = acc
    output['name'] = mesh_name
    output['loss'] = cross_entropy
    output['logits'] = logits
    output['predicted'] = predicted
    output['label'] = label
    return output



if __name__ == "__main__":
    model = PFCNN(parse_function_seg, Segmentation_network, FLAGS)
    model.build_graph()
    model.build_summary()
    model.train()

