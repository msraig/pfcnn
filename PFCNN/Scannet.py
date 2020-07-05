
import tensorflow as tf
from layers import * 
from utils import *
from PFCNN import PFCNN
import argparse
import math 
import numpy as np 
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='Config File')
args = parser.parse_args()
module = __import__(args.config)
FLAGS = module.FLAGS

def parse_function_scannet(example_proto):
    feature_discription = {
        'label': tf.FixedLenFeature([], tf.string),
        'mesh_name': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([], tf.string),
        #'coord': tf.FixedLenFeature([], tf.string),
        'normal': tf.FixedLenFeature([], tf.string),
        'feature_channel': tf.FixedLenFeature([], tf.int64),
        'input_feature': tf.FixedLenFeature([], tf.string),
        'rgb': tf.FixedLenFeature([], tf.string),
        'z': tf.FixedLenFeature([], tf.string),
        'maxpool/offset': tf.FixedLenFeature([], tf.string),
        'maxpool/arg': tf.FixedLenFeature([], tf.string),
        'maxpool/indices': tf.FixedLenFeature([], tf.string),
        'unpooling/indices': tf.FixedLenFeature([], tf.string),
        'conv/offset': tf.FixedLenFeature([], tf.string),
        'conv/indices': tf.FixedLenFeature([], tf.string),
        'conv/weights': tf.FixedLenFeature([], tf.string),
    }
    return tf.parse_single_example(example_proto, feature_discription)


def decode_record_scannet(record, axis_num, level_num, is_training):
    maxpooling_matrix_list = []
    maxpooling_arg_list = []
    unpooling_list = []
    conv_matrix_list = []

    label = tf.decode_raw(record['label'], tf.int32)
    
    rgb = tf.decode_raw(record['rgb'], tf.float32)
    rgb = tf.reshape(rgb, (-1, 3))
    
    normal = tf.decode_raw(record['normal'], tf.float32)
    normal = tf.reshape(normal, (-1, 3))

    #coord = tf.decode_raw(record['coord'], tf.float32)
    #coord = tf.reshape(coord, (-1, 3))

    #center = tf.reduce_mean(coord, axis=0)

    if(is_training):
        theta = tf.random.uniform([], -math.pi, math.pi, dtype=tf.float32)
        rotate_M = tf.convert_to_tensor([[tf.math.cos(theta), -tf.math.sin(theta), 0],[tf.math.sin(theta), tf.math.cos(theta),0], [0, 0, 1]])
        normal = tf.matmul(normal, rotate_M)
        
        #translation = tf.convert_to_tensor([center[0] , center[1], 0])
        #coord = tf.matmul((coord - translation), rotate_M) + translation

    z = tf.decode_raw(record['z'], tf.float32)
    z = tf.expand_dims(z, 1)

    pointwise_input = tf.concat([rgb, normal, z], axis=-1)
    #pointwise_input = tf.concat([rgb, normal, coord], axis=-1)
    pointwise_input = FeatureDuplicate(pointwise_input, axis_num)
    
    mesh_name = record['mesh_name']
    label = tf.cast(label, tf.int32)
    shape_list = tf.decode_raw(record['shape'], tf.int32)
    shape_list = tf.reshape(shape_list, (-1, 4))

    #feature_channel = record['feature_channel']
    #feature_channel = tf.cast(feature_channel, tf.int32)
    input_feature = tf.decode_raw(record['input_feature'], tf.float32)
    #input_feature = tf.reshape(input_feature, (shape_list[0][0]*shape_list[0][1]*shape_list[0][2]*shape_list[0][3], 1))

    maxpooling_offset = tf.decode_raw(record['maxpool/offset'], tf.int32)
    maxpooling_arg_list = tf.decode_raw(record['maxpool/arg'], tf.int32)
    maxpooling_indices_list = tf.decode_raw(record['maxpool/indices'], tf.int32)
    unpooling_indices_list = tf.decode_raw(record['unpooling/indices'], tf.int32)

    conv_offset = tf.decode_raw(record['conv/offset'], tf.int32)
    conv_indices_list = tf.decode_raw(record['conv/indices'], tf.int32)
    #conv_weights_list = tf.decode_raw(record['conv/weights'], tf.float64)
    conv_weights_list = tf.decode_raw(record['conv/weights'], tf.float32)
    conv_begin = 0
    pool_begin = 0
    input_begin = 0

    input_local_feature = []

    for level_i in range(level_num):
        input_feature_leveli = tf.slice(input_feature, [input_begin], [shape_list[level_i][0] * shape_list[level_i][1] * shape_list[level_i][2]* shape_list[level_i][3]])
        input_local_feature.append(input_feature_leveli)
        input_begin += shape_list[level_i][0] * shape_list[level_i][1] * shape_list[level_i][2]* shape_list[level_i][3]

        conv_indices = tf.slice(conv_indices_list, [2 * conv_begin], [2 * conv_offset[level_i]])
        conv_indices = tf.cast(tf.reshape(conv_indices, (conv_offset[level_i], 2)), tf.int64)
        conv_weights = tf.slice(conv_weights_list, [conv_begin], [conv_offset[level_i]])
        conv_matrix_list.append(Conv_Matrix(shape_list[level_i][0], shape_list[level_i][1], shape_list[level_i][2], shape_list[level_i][3], conv_indices, conv_weights))
        conv_begin = conv_begin + conv_offset[level_i]
        
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
    #print(conv_matrix_list[0].shape)
    #print(pointwise_input.shape)
    point_grid_input = tf.sparse_tensor_dense_matmul(conv_matrix_list[0], pointwise_input, adjoint_a=False, adjoint_b=False)
    input_feature = tf.concat([tf.expand_dims(input_local_feature[0], axis=1), point_grid_input], axis=-1)
    return mesh_name, label, shape_list, input_feature, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list, unpooling_list, input_local_feature


def Scannet_network(record, G_num, level_num, conv_shape, class_num, reuse, is_point_feature, feature_channel, drop_out=0):
    F_len = 64

    mesh_name, label, shape_list, input_tensor, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list, unpooling_matrix_list, input_local_feature = decode_record_scannet(record, G_num, level_num, is_training=not reuse)
    grid_input = True

    level0_feature = Conv_ResBlock(input_tensor, conv_shape, 'level0', conv_matrix_list[0], F_len, G_num, 2, reuse, feature_channel=feature_channel, grid_input=grid_input, drop_out=drop_out)
    level1_feature = MaxPooling(level0_feature, maxpooling_matrix_list[0], maxpooling_arg_list[0])
    level1_feature = Conv_ResBlock(level1_feature, conv_shape, 'level1', conv_matrix_list[1], F_len*2, G_num, 2, reuse, drop_out=drop_out)

    level2_feature = MaxPooling(level1_feature, maxpooling_matrix_list[1], maxpooling_arg_list[1])
    level2_feature = Conv_ResBlock(level2_feature, conv_shape, 'level2', conv_matrix_list[2], F_len*2, G_num, 2, reuse, drop_out=drop_out)

    level2_unpooling = AverageUnPooling(level2_feature, unpooling_matrix_list[1])
    level1_concated = tf.concat([level1_feature, level2_unpooling], axis=1)
    level1_f = Conv_ResBlock(level1_concated, conv_shape, 'level1_concated', conv_matrix_list[1], F_len, G_num, 2, reuse, drop_out=drop_out)

    level0_unpooling = AverageUnPooling(level1_f, unpooling_matrix_list[0])
    concated = tf.concat([level0_feature, level0_unpooling], axis=1)
    level0_f = Conv_ResBlock(concated, conv_shape, 'level0_concated', conv_matrix_list[0], F_len, G_num, 2, reuse, drop_out=drop_out)

    level0_f = FeatureReduce(level0_f, G_num)
    
    logits = tf.contrib.layers.fully_connected(level0_f, class_num, activation_fn=None, reuse=reuse, scope="FC2")
    
    class_freq = np.array([27058943,31580147,4774264,3489188,11780978,3990890,6459052,4220890,3569190,3114588,569066,615475,2631772,2124020,697917,368098,308121,305113,348714,5016365], dtype=np.float32)

    class_freq = class_freq / class_freq.sum()

    class_weights = 1/np.log(1.01+class_freq)
    class_weights = tf.expand_dims(class_weights, axis=-1)
    
    #mask label 0
    non_zero_mask = tf.cast(label, tf.bool)
    masked_logits = tf.boolean_mask(logits, non_zero_mask)
    masked_label = tf.boolean_mask(label, non_zero_mask)
    masked_predicted = tf.cast(tf.argmax(masked_logits, axis=1), tf.int32)
    raw_predicted = tf.cast(tf.argmax(logits, axis=1), tf.int32)

    label_count = tf.reduce_sum(masked_label, axis=0)

    output = {}

    output['raw_predicted'] = raw_predicted
    output['masked_label'] = masked_label
    output['masked_predicted'] = masked_predicted
    output['label_count'] = label_count
    normalized_logits = tf.nn.softmax(logits, axis=-1)
    output['normalized_logits'] = normalized_logits

    #v_num * 20
    #masked_label :  [1...20] -> [0...19]
    masked_label_onehot = tf.one_hot(indices=masked_label-1, depth=class_num-1)

    #pred_label : [0...20] -> [-1...19]
    #onehot : -1 -> [0, 0, ... ,0]
    masked_pred_onehot = tf.one_hot(indices=masked_predicted-1, depth=class_num-1)

    loss_weights = tf.matmul(masked_label_onehot, class_weights)
    loss_weights = tf.squeeze(loss_weights)

    cross_entropy = tf.reduce_mean(tf.multiply(loss_weights, tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_logits, labels=masked_label)))

    class_corr_sum = tf.reduce_sum(tf.multiply(masked_label_onehot, masked_pred_onehot), axis=0)
    class_corr_sum = tf.cast(class_corr_sum, tf.int32)

    Union = tf.reduce_sum(tf.cast(tf.cast(tf.add(masked_label_onehot, masked_pred_onehot), tf.bool), tf.int32), axis=0)
    Union = tf.cast(Union, tf.float32)

    #v_num * 1
    acc = tf.reduce_mean(tf.cast(tf.equal(masked_predicted, masked_label), tf.float32))
    acc = tf.expand_dims(acc, axis=0)
    
    label_sum = tf.reduce_sum(masked_label_onehot, axis=0)

    output['accuracy'] = acc
    output['name'] = mesh_name
    output['loss'] = cross_entropy
    output['logits'] = masked_logits
    output['predicted'] = masked_predicted
    output['vnum'] = tf.shape(logits)[0]
    output['class_corr_sum'] = class_corr_sum
    output['class_Union'] = Union
    output['label'] = label
    output['label_sum'] = label_sum
    return output

class PFCNN_Scannet(PFCNN):
    def __init__(self, FLAGS):
        super(PFCNN_Scannet, self).__init__(parse_function_scannet, Scannet_network, FLAGS)
    
    def build_dataset(self):
        train_names = []
        for i in range(self.flags.train_part):
            train_names.append(self.flags.tfrecord_path + self.flags.train_data + "_%d.tfrecords" % i)
        self.train_data = self.parse_dataset(train_names)
        test_names = []
        for i in range(self.flags.test_part):
            test_names.append(self.flags.tfrecord_path + self.flags.test_data + "_%d.tfrecords" % i)
        self.test_data = self.parse_dataset(test_names) 

    def build_summary(self):
        super(PFCNN_Scannet, self).build_summary()
        self.test_set_mA = tf.placeholder(dtype=tf.float32, name="test_set_mA")
        self.test_mA_summary = tf.summary.scalar('test_mean_accuracy', self.test_set_mA)

        self.test_set_IOU = tf.placeholder(dtype=tf.float32, name="test_set_IOU")
        self.test_IOU_summary = tf.summary.scalar('test_mean_IOU', self.test_set_IOU)

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            start_epoch = 0
            init = tf.global_variables_initializer()
            sess.run(init)
            if(self.flags.is_load_model):
                self.saver.restore(sess, self.flags.model_path)
                start_epoch = self.flags.start_epoch
                print(self.flags.model_path)
                print("load model success!")
            train_writer = tf.summary.FileWriter(self.flags.summaries_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(self.flags.summaries_dir + '/test')
            best_acc = 0

            if not os.path.exists(self.flags.summaries_dir+'/ckpt'):
                os.mkdir(self.flags.summaries_dir+'/ckpt')
            if not os.path.exists(self.flags.summaries_dir+'/pred'):
                os.mkdir(self.flags.summaries_dir+'/pred')
                os.mkdir(self.flags.summaries_dir+'/logits')
                
            lr = self.flags.learning_rate
            for epoch_i in range(start_epoch, start_epoch + self.flags.epoch_num):
                if(epoch_i > 0 and epoch_i % self.flags.decay_epoch == 0 and lr>1e-4):
                    lr = lr / 2
                print("\nepoch " + str(epoch_i) + " training:")
                print("learning rate: "+ str(lr))
                train_acc_sum = 0.0
                for iter_j in range(self.flags.train_size):
                    training_output, merged_sum, _ = sess.run([self.train_output, self.merged_train_summ, self.train_step], feed_dict={self.learning_rate: lr})
                    train_writer.add_summary(merged_sum, iter_j + self.flags.train_size * epoch_i)
                    train_acc_sum = train_acc_sum + training_output['accuracy'].sum() / len(training_output['accuracy'])
                print("epoch " + str(epoch_i) + " training acc:" + str(train_acc_sum / self.flags.train_size) + "\n")
                train_ave_acc = sess.run(self.train_acc_summary, feed_dict={self.train_set_accuracy: train_acc_sum / self.flags.train_size})
                train_writer.add_summary(train_ave_acc, epoch_i)
                if(epoch_i % self.flags.test_epoch == 0):
                    test_acc_sum = 0.0
                    result_list = []
                    Union_list = []
                    label_count_list = []
                    for iter_j in range(self.flags.test_size):
                        testing_output, test_summary = sess.run([self.test_output, self.merged_test_summ])
                        result_list.append(testing_output['class_corr_sum'])
                        Union_list.append(testing_output['class_Union'])
                        label_count_list.append(testing_output['label_count'])
                        test_writer.add_summary(test_summary, iter_j + self.flags.test_size * epoch_i)
                        test_acc_sum = test_acc_sum + testing_output['accuracy'].sum() / len(testing_output['accuracy'])
                    print("epoch " + str(epoch_i) + " testing acc:" + str(test_acc_sum / self.flags.test_size) + "\n")
                    test_ave_acc = sess.run(test_acc_summary, feed_dict={test_set_accuracy: test_acc_sum / self.flags.test_size})
                    test_writer.add_summary(test_ave_acc, epoch_i)

                    result_list = np.array(result_list)
                    pred_sum = np.sum(result_list, axis=0)
                    
                    label_count_list = np.array(label_count_list)
                    label_sum = np.sum(label_count_list)
                    
                    class_acc = (pred_sum/label_sum)
                    mean_acc = np.mean(class_acc)
                    print("testing mean acc:", mean_acc)

                    Union_list = np.array(Union_list)
                    Union_sum = np.sum(Union_list, axis=0)
                    class_IOU = (pred_sum/Union_sum)
                    mean_IOU = np.mean(class_IOU)
                    print("testing mean IOU:", mean_IOU)

                    test_IOU_acc = sess.run(self.test_IOU_summary, feed_dict={self.test_set_IOU: mean_IOU})
                    test_writer.add_summary(test_IOU_acc, epoch_i)

                    test_mA_acc = sess.run(self.test_mA_summary, feed_dict={self.test_set_mA: mean_acc})
                    test_writer.add_summary(test_mA_acc, epoch_i)

                    if(test_acc_sum / self.flags.test_size >= best_acc):
                        best_acc = test_acc_sum / self.flags.test_size
                        self.best_saver.save(sess, self.flags.summaries_dir+'/ckpt/' + self.flags.task + '_best.ckpt', global_step=epoch_i)

                if(epoch_i % self.flags.save_epoch == 0):
                    self.saver.save(sess, self.flags.summaries_dir+'/ckpt/' + self.flags.task + '.ckpt' , global_step=epoch_i)

if __name__ == "__main__":
    model = PFCNN_Scannet(FLAGS)
    model.build_graph()
    model.build_summary()
    model.train()

