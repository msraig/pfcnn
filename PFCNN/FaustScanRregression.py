
import tensorflow as tf
from layers import * 
from utils import *
from PFCNN import PFCNN
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='Config File')
args = parser.parse_args()
module = __import__(args.config)
FLAGS = module.FLAGS

def parse_function_regression(example_proto):
    feature_discription = {
        'label': tf.FixedLenFeature([], tf.string),
        'mesh_name': tf.FixedLenFeature([], tf.string),
        'shape': tf.FixedLenFeature([], tf.string),
        'feature_channel': tf.FixedLenFeature([], tf.int64),
        'input_feature': tf.FixedLenFeature([], tf.string),
        'e_num': tf.FixedLenFeature([], tf.int64),
        'e_pos': tf.FixedLenFeature([], tf.string),
        'maxpool/offset': tf.FixedLenFeature([], tf.string),
        'maxpool/arg': tf.FixedLenFeature([], tf.string),
        'maxpool/indices': tf.FixedLenFeature([], tf.string),
        'unpooling/indices': tf.FixedLenFeature([], tf.string),
        'conv/offset': tf.FixedLenFeature([], tf.string),
        'conv/indices': tf.FixedLenFeature([], tf.string),
        'conv/weights': tf.FixedLenFeature([], tf.string),
    }
    return tf.parse_single_example(example_proto, feature_discription)


def decode_record_regression(record, level_num, is_point_feature):
    with tf.variable_scope("DecodeData"):
        maxpooling_matrix_list = []
        maxpooling_arg_list = []
        unpooling_list = []
        conv_matrix_list = []

        label = tf.decode_raw(record['label'], tf.float32)
        mesh_name = record['mesh_name']
        label = tf.reshape(label, (-1, 6))
        shape_list = tf.decode_raw(record['shape'], tf.int32)
        shape_list = tf.reshape(shape_list, (-1, 4))

        e_num = record['e_num']
        e_num = tf.cast(e_num, tf.int32)
        e_pos_list = tf.decode_raw(record['e_pos'], tf.int32)
        e_pos_index, e_neg_index = Get_Ev_index(e_pos_list, e_num)
        e_pos_matrix = Edge_Pos_Matrix(e_pos_index, e_num, shape_list[0][0])
        ev_matrix = Edge_Matrix(e_pos_index, e_neg_index, e_num, shape_list[0][0])

        #points_coord = tf.decode_raw(record['points_coord'], tf.float64)
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
        unpooling_indices_list = tf.decode_raw(record['unpooling/indices'], tf.int32)

        conv_offset = tf.decode_raw(record['conv/offset'], tf.int32)
        conv_indices_list = tf.decode_raw(record['conv/indices'], tf.int32)
        #conv_weights_list = tf.decode_raw(record['conv/weights'], tf.float64)
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
            maxpooling_values = tf.ones(shape=[maxpooling_offset[level_i]], dtype=tf.float32)

            maxpooling_matrix_list.append(MaxPooling_Matrix(shape_list[level_i][0], shape_list[level_i + 1][0], shape_list[level_i][1], maxpooling_indices, maxpooling_values, maxpooling_arg_list[level_i]))

            unpooling_indices = tf.slice(unpooling_indices_list, [2 * pool_begin], [2 * maxpooling_offset[level_i]])
            unpooling_values = tf.ones(shape=[maxpooling_offset[level_i]], dtype=tf.float32)
            unpooling_indices = tf.cast(tf.reshape(unpooling_indices, (maxpooling_offset[level_i], 2)), tf.int64)
            unpooling_list.append(UnPooling_Matrix(shape_list[level_i][0], shape_list[level_i + 1][0], shape_list[level_i][1], unpooling_indices, unpooling_values))

            pool_begin = pool_begin + maxpooling_offset[level_i]
    return mesh_name, label, shape_list, input_feature, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list, unpooling_list, e_pos_matrix, ev_matrix


def Regression_network(record, G_num, level_num, conv_shape, pred_len, reuse ,is_point_feature, feature_channel, drop_out):
    F_len = 64

    mesh_name, label, shape_list, input_tensor, maxpooling_matrix_list, maxpooling_arg_list, conv_matrix_list, unpooling_matrix_list, e_pos_matrix, ev_matrix = decode_record_regression(record, level_num, is_point_feature)
    #input_tensor : (v.a.m.n) * (3:4)
    if(is_point_feature):
        input_tensor = tf.reshape(input_tensor, (-1, feature_channel))
        input_tensor = FeatureDuplicate(input_tensor, G_num)
        grid_input = False
    else:
        grid_input = True


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

    #level0_f = FeatureReduce(level0_f, G_num)
    #F_num x 4 * F_len
    level0_f = tf.contrib.layers.fully_connected(level0_f, pred_len , activation_fn=None, reuse=reuse, scope="FC")
    #F_num * 4 * F_len
    pred = FeatureReduce(level0_f, G_num)

    pred_normal = pred[:,0:3]
    pred_coord = pred[:,3:6]

    gt_normal = label[:,0:3]
    gt_coord = label[:,3:6]
    
    loss1 = tf.reduce_mean(tf.reduce_sum(tf.abs(gt_coord - pred_coord), axis=1), axis=0)
    loss2 = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.sparse_tensor_dense_matmul(ev_matrix, pred_coord)), axis=1), axis=0)
    loss3 = tf.reduce_mean(tf.reduce_sum(tf.abs(gt_normal - pred_normal), axis=1), axis=0)
    loss4 = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.sparse_tensor_dense_matmul(ev_matrix, pred_normal)), axis=1), axis=0)
    loss5 = tf.reduce_mean(tf.abs(tf.reduce_sum(tf.multiply(tf.sparse_tensor_dense_matmul(e_pos_matrix, pred_normal) , tf.sparse_tensor_dense_matmul(ev_matrix, pred_coord)), axis=1)), axis=0)


    w_reg = 0.2
    w_n = 0.1
    w_con = 20

    loss1 = loss1
    loss2 = w_reg * loss2
    loss3 = w_n * loss3 
    loss4 = w_n * w_reg * loss4
    loss5 = w_con * loss5

    loss = loss1 + loss2 + loss3 + loss4 + loss5

    output = {}
    output['name'] = mesh_name
    output['loss'] = loss
    output['loss1'] = loss1
    output['loss2'] = loss2
    output['loss3'] = loss3
    output['loss4'] = loss4
    output['loss5'] = loss5
    output['predicted'] = pred
    return output


class PFCNN_Regression(PFCNN):
    def __init__(self, FLAGS):
        super(PFCNN_Regression, self).__init__(parse_function_regression, Regression_network, FLAGS)
    
    def build_summary(self):
        L2_loss_summary = tf.summary.scalar('L2_Regular', self.lossL2)
        total_loss_summary = tf.summary.scalar('TotalLoss', self.total_loss)
        train_loss_summary = tf.summary.scalar('training_loss', self.train_output['loss'])
        train_loss1_summary = tf.summary.scalar('training_loss1', self.train_output['loss1'])
        train_loss2_summary = tf.summary.scalar('training_loss2', self.train_output['loss2'])
        train_loss3_summary = tf.summary.scalar('training_loss3', self.train_output['loss3'])
        train_loss4_summary = tf.summary.scalar('training_loss4', self.train_output['loss4'])
        train_loss5_summary = tf.summary.scalar('training_loss5', self.train_output['loss5'])
        self.merged_train_summ = tf.summary.merge([total_loss_summary, L2_loss_summary, train_loss_summary, train_loss1_summary, train_loss2_summary, train_loss3_summary, train_loss4_summary, train_loss5_summary])

        test_loss_summary = tf.summary.scalar('testing_loss', self.test_output['loss'])
        test_loss1_summary = tf.summary.scalar('testing_loss1', self.test_output['loss1'])    
        self.merged_test_summ = tf.summary.merge([test_loss_summary, test_loss1_summary])


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

            if not os.path.exists(self.flags.summaries_dir+'/pred'):
                os.mkdir(self.flags.summaries_dir+'/pred')
            best_loss1_sum = 10000

            lr = self.flags.learning_rate

            for epoch_i in tqdm(range(start_epoch, start_epoch + self.flags.epoch_num)):
                if(epoch_i > 0 and epoch_i % self.flags.decay_epoch == 0):
                    lr = lr / 2
                print("\nepoch " + str(epoch_i) + " training:")
                print("learning rate: "+ str(lr))
                for iter_j in range(self.flags.train_size):
                    training_output, merged_sum, _, = sess.run([self.train_output, self.merged_train_summ, self.train_step], feed_dict={self.learning_rate: lr})
                    train_writer.add_summary(merged_sum, iter_j + self.flags.train_size * epoch_i)
                print("\nepoch " + str(epoch_i) + " testing:")
                test_loss1_sum = 0.0
                for iter_j in range(self.flags.test_size):
                    testing_output, test_summary, testing_loss1 = sess.run([self.test_output, self.merged_test_summ, self.test_output['loss1']])
                    test_loss1_sum += testing_loss1
                    test_writer.add_summary(test_summary, iter_j + self.flags.test_size * epoch_i)
                print('test_loss_sum: ' + str(test_loss1_sum))
                if(epoch_i % self.flags.save_epoch == 0):
                    self.saver.save(sess, self.flags.checkpoint_dir+self.flags.task+'.ckpt' , global_step=epoch_i)
                if(test_loss1_sum <= best_loss1_sum):
                    best_loss1_sum = test_loss1_sum
                    self.best_saver.save(sess, self.flags.checkpoint_dir+self.flags.task+'_best.ckpt' , global_step=epoch_i)
                    # for iter_j in range(self.flags.train_size):
                    #     training_output = sess.run(self.train_output)
                    #     np.savetxt(self.flags.summaries_dir+'/pred/' + training_output['name'].decode()+'_pred.txt', training_output['predicted'], fmt='%.6f', delimiter='\n')
                    # for iter_j in range(self.flags.test_size):
                    #     testing_output = sess.run(self.test_output)
                    #     np.savetxt(self.flags.summaries_dir+'/pred/' + testing_output['name'].decode()+'_pred.txt', testing_output['predicted'], fmt='%.6f', delimiter='\n')


if __name__ == "__main__":
    model = PFCNN_Regression(FLAGS)
    model.build_graph()
    model.build_summary()
    model.train()
