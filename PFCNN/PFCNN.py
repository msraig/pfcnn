from tqdm import tqdm
import tensorflow as tf
import os

class PFCNN:
    def __init__(self, parse_function, network, flags):
        self.parse_function = parse_function
        self.network = network
        self.flags = flags

    def parse_dataset(self, dataset_path):
        dataset = tf.data.TFRecordDataset(dataset_path)
        dataset = dataset.map(self.parse_function, num_parallel_calls=4)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    
    def build_dataset(self):
        self.train_data = self.parse_dataset(self.flags.tfrecord_path + self.flags.train_data)
        self.test_data = self.parse_dataset(self.flags.tfrecord_path + self.flags.test_data)

    def build_graph(self):
        self.build_dataset()        
        self.train_output = self.network(self.train_data, self.flags.G_num, self.flags.level_num, self.flags.conv_shape, self.flags.class_num, reuse=False, is_point_feature=self.flags.is_point_feature, feature_channel=self.flags.feature_channel, drop_out=self.flags.drop_out)

        self.test_output = self.network(self.test_data, self.flags.G_num, self.flags.level_num, self.flags.conv_shape, self.flags.class_num, reuse=True, is_point_feature=self.flags.is_point_feature, feature_channel=self.flags.feature_channel, drop_out=0)

        self.train_vars = tf.trainable_variables()
        self.lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.train_vars]) / len(self.train_vars) * 1e-4
        self.total_loss = self.train_output['loss'] + self.lossL2

        self.saver=tf.train.Saver(max_to_keep=5)
        self.best_saver = tf.train.Saver(max_to_keep=3)

        self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

    def build_summary(self):
        L2_loss_summary = tf.summary.scalar('L2_Regular', self.lossL2)
        total_loss_summary = tf.summary.scalar('TotalLoss', self.total_loss)
        train_loss_summary = tf.summary.scalar('training_loss', self.train_output['loss'])
        self.merged_train_summ = tf.summary.merge([L2_loss_summary, total_loss_summary, train_loss_summary])

        self.train_set_accuracy = tf.placeholder(dtype=tf.float32, name="train_set_accuracy")
        self.train_acc_summary = tf.summary.scalar('train_epoch_accuracy', self.train_set_accuracy)

        test_loss_summary = tf.summary.scalar('testing_loss', self.test_output['loss'])

        self.test_set_accuracy = tf.placeholder(dtype=tf.float32, name="test_set_accuracy")
        self.test_acc_summary = tf.summary.scalar('test_epoch_accuracy', self.test_set_accuracy)
        self.merged_test_summ = tf.summary.merge([test_loss_summary])

        
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

            for epoch_i in tqdm(range(start_epoch, start_epoch + self.flags.epoch_num)):
                if(epoch_i > 0 and epoch_i % self.flags.decay_epoch == 0 and lr>=1e-4):
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
                test_acc_sum = 0.0
                for iter_j in range(self.flags.test_size):
                    testing_output, test_summary = sess.run([self.test_output, self.merged_test_summ])
                    test_writer.add_summary(test_summary, iter_j + self.flags.test_size * epoch_i)
                    test_acc_sum = test_acc_sum + testing_output['accuracy'].sum() / len(testing_output['accuracy'])
                print("epoch " + str(epoch_i) + " testing acc:" + str(test_acc_sum / self.flags.test_size) + "\n")
                test_ave_acc = sess.run(self.test_acc_summary, feed_dict={self.test_set_accuracy: test_acc_sum / self.flags.test_size})
                test_writer.add_summary(test_ave_acc, epoch_i)
                if(test_acc_sum / self.flags.test_size >= best_acc):
                    best_acc = test_acc_sum / self.flags.test_size
                    self.best_saver.save(sess, self.flags.summaries_dir+'/ckpt/' + self.flags.task + '_best.ckpt', global_step=epoch_i)
                if(epoch_i % self.flags.save_epoch == 0):
                    self.saver.save(sess, self.flags.summaries_dir+'/ckpt/' + self.flags.task + '.ckpt' , global_step=epoch_i)

    def test(self):
        with tf.Session() as sess:
            self.saver.restore(sess, self.flags.model_path)
            print(self.flags.model_path)
            print("load model success!")
            train_acc_sum = 0.0
            for iter_j in range(self.flags.train_size):
                training_output = sess.run(self.train_output)
                train_acc_sum += training_output['accuracy'].sum() / len(training_output['accuracy'])
            print("training acc:" + str(train_acc_sum / self.flags.train_size) + "\n")
            test_acc_sum = 0.0
            for iter_j in range(self.flags.test_size):
                testing_output = sess.run(self.test_output)
                #np.savetxt(self.flags.summaries_dir+"/"+testing_output['name'].decode()+".txt", testing_output['predicted'], fmt="%d", delimiter="\n")
                #np.savetxt(self.flags.summaries_dir+"/"+testing_output['name'].decode()+"_max.txt", testing_output['logits'], fmt="%.6f", delimiter="\n")
                print(testing_output['name'].decode(), testing_output['accuracy'])
                test_acc_sum = test_acc_sum + testing_output['accuracy'].sum() / len(testing_output['accuracy'])
            print("testing acc:" + str(test_acc_sum / self.flags.test_size) + "\n")
