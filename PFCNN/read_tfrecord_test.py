import tensorflow as tf 



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


dataset_path = "D:/data/SHREC15_nonrigid/tfrecords/test_50class_localInput.tfrecords"
dataset = tf.data.TFRecordDataset(dataset_path)
dataset = dataset.map(parse_function_class, num_parallel_calls=4)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
record = iterator.get_next()

sess = tf.Session()

shape_list = tf.decode_raw(record['shape'], tf.int32)
mesh_id = record['mesh_id']

for i in range(10):
    print(sess.run([shape_list, mesh_id]))