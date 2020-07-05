Is_training=True

class Training_arg:
    G_num = 4
    level_num = 1
    conv_shape = (5, 5)
    task = 'matching'
    class_num = 6890
    learning_rate = 1e-3
    epoch_num = 400
    decay_epoch = 50
    train_size = 80
    test_size = 20
    save_epoch = 5
    feature_channel = 4
    is_point_feature = False
    drop_out = 0.3
    summaries_dir = task + str(level_num)+'level' + ('_drop_out'+str(drop_out) if drop_out>0 else '')
    checkpoint_dir= summaries_dir + "\\ckpt\\"
    #tfrecord_path = "D:\\data\\matching\\TFRecords\\"
    tfrecord_path = "D:\\data\\MPI-FAUST_training\\TFRecords\\"
    train_data = "matching_train_1level_tan.tfrecords"
    test_data = "matching_test_1level_tan.tfrecords"
    is_load_model = False
    model_path = "matching_1level_9conv_drop_out0.5/ckpt/matching.ckpt-399"
    start_epoch = 400

class Test_arg:
    G_num = 4
    level_num = 1
    conv_shape = (5, 5)
    task = 'matching'
    class_num = 6890
    learning_rate = 1e-3
    train_size = 80
    test_size = 20
    feature_channel = 4
    is_point_feature = False
    drop_out = 1
    model_path = "matching1level_9conv_JointNorm_drop_out0.7"
    model_name = "matching_best.ckpt-801"
    #tfrecord_path = "D:\\data\\matching\\TFRecords\\"
    tfrecord_path = "D:\\data\\MPI-FAUST_training\\TFRecords\\"
    train_data = "matching_train_1level_tan.tfrecords"
    test_data = "matching_test_1level_tan.tfrecords"
    pred_path = model_path + "\\pred\\"

    

if(Is_training):
    FLAGS = Training_arg()
else:
    FLAGS = Test_arg()
    