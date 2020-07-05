
Is_training=True

class Training_arg:
    G_num = 4
    level_num = 3
    conv_shape = (5, 5)
    task = 'seg'
    learning_rate = 2e-4
    class_num = 8
    epoch_num = 200
    decay_epoch = 20
    train_size = 381
    test_size = 18
    save_epoch = 3
    feature_channel = 4
    is_point_feature = False
    drop_out = 0.5
    summaries_dir = task+'_'+str(level_num)+'level_' + ('_drop_out'+str(drop_out) if drop_out>0 else '')
    checkpoint_dir= summaries_dir + "\\ckpt\\"
    tfrecord_path = "D:/data/sig17_seg_benchmark/TFRecords/"
    train_data = "seg_train_3level_tan.tfrecords"
    test_data = "seg_test_3level_tan.tfrecords"
    is_load_model = False
    model_path = '_'
    start_epoch = 0

class Test_arg:
    G_num = 4
    level_num = 3
    conv_shape = (5, 5)
    task = 'seg'
    learning_rate = 1e-4
    class_num = 8
    train_size = 381
    test_size = 18
    feature_channel = 4
    is_point_feature = False
    drop_out = 0.5
    summaries_dir = task+'_'+str(level_num)+'level_' + ('_drop_out'+str(drop_out) if drop_out>0 else '')
    tfrecord_path = "D:/data/sig17_seg_benchmark/TFRecords/"
    train_data = "seg_train_3level_tan.tfrecords"
    test_data = "seg_test_3level_tan.tfrecords"
    model_path = "seg_3level_all_7conv_drop_out0.5\\ckpt\\seg_best.ckpt-46"

if(Is_training):
    FLAGS = Training_arg()
else:
    FLAGS = Test_arg()
    