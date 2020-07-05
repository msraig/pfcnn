
Is_training=True

class Training_arg:
    G_num = 4
    level_num = 3
    conv_shape = (5, 5)
    task = 'class'
    learning_rate = 1e-4
    class_num = 50
    epoch_num = 200
    decay_epoch = 40
    train_size = 1000
    test_size = 200
    save_epoch = 10
    feature_channel = 4
    is_point_feature = False
    drop_out = 0.0
    summaries_dir = task+'_'+str(level_num)+'level_' + ('_drop_out'+str(drop_out) if drop_out>0 else '')
    checkpoint_dir= summaries_dir + "//ckpt//"
    tfrecord_path = "D:/data/SHREC15_nonrigid/tfrecords/"
    train_data = "train_50class_localInput.tfrecords"
    test_data = "test_50class_localInput.tfrecords"
    is_load_model = False
    model_path = '_'
    start_epoch = 0

class Test_arg:
    G_num = 4
    level_num = 3
    conv_shape = (5, 5)
    task = 'class'
    learning_rate = 1e-4
    class_num = 8
    train_size = 381
    test_size = 18
    feature_channel = 4
    is_point_feature = False
    drop_out = 0.0
    summaries_dir = task+'_'+str(level_num)+'level_' + ('_drop_out'+str(drop_out) if drop_out>0 else '')
    tfrecord_path = "D:/data/SHREC15_nonrigid/tfrecords/"
    train_data = "train_50class_localInput.tfrecords"
    test_data = "test_50class_localInput.tfrecords"
    model_path = "class_3level//ckpt//seg_best.ckpt-46"

if(Is_training):
    FLAGS = Training_arg()
else:
    FLAGS = Test_arg()
    