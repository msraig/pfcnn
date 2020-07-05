
Is_training=True

class Training_arg:
    G_num = 4
    level_num = 3
    conv_shape = (3, 3)
    task = 'scannet'
    class_num = 21
    learning_rate = 5e-4
    epoch_num = 50
    decay_epoch = 5
    train_size = 119382
    test_size = 10327
    save_epoch = 3
    test_epoch = 2
    feature_channel = 8
    is_point_feature = True
    drop_out = 0
    summaries_dir = task + str(level_num)+'level_3conv_pmesh100_3x3' + ('_drop_out'+str(drop_out) if drop_out>0 else '')
    checkpoint_dir= summaries_dir + "/ckpt/"  
    tfrecord_path = "E:/data/scannet_training/tfrecord/"
    #tfrecord_path = "D:/data/scannet_training/tfrecord/"
    train_data = "scannet_pmesh100_train_N_3level_tan_3x3"
    train_part = 10 
    test_data = "scannet_pmesh5_val_N_3level_tan_3x3"
    test_part = 4
    is_load_model = False
    model_path = "scannet3level_3conv_pmesh1003x3_WT/ckpt/" + task + ".ckpt-6"
    start_epoch = 6

class Test_arg:
    G_num = 4
    level_num = 3
    conv_shape = (3, 3)
    task = 'scannet'
    class_num = 21
    learning_rate = 1e-3
    train_size = 119382
    test_size = 948
    feature_channel = 8
    is_point_feature = True
    drop_out = 1
    model_path = "scannet3level_3conv_pmesh100_3x3_WT/ckpt/scannet.ckpt-18"
    tfrecord_path = "E:/data/scannet_training/tfrecord/"
    train_data = "scannet_pmesh100_train_N_3level_tan_3x3"
    train_part = 10 
    test_data = "scannet_pmesh2_test_N_3level_tan_3x3"
    test_part = 4
    pred_path = "scannet3level_3conv_pmesh100_3x3_WT/pred_part10/"


if(Is_training):
    FLAGS = Training_arg()
else:
    FLAGS = Test_arg()
    