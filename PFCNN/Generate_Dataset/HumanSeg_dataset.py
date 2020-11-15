from write_tfrecords import write_seg_data_to_tfrecords, load_name
from read_data import DataReader
import os

train_data_path = 'E:/data/sig17_seg/v_based_3level/train/'
test_data_path = 'E:/data/sig17_seg/v_based_3level/test/'

output_path = 'E:/data/TFRecords/'

train_label_path = 'D:/data/sig17_seg_benchmark/segs/train/vbased/'
test_label_path = 'D:/data/sig17_seg_benchmark/segs/test/vbased/'

train_name = []
test_name = []

train_name = load_name("D:/data/sig17_seg/MDGCNN_dataset/train.txt")
test_name = load_name("D:/data/sig17_seg/MDGCNN_dataset/test.txt")

write_seg_data_to_tfrecords(train_data_path, train_label_path, output_path, 'seg_train_3level_tan', train_name, start_level=0, level_num=3)
write_seg_data_to_tfrecords(test_data_path, test_label_path, output_path, 'seg_test_3level_tan', test_name, start_level=0, level_num=3)
