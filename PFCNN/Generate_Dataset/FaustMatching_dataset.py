from write_tfrecords import write_seg_data_to_tfrecords, load_name
from read_data import DataReader
import os
import random


data_path = "D:/data/MPI-FAUST_training/normalized/reg_dataset/"
label_path = "D:/data/MPI-FAUST_training/normalized/matching_labels/"
output_path = "D:/data/MPI-FAUST_training/normalized/"

train_name = load_name("D:/data/MPI-FAUST_training/normalized/reg_dataset/train_name.txt")
test_name = load_name("D:/data/MPI-FAUST_training/normalized/reg_dataset/test_name.txt")

print(len(train_name))
print(len(test_name))

write_seg_data_to_tfrecords(data_path, label_path, output_path, 'matching_train_1level_tan', train_name, start_level=0, level_num=1)
write_seg_data_to_tfrecords(data_path, label_path, output_path, 'matching_test_1level_tan', test_name, start_level=0, level_num=1)
