from utils import write_reg_data_to_tfrecords, load_name
from read_data import DataReader
import os
import random


data_path = "D:/data/MPI-FAUST_training/normalized/cleaned_dataset/"
label_path = "D:/data/MPI-FAUST_training/normalized/reg_labels/"
output_path = "D:/data/MPI-FAUST_training/normalized/"

train_name = load_name("D:/data/MPI-FAUST_training/normalized/cleaned_dataset/train_name.txt")
test_name = load_name("D:/data/MPI-FAUST_training/normalized/cleaned_dataset/test_name.txt")

print(len(train_name))
print(len(test_name))

write_reg_data_to_tfrecords(data_path, label_path, output_path, 'regression_train', train_name, start_level=0, level_num=3)
write_reg_data_to_tfrecords(data_path, label_path, output_path, 'regression_test', test_name, start_level=0, level_num=3)
