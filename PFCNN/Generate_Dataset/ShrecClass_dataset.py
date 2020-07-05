from write_tfrecords import write_class_data_to_tfrecords
from read_data import DataReader
#write_data_to_tfrecords(path, 'test_new', [0], 3)


data_path = 'D:/data/SHREC15_nonrigid/SHREC15NonRigidTestDB/cleaned/output/'
output_path = 'E:/data/TFRecords/'
raw_label_path = "D:/data/SHREC15_nonrigid/test.cla"
label_path = "D:/Yuqi/data/SHREC15_nonrigid/labels.txt"

reader = DataReader()
class_dict = reader.read_raw_labels(raw_label_path)

class_all = list(class_dict.keys())


train_set = []
test_set = []
for class_name in class_all:
    train_set = train_set + class_dict[class_name][4:24] 
    test_set = test_set + class_dict[class_name][0:4]

write_class_data_to_tfrecords(data_path, label_path, output_path, 'train_50class_gridinput', train_set, start_level=0, level_num=3)
print("Training DATA finished")
write_class_data_to_tfrecords(data_path, label_path, output_path, 'test_50class_gridinput', test_set, start_level=0, level_num=3)
print("Test DATA finished")
