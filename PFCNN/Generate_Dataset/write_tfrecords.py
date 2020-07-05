import numpy as np
import random
import math
import tensorflow as tf
import openmesh
import time
from matrix_arg import MaxPooling_Matrix_arg, Conv_Matrix_arg, AveUnPooling_Matrix_arg
from read_data import DataReader
from sklearn.metrics import accuracy_score
from multiprocessing import Pool

def load_name(path):
    name_list = []
    f=open(path,"r")
    while True:
        line = f.readline()
        if not line:
            return name_list
        else:
            name_list.append(line.strip()) 
    f.close()



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def IntList_to_Bytes(int_list):
    #list_bytes = struct.pack('i'*len(int_list), *int_list)
    x = np.array(int_list, dtype=np.int32)
    list_bytes = x.tobytes()
    return list_bytes

def DoubleList_to_Bytes(float_list):
    #list_bytes = struct.pack('d'*len(float_list), *float_list)
    x = np.array(float_list, dtype=np.float64)
    list_bytes = x.tobytes()
    return list_bytes

def Float32List_to_Bytes(float_list):
    #list_bytes = struct.pack('f'*len(float_list), *float_list)
    x = np.array(float_list, dtype=np.float32)
    list_bytes = x.tobytes()
    return list_bytes

def decode_intlist_from_bytes(code):
    list_len = int(len(code)/4)
    #return list(struct.unpack('i'*list_len, code))
    return list(np.frombuffer(code, dtype=np.int32))

def decode_doublelist_from_bytes(code):
    list_len = int(len(code)/8)
    #return list(struct.unpack('d'*list_len, code))
    return list(np.frombuffer(code, dtype=np.float64))

def flat_list(list_l):
    return list(np.array(list_l).reshape(-1))

def Ev_from_mesh(mesh):
    E_pos_index = []
    E_value = []
    e_count = 0
    for e in mesh.ev_indices():
        i, j = e
        E_pos_index.append(i)
        E_pos_index.append(j)
        e_count += 1
    return E_pos_index, e_count, mesh.n_vertices()

def K_from_mesh(mesh):
    K_index = []
    K_value = []
    points = mesh.points()
    for vh in mesh.vertices():
        i = vh.idx()
        acc_diff = np.array([0,0,0], dtype=np.float32)
        for nvh in mesh.vv(vh):
            j = nvh.idx()
            diff = points[i] - points[j]
            acc_diff += diff
            K_index += [i, 3*j]
            K_index += [i, 3*j + 1]
            K_index += [i, 3*j + 2]
            K_value += list(diff)
        K_index += [i, 3*i]
        K_index += [i, 3*i + 1]
        K_index += [i, 3*i + 2]
        K_value += list(acc_diff)
    return K_index, K_value

def write_reg_data_to_tfrecords(dir_path, label_path, output_path, record_name, mesh_list, start_level=0, level_num=3):
    reader = DataReader()
    writer = tf.python_io.TFRecordWriter(output_path + record_name + '.tfrecords')
    count = 0
    one_iter_time = 0
    random.shuffle(mesh_list)
    total_num = len(mesh_list)
    for mesh_name in mesh_list:
        time_start = time.time()
        shape_list = []
        maxpool_indices_list = []
        maxpool_offset_list = []
        maxpool_arg_list = []
        conv_indices_list = []
        conv_weight_list = []
        conv_offset_list = []
        conv_shape_list = []
        unpool_indices_list = []

        for j in range(start_level, start_level + level_num):
            para_name = dir_path + mesh_name + '_' + str(j) + '_pad.para'
            para_shape, indices, axis_indices, weights = reader.read_para(para_name)
            shape_list = shape_list + para_shape
            indices = decode_intlist_from_bytes(indices)
            axis_indices = decode_intlist_from_bytes(axis_indices)
            weights = decode_doublelist_from_bytes(weights)

            if(j == start_level):
                mesh_path = dir_path + mesh_name + '_' + str(j) + '.obj'
                mesh = openmesh.read_trimesh(mesh_path)
                e_pos_list, e_num, v_num = Ev_from_mesh(mesh)
                if(v_num != shape_list[0]):
                    print("[Error]Unequal v_num in Ev")
                    return
                label = reader.read_reg_label(label_path + mesh_name + '.txt')
                label = flat_list(label)
                if(len(label)!= 6 * shape_list[0]):
                    print("[Error]Unequal v_num in label")
                    return
            #print("decode finished")
            conv_indices = Conv_Matrix_arg(para_shape, indices, axis_indices, weights)

            conv_offset_list.append(len(conv_indices))
            conv_indices = flat_list(conv_indices)
            conv_indices_list = conv_indices_list + conv_indices
            conv_weight_list = conv_weight_list + weights

            if(j > start_level):
                hrch_name = dir_path + mesh_name + '_' + str(j) + '.hrch'
                cvt_nums, cover_vts = reader.read_hrch(hrch_name)
                
                hrch_axis_name = dir_path + mesh_name + '_' + str(j) + '.pool'
                cover_vts_axis = reader.read_pool(hrch_axis_name)

                maxpool_arg, maxpool_indices = MaxPooling_Matrix_arg(shape_list[(j - start_level - 1) * 4], shape_list[(j - start_level) * 4], para_shape[1], cvt_nums, cover_vts, cover_vts_axis)
                maxpool_arg_list.append(maxpool_arg)
                #flat
                maxpool_offset_list.append(len(maxpool_indices))
                maxpool_indices = flat_list(maxpool_indices)
                maxpool_indices_list = maxpool_indices_list + maxpool_indices

                unpooling_indices = AveUnPooling_Matrix_arg(shape_list[(j - start_level - 1) * 4], shape_list[(j - start_level) * 4], para_shape[1], cvt_nums, cover_vts, cover_vts_axis)
                unpooling_indices = flat_list(unpooling_indices)
                unpool_indices_list = unpool_indices_list + unpooling_indices

        new_feature = {
                'mesh_name': _bytes_feature(mesh_name.encode()),
                'shape': _bytes_feature(IntList_to_Bytes(shape_list)),
                'label': _bytes_feature(Float32List_to_Bytes(label)),
                'e_num': _int64_feature(e_num),
                'e_pos': _bytes_feature(IntList_to_Bytes(e_pos_list)),
                'maxpool/offset': _bytes_feature(IntList_to_Bytes(maxpool_offset_list)),
                'maxpool/arg': _bytes_feature(IntList_to_Bytes(maxpool_arg_list)),
                'maxpool/indices': _bytes_feature(IntList_to_Bytes(maxpool_indices_list)),
                'unpooling/indices': _bytes_feature(IntList_to_Bytes(unpool_indices_list)),
                'conv/offset': _bytes_feature(IntList_to_Bytes(conv_offset_list)),
                'conv/indices': _bytes_feature(IntList_to_Bytes(conv_indices_list)),
                'conv/weights': _bytes_feature(Float32List_to_Bytes(conv_weight_list))
        }
        #height scaled
        feature_name =  dir_path + mesh_name + '_0_reweighted.input'
        feature_channel, grid_feature = reader.read_grid_feature(feature_name)
        new_feature['input_feature'] = _bytes_feature(Float32List_to_Bytes(grid_feature))
        new_feature['feature_channel'] = _int64_feature(feature_channel)

        example = tf.train.Example(features=tf.train.Features(feature=new_feature))
        writer.write(example.SerializeToString())

        time_end = time.time()
        one_iter_time = ((time_end-time_start) + one_iter_time*count) / (count+1)
        time_left = one_iter_time*(total_num-count)
        print("Finish mesh " + mesh_name)
        print(str(100.0*count/total_num) + "%" + "finished!")
        print("%.2f min left!" % (time_left/60))
        count = count + 1
    writer.close()

def write_seg_data_to_tfrecords(dir_path, label_path, output_path, record_name, mesh_list, start_level=0, level_num=3):
    reader = DataReader()
    writer = tf.python_io.TFRecordWriter(output_path + record_name + '.tfrecords')
    count = 0
    one_iter_time = 0
    random.shuffle(mesh_list)
    total_num = len(mesh_list)
    for mesh_name in mesh_list:
        time_start = time.time()

        shape_list = []
        maxpool_indices_list = []
        maxpool_offset_list = []
        maxpool_arg_list = []
        conv_indices_list = []
        conv_weight_list = []
        conv_offset_list = []
        conv_shape_list = []
        unpool_indices_list = []

        for j in range(start_level, start_level + level_num):
            para_name = dir_path + mesh_name + '_' + str(j) + '_pad.para'
            para_shape, indices, axis_indices, weights = reader.read_para(para_name)
            shape_list = shape_list + para_shape
            indices = decode_intlist_from_bytes(indices)
            axis_indices = decode_intlist_from_bytes(axis_indices)
            weights = decode_doublelist_from_bytes(weights)

            if(j == start_level):
                label = reader.read_seg_label(label_path + mesh_name + '.txt')
                #label = [0] * shape_list[0]
                label = flat_list(label)
                if(len(label)!=shape_list[0]):
                    print("[Error]Unequal v_num")
                    return
            #print("decode finished")
            conv_indices = Conv_Matrix_arg(para_shape, indices, axis_indices, weights)

            conv_offset_list.append(len(conv_indices))
            conv_indices = flat_list(conv_indices)
            conv_indices_list = conv_indices_list + conv_indices
            conv_weight_list = conv_weight_list + weights

            if(j > start_level):
                hrch_name = dir_path + mesh_name + '_' + str(j) + '.hrch'
                cvt_nums, cover_vts = reader.read_hrch(hrch_name)
                
                hrch_axis_name = dir_path + mesh_name + '_' + str(j) + '.pool'
                cover_vts_axis = reader.read_pool(hrch_axis_name)

                maxpool_arg, maxpool_indices = MaxPooling_Matrix_arg(shape_list[(j - start_level - 1) * 4], shape_list[(j - start_level) * 4], para_shape[1], cvt_nums, cover_vts, cover_vts_axis)
                maxpool_arg_list.append(maxpool_arg)
                #flat
                maxpool_offset_list.append(len(maxpool_indices))
                maxpool_indices = flat_list(maxpool_indices)
                maxpool_indices_list = maxpool_indices_list + maxpool_indices

                unpooling_indices = AveUnPooling_Matrix_arg(shape_list[(j - start_level - 1) * 4], shape_list[(j - start_level) * 4], para_shape[1], cvt_nums, cover_vts, cover_vts_axis)
                unpooling_indices = flat_list(unpooling_indices)
                unpool_indices_list = unpool_indices_list + unpooling_indices

        new_feature = {
                'mesh_name': _bytes_feature(mesh_name.encode()),
                'shape': _bytes_feature(IntList_to_Bytes(shape_list)),
                'label': _bytes_feature(IntList_to_Bytes(label)),
                'maxpool/offset': _bytes_feature(IntList_to_Bytes(maxpool_offset_list)),
                'maxpool/arg': _bytes_feature(IntList_to_Bytes(maxpool_arg_list)),
                'maxpool/indices': _bytes_feature(IntList_to_Bytes(maxpool_indices_list)),
                'unpooling/indices': _bytes_feature(IntList_to_Bytes(unpool_indices_list)),
                'conv/offset': _bytes_feature(IntList_to_Bytes(conv_offset_list)),
                'conv/indices': _bytes_feature(IntList_to_Bytes(conv_indices_list)),
                'conv/weights': _bytes_feature(Float32List_to_Bytes(conv_weight_list))
        }
        #height scaled
        feature_name =  dir_path + mesh_name + '_0_reweighted.input'
        feature_channel, grid_feature = reader.read_grid_feature(feature_name)

       
        new_feature['input_feature'] = _bytes_feature(Float32List_to_Bytes(grid_feature))
        new_feature['feature_channel'] = _int64_feature(feature_channel)

        example = tf.train.Example(features=tf.train.Features(feature=new_feature))
        writer.write(example.SerializeToString())

        time_end = time.time()
        one_iter_time = ((time_end-time_start) + one_iter_time*count) / (count+1)
        time_left = one_iter_time*(total_num-count)
        print("Finish mesh " + mesh_name)
        print(str(100.0*count/total_num) + "%" + "finished!")
        print("%.2f min left!" % (time_left/60))
        count = count + 1
    writer.close()

def write_class_data_to_tfrecords(dir_path, label_path, output_path, record_name, mesh_list, start_level=0, level_num=3):
    reader = DataReader()
    labels = reader.read_labels(label_path)
    writer = tf.python_io.TFRecordWriter(output_path + record_name + '.tfrecords')
    count = 0
    one_iter_time = 0
    random.shuffle(mesh_list)
    total_num = len(mesh_list)
    for i in mesh_list:
        time_start = time.time()
        label = labels[i]
        shape_list = []
        maxpool_indices_list = []
        maxpool_offset_list = []
        maxpool_arg_list = []
        conv_indices_list = []
        conv_weight_list = []
        conv_offset_list = []
        for j in range(start_level, start_level + level_num):
            para_name = dir_path + 'T' + str(i) + '_' + str(j) + '_pad.para'
            para_shape, indices, axis_indices, weights = reader.read_para(para_name)
            #print(shape)
            shape_list = shape_list + para_shape
            #print("shape_list:")
            #print(shape_list)
            indices = decode_intlist_from_bytes(indices)
            axis_indices = decode_intlist_from_bytes(axis_indices)
            weights = decode_doublelist_from_bytes(weights)
            #print("decode finished")
            conv_indices = Conv_Matrix_arg(para_shape, indices, axis_indices, weights)
            #print("finish build matrix")
            #flat
            conv_offset_list.append(len(conv_indices))
            conv_indices = flat_list(conv_indices)
            conv_indices_list = conv_indices_list + conv_indices
            conv_weight_list = conv_weight_list + weights
            #print(conv_offset_list)
            #print(len(weights))
            if(j > 0):
                hrch_name = dir_path + 'T' + str(i) + '_' + str(j) + '.hrch'
                cvt_nums, cover_vts = reader.read_hrch(hrch_name)
                
                hrch_axis_name = dir_path + 'T' + str(i) + '_' + str(j) + '.pool'
                cover_vts_axis = reader.read_pool(hrch_axis_name)

                maxpool_arg, maxpool_indices = MaxPooling_Matrix_arg(shape_list[(j - 1) * 4], shape_list[j * 4], para_shape[1], cvt_nums, cover_vts, cover_vts_axis)
                maxpool_arg_list.append(maxpool_arg)
                #flat
                maxpool_offset_list.append(len(maxpool_indices))
                maxpool_indices = flat_list(maxpool_indices)
                maxpool_indices_list = maxpool_indices_list + maxpool_indices


        new_feature = {
                'label': _int64_feature(label),
                'mesh_id': _int64_feature(i),
                'shape': _bytes_feature(IntList_to_Bytes(shape_list)),
                'maxpool/offset': _bytes_feature(IntList_to_Bytes(maxpool_offset_list)),
                'maxpool/arg': _bytes_feature(IntList_to_Bytes(maxpool_arg_list)),
                'maxpool/indices': _bytes_feature(IntList_to_Bytes(maxpool_indices_list)),
                'conv/offset': _bytes_feature(IntList_to_Bytes(conv_offset_list)),
                'conv/indices': _bytes_feature(IntList_to_Bytes(conv_indices_list)),
                'conv/weights': _bytes_feature(Float32List_to_Bytes(conv_weight_list))
        }
        
        feature_name =  dir_path + 'T' + str(i) + '_0_reweighted.input'
        feature_channel, grid_feature = reader.read_grid_feature(feature_name)
        new_feature['input_feature'] = _bytes_feature(Float32List_to_Bytes(grid_feature))
        new_feature['feature_channel'] = _int64_feature(feature_channel)
        example = tf.train.Example(features=tf.train.Features(feature=new_feature))
        writer.write(example.SerializeToString())

        time_end = time.time()
        one_iter_time = ((time_end-time_start) + one_iter_time*count) / (count+1)
        time_left = one_iter_time*(total_num-count)
        print("Finish mesh " + str(i))
        print(str(100.0*count/total_num) + "%" + "finished!")
        print("%.2f min left!" % (time_left/60))
        count = count + 1
    writer.close()    

def write_mesh_scannet_to_tfrecords(dir_path, label_path, output_path, record_name, mesh_list, start_level=0, level_num=3):
    reader = DataReader()
    writer = tf.python_io.TFRecordWriter(output_path + record_name + '.tfrecords')
    count = 0
    one_iter_time = 0
    random.shuffle(mesh_list)
    total_num = len(mesh_list)
    for mesh_name in mesh_list:
        print("Start mesh " + mesh_name)
        time_start = time.time()

        shape_list = []
        maxpool_indices_list = []
        maxpool_offset_list = []
        maxpool_arg_list = []
        conv_indices_list = []
        conv_weight_list = []
        conv_offset_list = []
        conv_shape_list = []
        unpool_indices_list = []

        for j in range(start_level, start_level + level_num):
            para_name = dir_path + mesh_name + '_' + str(j) + '_pad.para'
            para_shape, indices, axis_indices, weights = reader.read_para(para_name)
            shape_list = shape_list + para_shape
            indices = decode_intlist_from_bytes(indices)
            axis_indices = decode_intlist_from_bytes(axis_indices)
            weights = decode_doublelist_from_bytes(weights)

            if(j == start_level):                
                mesh_path = dir_path + mesh_name + '_' + str(j) + '.obj'
                mesh = openmesh.read_trimesh(mesh_path)
                mesh.update_vertex_normals()
                normal_list = mesh.vertex_normals()
                points = mesh.points()
                z_list = points[:,2]
                rgb_list, label = reader.read_rgb_label(label_path + mesh_name + '.rgbl')
                #label = [0] * shape_list[0]
                label = flat_list(label)
                if(len(label)!=shape_list[0]):
                    print("[Error]Unequal v_num")
                    return
            #print("decode finished")
            conv_indices = Conv_Matrix_arg(para_shape, indices, axis_indices, weights)

            conv_offset_list.append(len(conv_indices))
            conv_indices = flat_list(conv_indices)
            conv_indices_list = conv_indices_list + conv_indices
            conv_weight_list = conv_weight_list + weights

            if(j > start_level):
                hrch_name = dir_path + mesh_name + '_' + str(j) + '.hrch'
                cvt_nums, cover_vts = reader.read_hrch(hrch_name)
                
                hrch_axis_name = dir_path + mesh_name + '_' + str(j) + '.pool'
                cover_vts_axis = reader.read_pool(hrch_axis_name)

                maxpool_arg, maxpool_indices = MaxPooling_Matrix_arg(shape_list[(j - start_level - 1) * 4], shape_list[(j - start_level) * 4], para_shape[1], cvt_nums, cover_vts, cover_vts_axis)
                maxpool_arg_list.append(maxpool_arg)
                #flat
                maxpool_offset_list.append(len(maxpool_indices))
                maxpool_indices = flat_list(maxpool_indices)
                maxpool_indices_list = maxpool_indices_list + maxpool_indices

                unpooling_indices = AveUnPooling_Matrix_arg(shape_list[(j - start_level - 1) * 4], shape_list[(j - start_level) * 4], para_shape[1], cvt_nums, cover_vts, cover_vts_axis)
                unpooling_indices = flat_list(unpooling_indices)
                unpool_indices_list = unpool_indices_list + unpooling_indices

        new_feature = {
                'mesh_name': _bytes_feature(mesh_name.encode()),
                'shape': _bytes_feature(IntList_to_Bytes(shape_list)),
                'label': _bytes_feature(IntList_to_Bytes(label)),
                'z': _bytes_feature(Float32List_to_Bytes(z_list)),
                'normal': _bytes_feature(Float32List_to_Bytes(normal_list)),
                'rgb': _bytes_feature(Float32List_to_Bytes(rgb_list)),
                'maxpool/offset': _bytes_feature(IntList_to_Bytes(maxpool_offset_list)),
                'maxpool/arg': _bytes_feature(IntList_to_Bytes(maxpool_arg_list)),
                'maxpool/indices': _bytes_feature(IntList_to_Bytes(maxpool_indices_list)),
                'unpooling/indices': _bytes_feature(IntList_to_Bytes(unpool_indices_list)),
                'conv/offset': _bytes_feature(IntList_to_Bytes(conv_offset_list)),
                'conv/indices': _bytes_feature(IntList_to_Bytes(conv_indices_list)),
                'conv/weights': _bytes_feature(Float32List_to_Bytes(conv_weight_list))
        }
        #height scaled
        input_feature = np.array([], dtype=np.float32)
        for j in range(start_level, start_level + level_num):
            feature_name =  dir_path + mesh_name + '_' + str(j) + '_reweighted.input'
            feature_channel, grid_feature = reader.read_grid_feature(feature_name)
            grid_feature = np.reshape(grid_feature, (-1, 4))
            grid_feature = grid_feature[:, 3]
            input_feature = np.concatenate((input_feature, grid_feature), axis=0)

        new_feature['input_feature'] = _bytes_feature(Float32List_to_Bytes(input_feature))
        new_feature['feature_channel'] = _int64_feature(feature_channel)

        example = tf.train.Example(features=tf.train.Features(feature=new_feature))
        writer.write(example.SerializeToString())

        time_end = time.time()
        one_iter_time = ((time_end-time_start) + one_iter_time*count) / (count+1)
        time_left = one_iter_time*(total_num-count)
        print("Finish mesh " + mesh_name)
        print(str(100.0*count/total_num) + "%" + "finished!")
        print("%.2f min left!" % (time_left/60))
        count = count + 1
    writer.close()

