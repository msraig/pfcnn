import numpy as np
import struct

class DataReader:
    def read_hrch(self, path):
        f = open(path,'r')
        vnum = int(f.readline())
        #print(vnum)
        cover_vts = []
        cvt_nums = []
        #for i in range(vnum):
        for line in f.readlines():
            linestr = line.strip().split(' ')
            cvt_nums.append(len(linestr))
            cover_vts = cover_vts + list(map(int, linestr))
            #print(cover_vts)
            #print(len(cover_vts[len(cover_vts)-1]))    
        f.close()
        return cvt_nums, cover_vts

    def read_hrch2(self, path):
        f = open(path,'r')
        cover_vts = []
        cvt_nums = []
        #for i in range(vnum):
        for line in f.readlines():
            linestr = line.strip().split(' ')
            cvt_nums.append(len(linestr))
            cover_vts = cover_vts + list(map(int, linestr))
            #print(cover_vts)
            #print(len(cover_vts[len(cover_vts)-1]))    
        f.close()
        return cvt_nums, cover_vts


    def read_pool(self, path):
        f = open(path,'r')
        cover_vts_axis = []
        #cvt_nums = []
        #for i in range(vnum):
        for line in f.readlines():
            linestr = line.strip().split(' ')
            #cvt_nums.append(len(linestr))
            cover_vts_axis = cover_vts_axis + list(map(int, linestr))
        f.close()
        #return cvt_nums, cover_vts_axis
        return cover_vts_axis

    def read_tangent_para(self, t_para_file):
        v_num = int(t_para_file.readline().strip())
        a_num = 1
        gm, gn = 3, 3
        indices = []
        for i in range(v_num):
            ind = list(map(int, t_para_file.readline().strip().split(' ')))
            indices = indices + ind
        #print(len(indices))
        t_para_file.readline()
        return [v_num, a_num, gm, gn], indices

    def read_tangent_para_byname(self, name):
        indices = np.fromfile(name, dtype=np.int32)
        indices = indices.reshape((9, -1)).transpose().reshape(-1)
        v_num = int(len(indices) / 9)
        a_num = 1
        gm, gn = 3, 3
        return [v_num, a_num, gm, gn], indices

    def read_para(self, path):
        f = open(path, 'rb')
        v_num = int.from_bytes(f.read(4), byteorder='little')
        a_num = int.from_bytes(f.read(4), byteorder='little')
        gm = int.from_bytes(f.read(4), byteorder='little')
        gn = int.from_bytes(f.read(4), byteorder='little')
        #print(v_num, a_num, gm, gn)
        length = v_num*a_num*gm*gn*3
        #indices = np.array(struct.unpack('<' + 'i'*len, f.read(4*len)))
        #axis_indices = np.array(struct.unpack('<' + 'i'*len, f.read(4*len)))
        #weights = np.array(struct.unpack('<' + 'd'*len, f.read(8*len)))
        #indices = indices.reshape((v_num, a_num, gm, gn, 3))
        #axis_indices = axis_indices.reshape((v_num, a_num, gm, gn, 3))
        #weights = weights.reshape((v_num, a_num, gm, gn, 3))
        indices = f.read(4*length)
        axis_indices = f.read(4*length)
        weights = f.read(8*length)
        f.close()
        return [v_num, a_num, gm, gn], indices, axis_indices, weights

    def read_labels(self, path):
        labels = []
        f = open(path, 'r')
        for line in f.readlines():
            labels.append(int(line))
        f.close()
        return labels

    def read_decoded_para(self, path):
        f = open(path, 'rb')
        v_num = int.from_bytes(f.read(4), byteorder='little')
        a_num = int.from_bytes(f.read(4), byteorder='little')
        gm = int.from_bytes(f.read(4), byteorder='little')
        gn = int.from_bytes(f.read(4), byteorder='little')
        #print(v_num, a_num, gm, gn)
        length = v_num*a_num*gm*gn*3
        indices = np.array(struct.unpack('<' + 'i'*length, f.read(4*length)))
        axis_indices = np.array(struct.unpack('<' + 'i'*length, f.read(4*length)))
        weights = np.array(struct.unpack('<' + 'd'*length, f.read(8*length)))
        #indices = indices.reshape((v_num, a_num, gm, gn, 3))
        #axis_indices = axis_indices.reshape((v_num, a_num, gm, gn, 3))
        #weights = weights.reshape((v_num, a_num, gm, gn, 3))
        #indices = f.read(4*length)
        #axis_indices = f.read(4*length)
        #weights = f.read(8*length)
        f.close()
        return [v_num, a_num, gm, gn], indices, axis_indices, weights


    def read_raw_labels(self, path):
        f=open(path, 'r')
        f.readline()
        class_num, mesh_num = map(int, f.readline().strip().split(' '))
        labels = [0] * mesh_num
        labels_name = []
        class_dict = {}
        for i in range(class_num):
            f.readline()
            class_name, sub_class, mesh_in_class = f.readline().strip().split(' ')
            class_dict[class_name] = []
            labels_name.append(class_name)
            mesh_in_class = int(mesh_in_class)
            for j in range(mesh_in_class):
                mesh_index = int(f.readline())
                class_dict[class_name].append(mesh_index)
        f.close()
        return class_dict
    def read_local_depth(self, path):
        depth = np.fromfile(path)
        return depth

    def read_grid_feature(self, path):
        f = open(path, 'rb')
        v_num = int.from_bytes(f.read(4), byteorder='little')
        a_num = int.from_bytes(f.read(4), byteorder='little')
        gm = int.from_bytes(f.read(4), byteorder='little')
        gn = int.from_bytes(f.read(4), byteorder='little')
        feature_channel = int.from_bytes(f.read(4), byteorder='little')
        length = v_num*a_num*gm*gn*feature_channel
        fbytes = f.read(8*length)
        grid_input = np.frombuffer(fbytes, dtype=np.float64)
        #grid_input = struct.unpack('<' + 'd'*length, fbytes)
        del fbytes
        #grid_input = f.read(8*length)
        
        #indices = indices.reshape((v_num, a_num, gm, gn, 3))
        #axis_indices = axis_indices.reshape((v_num, a_num, gm, gn, 3))
        #weights = weights.reshape((v_num, a_num, gm, gn, 3))
        #indices = f.read(4*length)
        #axis_indices = f.read(4*length)
        #weights = f.read(8*length)
        f.close()
        return feature_channel, grid_input
    
    def read_seg_label(self, path):
        return np.loadtxt(path, dtype=np.int32)

    def read_reg_label(self, path):
        return np.loadtxt(path, dtype=np.float32)

    def read_rgb_label(self, path):
        x = np.loadtxt(path, dtype=np.float32)
        rgb = x[:, 0:3]
        label = x[:, 3]
        label = np.array(label, dtype=np.int32)
        return rgb, label