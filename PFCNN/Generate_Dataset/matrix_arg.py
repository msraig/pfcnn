import numpy as np
import struct

# (Vp*G x F)=(Vp*G x Vc*G)x(Vc*G x F)
def AvePooling_Matrix_arg(child_v_num, parent_v_num, a_num, cvt_nums_list, cover_vts_list, cover_vts_axis_list):
    sparse_indices = [[0,0]] * child_v_num * a_num
    sparse_values = [0] * child_v_num * a_num
    #unpooling_indices = [[0,0]] * child_v_num * a_num
    #unpooling_values = [1] * child_v_num * a_num

    count = 0
    for i in range(len(cvt_nums_list)):
        # parent v : i   len(cvt_nums_list) == parent_v_num
        for j in range(cvt_nums_list[i]):
            #child v : cover_vts_list[count],
            #child a : cover_vts_axis_list[count]
            for k in range(a_num):
                v_index = cover_vts_list[count]
                a_index = cover_vts_axis_list[count]
                sparse_indices[count * a_num + k] = [i * a_num + k, v_index * a_num + ((a_index + k) % a_num)]
                #unpooling = pooling.Transpose
                #unpooling_indices[count * a_num + k] = [v_index * a_num + ((a_index + k) % a_num), i * a_num + k]
                sparse_values[count * a_num + k] = 1.0/cvt_nums_list[i]
            count = count + 1
    #return tf.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=[parent_v_num * a_num, child_v_num * a_num]),\
    #        tf.SparseTensor(indices=unpooling_indices, values=unpooling_values, dense_shape=[child_v_num * a_num, parent_v_num * a_num])
    return sparse_indices, sparse_values

def MaxPooling_Matrix_arg(child_v_num, parent_v_num, a_num, cvt_nums_list, cover_vts_list, cover_vts_axis_list):
    max_cover_num = max(cvt_nums_list)
    sparse_indices = [[0,0]] * child_v_num * a_num
    #sparse_values = [1] * child_v_num * a_num
    #unpooling_indices = [[0,0]] * child_v_num * a_num
    #unpooling_values = [1] * child_v_num * a_num
    count = 0
    for i in range(len(cvt_nums_list)):
        # parent v : i   len(cvt_nums_list) == parent_v_num
        for j in range(cvt_nums_list[i]):
            #child v : cover_vts_list[count],
            #child a : cover_vts_axis_list[count]
            for k in range(a_num):
                v_index = cover_vts_list[count]
                a_index = cover_vts_axis_list[count]
                sparse_indices[count * a_num + k] = [(i * a_num + k) * max_cover_num + j, v_index * a_num + ((a_index + k) % a_num)]
                #unpooling = pooling.Transpose
                #unpooling_indices[count * a_num + k] = [v_index * a_num + ((a_index + k) % a_num), i * a_num + k]
                #sparse_values[count * a_num + k] = 1.0/cvt_nums_list[i]
            count = count + 1
    #return max_cover_num, tf.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=[parent_v_num * a_num * max_cover_num, child_v_num * a_num])
            #tf.SparseTensor(indices=unpooling_indices, values=unpooling_values, dense_shape=[child_v_num * a_num, parent_v_num * a_num])
    return max_cover_num, sparse_indices

# (Vc*G x F)=(Vc*G x Vp*G)x(Vp*G x F)
def Conv_Matrix_arg(shape, indices, axis_indices, weights):
    [v_num, a_num, gm, gn] = shape
    sparse_indices = [[0,0]] * v_num * a_num * gm * gn * 3
    count = 0  # count = ((((i * a_num + j) * gm) + m) * gn + n) * 3 
    for i in range(v_num):
        for j in range(a_num):
            for m in range(gm):
                for n in range(gn):
                    for k in range(3):
                        #print(count * 3 + k)
                        #sparse_indices.append([count, indices[count + k] * a_num + axis_indices[count + k]])
                        sparse_indices[count * 3 + k] = [count, indices[count * 3 + k] * a_num + axis_indices[count * 3 + k]]
                    count = count + 1
    #print(sparse_indices)
    #print(weights)
    #return tf.SparseTensor(indices=sparse_indices, values=weights, dense_shape=[v_num*a_num*gm*gn, v_num*a_num])
    return sparse_indices

def Tangent_Conv_Matrix_arg(shape, indices):
    [v_num, a_num, gm, gn] = shape
    #print(shape)
    #print(len(indices))
    '''sparse_indices = [[0,0]] * v_num * a_num * gm * gn
    count = 0
    for i in range(v_num):
        for j in range(a_num):
            for m in range(gm):
                for n in range(gn):
                    sparse_indices[count] = [count, indices[count]]
                    count = count + 1'''
    y = np.concatenate([np.arange(v_num * a_num * gm * gn), indices], axis=0)
    y = np.reshape(y,(2,-1))
    sparse_indices = np.transpose(y)
    return sparse_indices


def AveUnPooling_Matrix_arg(child_v_num, parent_v_num, a_num, cvt_nums_list, cover_vts_list, cover_vts_axis_list):
    unpooling_indices = [[0,0]] * len(cover_vts_list) * a_num
    count = 0
    for i in range(len(cvt_nums_list)):
        # parent v : i   len(cvt_nums_list) == parent_v_num
        for j in range(cvt_nums_list[i]):
            #child v : cover_vts_list[count],
            #child a : cover_vts_axis_list[count]
            for k in range(a_num):
                v_index = cover_vts_list[count]
                a_index = cover_vts_axis_list[count]
                #unpooling = pooling.Transpose
                unpooling_indices[count * a_num + k] = [v_index * a_num + ((a_index + k) % a_num), i * a_num + k]
            count = count + 1
    return unpooling_indices