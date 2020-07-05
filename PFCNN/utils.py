import tensorflow as tf

def Get_Ev_index(e_pos_list, e_count):
    e_row_list = tf.range(0, e_count*2)
    e_row_list = tf.cast(e_row_list, tf.int64)
    e_pos_list = tf.cast(e_pos_list, tf.int64)
    e_pos_index = tf.transpose(tf.stack([e_row_list, e_pos_list], axis=0))
    reshaped = tf.reshape(e_pos_list, (-1, 2))
    e_neg_list = tf.reshape(tf.transpose(tf.stack([reshaped[:,1], reshaped[:,0]], axis=0)), [-1])
    e_neg_index = tf.transpose(tf.stack([e_row_list, e_neg_list], axis=0))

    return e_pos_index, e_neg_index

def MaxPooling_Matrix(child_v_num, parent_v_num, a_num, sparse_indices, sparse_values, max_cover_num):
    return tf.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=[parent_v_num * a_num * max_cover_num, child_v_num * a_num])

def UnPooling_Matrix(child_v_num, parent_v_num, a_num, sparse_indices, sparse_values):
    return tf.SparseTensor(indices=sparse_indices, values=sparse_values, dense_shape=[child_v_num * a_num, parent_v_num * a_num])

# (Vc*G x F)=(Vc*G x Vp*G)x(Vp*G x F)
def Conv_Matrix(v_num, a_num, gm, gn, indices, weights):
    return tf.SparseTensor(indices=indices, values=weights, dense_shape=[v_num*a_num*gm*gn, v_num*a_num])

def Edge_Pos_Matrix(e_pos_index, e_num, v_num):
    pos_value = tf.ones(shape=[2*e_num], dtype=tf.float32)
    e_pos_index = tf.cast(e_pos_index, dtype=tf.int64)
    
    return tf.SparseTensor(e_pos_index, pos_value, dense_shape=[2*e_num, v_num])

def Edge_Matrix(e_pos_index, e_neg_index, e_num, v_num):
    pos_value = tf.ones(shape=[2*e_num], dtype=tf.float32)
    neg_value = -1 * tf.ones(shape=[2*e_num], dtype=tf.float32)
    value_list = tf.concat([pos_value, neg_value], axis=0)
    index_list = tf.concat([e_pos_index, e_neg_index], axis=0)
    index_list = tf.cast(index_list, dtype=tf.int64)
    return tf.SparseTensor(index_list, value_list, dense_shape=[2*e_num, v_num])

def K_Matrix(K_index, K_value, v_num):
    return tf.SparseTensor(K_index, K_value, dense_shape=[v_num, 3*v_num])
