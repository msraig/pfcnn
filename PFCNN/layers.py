import tensorflow as tf
import sys

def SurfaceConv_with_gridInput(grid_feature, conv_shape, conv_matrix, Fout, scope_name, reuse, feature_channel):
    with tf.variable_scope(scope_name, reuse=reuse):
        #grid_feature.shape = VGmn * Fin
        #conv_matrix.shape = VGmn * VG
        m, n = conv_shape
        y = tf.reshape(grid_feature, [-1, m*n*feature_channel])
        weights = tf.get_variable(name='grid_Input_conv_w', shape=(m*n*feature_channel, Fout), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(True))
        #y.shape = VG * mnFin
        y = tf.matmul(y, weights)
        #y.shape = VG * Fout
        bias = tf.get_variable(name='grid_Input_conv_b', shape=(Fout), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(True))
        y = y + bias
        y = tf.layers.batch_normalization(y, training=True)
    return y

def SurfaceConv(x, conv_shape, conv_matrix, Fout, scope_name, reuse):
    with tf.variable_scope(scope_name, reuse=reuse):
        #x.shape = VG * Fin
        #conv_matrix.shape = VGmn * VG
        m, n = conv_shape
        VG, Fin = x.shape
        weights = tf.get_variable(name='conv_w', shape=(m*n*Fin, Fout), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(True))
        y = tf.sparse_tensor_dense_matmul(conv_matrix, x, adjoint_a=False, adjoint_b=False)
        #y.shape = VGmn * Fin
        y = tf.reshape(y, [-1, m*n*Fin])
        #y.shape = VG * mnFin
        y = tf.matmul(y, weights)
        #y.shape = VG * Fout
        bias = tf.get_variable(name='conv_b', shape=(Fout), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(True))
        y = y + bias
        y = tf.layers.batch_normalization(y, training=True)
    return y

def ResBlock(x, conv_shape, conv_matrix, Fout, name, reuse):
    with tf.variable_scope(name, reuse=reuse):
        y = SurfaceConv(x, conv_shape, conv_matrix, Fout, 'res_conv1', reuse)
        y = tf.nn.relu(y, name='relu1')
        y = SurfaceConv(y, conv_shape, conv_matrix, Fout, 'res_conv2', reuse)
        y = y + x
        y = tf.nn.relu(y, name='relu2')
    return y

def Conv_ResBlock(x, conv_shape, name, conv_matrix, F_len, G_num, block_num, reuse, feature_channel=None, grid_input=False, drop_out=0):
    with tf.variable_scope(name, reuse=reuse):
        if grid_input:
            y = SurfaceConv_with_gridInput(x, conv_shape, conv_matrix, F_len, name + 'conv0', reuse, feature_channel)
        else:
            y = SurfaceConv(x, conv_shape, conv_matrix, F_len, name + 'conv0', reuse)
        y = tf.nn.relu(y)
        for i in range(block_num):
            y = ResBlock(y, conv_shape, conv_matrix, F_len, name + 'ResBlock_%d' % i, reuse)
        if drop_out>0:
            y = tf.nn.dropout(y, keep_prob=(1-drop_out))
        return y

def MaxPooling(x, maxpooling_matrix, max_cover_num):
    #x.shape = CG * F  maxpooling_matrix.shape = (PG*max_cover_num) * CG
    y = tf.sparse_tensor_dense_matmul(maxpooling_matrix, x)
    #y.shape = (PG*max_cover_num) * F
    _, F_len = y.shape
    y = tf.reshape(y, (-1, max_cover_num, F_len))
    y = tf.reduce_max(y, axis=1, keepdims=False)
    #y.shape = PG * F
    return y

def AverageUnPooling(x, aveunpooling_matrix):
    y = tf.sparse_tensor_dense_matmul(aveunpooling_matrix, x)
    return y

def FeatureDuplicate(x, G_num):
    #x.shape = V * F
    V_num, F_len = x.shape
    y = tf.stack([x] * G_num, axis=1)
    y = tf.reshape(y, (-1, F_len))
    #y = tf.tile(x, [G_num, 1])
    return y

def FeatureReduce(x, G_num):
    #x.shape = VG * F
    VG, F_len = x.shape
    y = tf.reshape(x, (-1, G_num, F_len))
    y = tf.reduce_max(y, axis=1, keepdims=False)
    #y.shape = V * F
    return y

