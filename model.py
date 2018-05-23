#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os

import numpy as np
import tensorflow as tf
from trnn import TGRUCell

class Seq2Point:
    def __init__(self,
                 table_name="", checkpoint_dir="",
                 epochs=100, batch_size=32, seq_length=10, feature_dim=56, m1=1, m2=1, m3=1,
                 is_debug=True, is_train=True, need_visual=False):

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = 0.0001

        # IO config
        self.table_name = table_name
        self.checkpointDir = checkpoint_dir
        # param config
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.output_length = 1
        self.feature_dim = feature_dim
        self.is_debug = is_debug
        self.is_train = is_train
        self.need_visual = need_visual
        # model_param
        self.rnn_cell_dim = 256
        self.dense_dim = 128
        self.attention_dim = 10

        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

    # Init Graph
    def build_input(self):
        self.X, self.Xid, self.X_last, self.X_lastid, self.action, self.Ttime_interval, self.label_value, self.label_oh = self.__data__()

    def build_graph(self):
        self.att_value, self.pred_logit = self.__build_graph__()
        self.loss, self.optimizer = self.__loss_optimizer__()

    def build_summary(self, name='train'):
        pset, mset = self.__evaluation__()
        self.positive_score, self.pred_binary = pset
        self.acc, self.precision, self.recall, self.auc = mset
        self.summary_op = self.__add_summary__(name)

    def __add_summary__(self, name):
        print('summary')
        summary = [
            tf.summary.scalar(name + '/loss', self.loss),
            tf.summary.scalar(name + '/metrics/acc', self.acc),
            tf.summary.scalar(name + '/metrics/precision', self.precision),
            tf.summary.scalar(name + '/metrics/recall', self.recall),
            tf.summary.scalar(name + '/metrics/auc', self.auc),
            # tf.summary.histogram(name+'/attention', self.att_value),
            # tf.summary.histogram(name+'/final_out', self.final_out),
        ]
        summary_op = tf.summary.merge(summary)
        return summary_op

    def __evaluation__(self):
        print('evaluation')
        positive_score = tf.slice(self.pred_logit, [0, 1], [-1, 1])
        pred_binary = tf.cast(positive_score * 2, tf.int32)

        _, acc = tf.metrics.accuracy(self.label_value, pred_binary)
        _, precision = tf.metrics.precision(self.label_value, pred_binary)
        _, recall = tf.metrics.recall(self.label_value, pred_binary)
        _, auc = tf.metrics.auc(self.label_value, positive_score)
        return (positive_score, pred_binary), (acc, precision, recall, auc)

    def __train__(self, loss, optimizer=None):
        if optimizer is None:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(loss, global_step=self.global_step)

    def __loss_optimizer__(self):
        print('make loss')
        loss = tf.losses.softmax_cross_entropy(onehot_labels=self.label_oh, logits=self.pred_logit)
        if self.is_train:
            optimizer = self.__train__(loss)
            return loss, optimizer
        else:
            return loss, None

    def __build_graph__(self):
        print('Build Graph')

        with tf.variable_scope('embedding', initializer=tf.zeros_initializer()):
            X = self.X
            X_last = self.X_last
            print 'self.m1,self.m2,self.m3:',self.m1,self.m2,self.m3
            if self.m1 == "1":
                print 'm1 is activated'
                self.embedding = tf.get_variable('brand_embedding', [100000, self.feature_dim], tf.float32)
                tmp_ids = tf.reshape(self.Xid, [-1, 1]) + 1
                tmp_emb = tf.nn.embedding_lookup([self.embedding for i in range(10)], tmp_ids)
                tmp_emb = tf.reshape(tmp_emb, [self.batch_size, self.seq_length, self.feature_dim])
                X += tmp_emb

                last_emb = tf.nn.embedding_lookup([self.embedding for i in range(10)], self.X_lastid + 1)
                last_emb = tf.reshape(last_emb, [self.batch_size, self.feature_dim])
                X_last += last_emb

            if self.m2 == "1":
                print 'm2 is activated'
                self.action_trans = tf.get_variable('action_trans', [100, self.feature_dim*self.feature_dim], tf.float32,
                                                    initializer=tf.constant_initializer(
                                                        np.diag(np.ones(self.feature_dim, np.float32))
                                                            .reshape(-1)
                                                            .repeat(100)
                                                            .reshape((100,-1))
                                                    ))

                this_action_trans = tf.nn.embedding_lookup(self.action_trans, self.action)
                X = tf.matmul(
                    tf.reshape(X, (self.batch_size, self.seq_length, 1, self.feature_dim)),
                    tf.reshape(this_action_trans, (self.batch_size, self.seq_length, self.feature_dim, self.feature_dim))
                )
  
            X = tf.reshape(X, (self.batch_size, self.seq_length, self.feature_dim))
            X = tf.concat([X,self.Ttime_interval],2)
            print 'X.shape:', X.shape

            # X = tf.reshape(X, [self.batch_size, self.seq_length, self.feature_dim, 1])
            # X = tf.layers.conv2d(X, 3, [5,5], padding='same', activation=tf.nn.relu, name='conv_1')
            # X = tf.layers.conv2d(X, 9, [5,5], padding='same', activation=tf.nn.relu, name='conv_2')
            # X = tf.layers.max_pooling2d(X, [3,3], [1,3], padding='same', name='pooling_1')
            # X = tf.layers.conv2d(X, 16, [5,5], padding='same', activation=tf.nn.relu, name='conv_3')
            # X = tf.layers.conv2d(X, 8, [3,3], padding='same', activation=tf.nn.relu, name='conv_4')
            # X = tf.layers.max_pooling2d(X, [3,3], [1,3], padding='same', name='pooling_2')
            # X = tf.reshape(X, [self.batch_size, self.seq_length, -1])
            X = tf.unstack(X, axis=1)


            # X_last = tf.reshape(X_last, [self.batch_size, 1, self.feature_dim, 1])
            # X_last = tf.layers.conv2d(X_last, 3, [5,5], padding='same', activation=tf.nn.relu, name='conv_1', reuse=True)
            # X_last = tf.layers.conv2d(X_last, 9, [5,5], padding='same', activation=tf.nn.relu, name='conv_2', reuse=True)
            # X_last = tf.layers.max_pooling2d(X_last, [3,3], [1,3], padding='same', name='pooling_1')
            # X_last = tf.layers.conv2d(X_last, 16, [5,5], padding='same', activation=tf.nn.relu, name='conv_3', reuse=True)
            # X_last = tf.layers.conv2d(X_last, 8, [3,3], padding='same', activation=tf.nn.relu, name='conv_4', reuse=True)
            # X_last = tf.layers.max_pooling2d(X_last, [3,3], [1,3], padding='same', name='pooling_2')
            # X_last = tf.reshape(X_last, [self.batch_size, -1])

        with tf.device('/gpu'):
            #####       Encoder Level       #####
            if self.m3 == "1":
                print 'm3 is activated'
                fw_cell = TGRUCell(num_units=self.rnn_cell_dim)
                bw_cell = TGRUCell(num_units=self.rnn_cell_dim)
                bi_output, (state_fw,state_fw), (state_bw,state_bw) = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, X,
                                                                                        dtype=tf.float32)
            else:
                fw_cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_cell_dim)
                bw_cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_cell_dim)
                bi_output, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, X,
                                                                                    dtype=tf.float32)
            if self.is_train:
                fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.9)
                bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.9)
            # 19*cell
            outputs = tf.stack(bi_output)  # L, N, D

            #print type(state_fw)
            #print 'state_fw:',state_fw.shape
            #print 'state_bw:',state_bw
            #print 'X_last:',X_last

            state = tf.concat([state_fw, state_bw, X_last], 1)  # N, 3*D        
            # self.X_last   shape= (N, D)
            # state = tf.concat([state, self.X_last], axis=1)   # N, 3*D
            # outputs = tf.concat()

            #####       Attention Level       #####
            # att_w = tf.Variable(tf.random_normal([self.rnn_cell_dim*2, self.attention_dim]))
            # att_u = tf.Variable(tf.random_normal([self.rnn_cell_dim*2+self.feature_dim, self.attention_dim]))
            # att_b = tf.Variable(tf.ones([self.attention_dim]), dtype=tf.float32)
            # att_v = tf.Variable(tf.random_normal([self.attention_dim, 1]))
            with tf.variable_scope("attention", initializer=tf.random_normal_initializer()):
                att_w = tf.get_variable('att_w', [self.rnn_cell_dim * 2, self.attention_dim], tf.float32)
                att_u = tf.get_variable('att_u', [self.rnn_cell_dim * 2 + self.feature_dim, self.attention_dim],
                                        tf.float32)
                att_b = tf.get_variable('att_b', [self.attention_dim], tf.float32)
                att_v = tf.get_variable('att_v', [self.attention_dim, 1], tf.float32)

                att_ht = tf.tensordot(outputs, att_w, axes=1)  # L, N , 10
                att_h = tf.tensordot(state, att_u, axes=1)  # N, 10
                e = att_ht + att_h + att_b  # L, N, 10
                # e = ttargetpose( tf.reshape(e, shape=[self.seq_length-1, self.batch_size]), perm=[1,0]) # N,L
                e = tf.transpose(e, perm=[1, 0, 2])  # N,L, 10
                e = tf.nn.elu(e)
                e = tf.tensordot(e, att_v, axes=[[2], [0]])
                e = tf.reshape(e, shape=[self.batch_size, self.seq_length])  # N, L
                # 1-norm   to avoid tanh get all -1 or 1
                # which caused by float32 precision lost 
                # emean = tf.reduce_mean(e, axis=1, keep_dims=True)
                # emax = tf.reduce_max(e, axis=1, keep_dims=True)
                # emin = tf.reduce_min(e, axis=1, keep_dims=True)
                # e = (e-emean+0.0001)/(emax-emin+0.0001)
                # e = tf.tanh(e)
                att_value = tf.nn.softmax(e)

                weighted_ht = tf.transpose(outputs, perm=[2, 1, 0]) * att_value
                att_outputs = tf.transpose(tf.reduce_sum(weighted_ht, axis=2), perm=[1, 0])

            self.final_out = tf.concat([att_outputs, X_last], axis=1)

            #####       Dense Classification    #####
            fc1 = tf.layers.dense(self.final_out, self.dense_dim, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, 2, activation=None, name='fc2')
            pred_logit = tf.nn.softmax(fc2)
            return att_value, pred_logit

    def __load_data__(self):
        # pos,neg sample combine     by muming
        if self.is_train:
            table = os.path.join(self.table_name, 'train/*.tf.*')
        else:
            table = os.path.join(self.table_name, 'test/*.tf.*')
        print("reading data from", table)

        def read_file(table):
            selected_cols = 'encodeid_list,brand_vec_list,target_brand_vec,target_encodeid,type_list,time_interval_list,target'
            filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(table),
                                                            num_epochs=self.epochs, shuffle=True)
            reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example,
                                               features={
                                                   'encodeid_list': tf.FixedLenFeature([10], tf.int64),
                                                   'brand_vec_list': tf.FixedLenFeature([10 * 56], tf.float32),
                                                   'target_brand_vec': tf.FixedLenFeature([56], tf.float32),
                                                   'type_list': tf.FixedLenFeature([20], tf.int64),
                                                   'time_interval_list': tf.FixedLenFeature([10], tf.float32),
                                                   'target_encodeid': tf.FixedLenFeature([1], tf.int64),
                                                   'target': tf.FixedLenFeature([1], tf.float32)
                                               })
            Tseqid, Tseq, Tlast, Tlastid, Ttype, Ttime_interval, Tlabel1 = [features[x] for x in selected_cols.split(',')]
            return Tseqid, Tseq, Tlast, Tlastid, Ttype, Ttime_interval, Tlabel1

        var_list = read_file(table)
        capacity = 20000 + 3 * self.batch_size
        tmp = tf.train.shuffle_batch(var_list,
                                     batch_size=self.batch_size,
                                     capacity=capacity,
                                     min_after_dequeue=20000,
                                     num_threads=8)

        Tseqid, Tseq, Tlast, Tlastid, Ttype, Ttime_interval, Tlabel1 = tmp

        Tseqid = tf.reshape(Tseqid, [self.batch_size, self.seq_length])
        X = tf.reshape(Tseq, [self.batch_size, self.seq_length, self.feature_dim])
        X_last = tf.reshape(Tlast, [self.batch_size, self.feature_dim])
        Tlastid = tf.reshape(Tlastid, shape=(self.batch_size, 1))
        Ttype = tf.reshape(Ttype, shape=(self.batch_size, self.seq_length,2))
        Ttype = tf.reshape(Ttype[:,:, 0], shape=(self.batch_size, self.seq_length))
        Ttime_interval = tf.reshape(Ttime_interval, shape = [self.batch_size, self.seq_length, 1])

        label_value = tf.reshape(Tlabel1, shape=[-1])
        label_oh = tf.one_hot(indices=tf.cast(label_value, tf.int32), depth=2, dtype=tf.float32)
        return X, Tseqid, X_last, Tlastid, Ttype, Ttime_interval, label_value, label_oh

    def __data__(self):
        print('load data')
        return self.__load_data__()
