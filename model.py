import tensorflow as tf
from config import *

class model():
    def __init__(self):
        self.batch_size = batch_size
        self.embedding_size = embadding_size
        self.vocab_size = vocab_size
        self.truncature_len = truncature_len
        self.hidden_size = hidden_size
        self.map_lines = map_lines
        self.len_sq = len_of_sq
        self.len_word = len_of_word

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.int32,shape=(self.batch_size,self.truncature_len),name='input_data')
            self.input_label_rule = tf.placeholder(dtype=tf.int64,shape=(self.batch_size,),name='input_label_rule')
            self.input_label_zm = tf.placeholder(dtype=tf.int64, shape=(self.batch_size,), name='input_label_zm')
            self.input_label_xq = tf.placeholder(dtype=tf.int64, shape=(self.batch_size,), name='input_label_xq')
            self.input_map = tf.placeholder(dtype=tf.int32,shape=(self.map_lines,3),name='input_map')
            self.keep_rate = tf.placeholder(dtype=tf.float32,shape=(None),name='keep')
            self.is_trainning = tf.placeholder(dtype=tf.bool,shape=(None),name='trainning')

        with tf.name_scope('embedding'):
            embedding_table = tf.get_variable(name='embedding_table',shape=(self.vocab_size,self.embedding_size),dtype=tf.float32)
            data_embedding = tf.nn.embedding_lookup(embedding_table,self.input_data)
            data_embedding = tf.layers.batch_normalization(data_embedding,training=self.is_trainning)
            zm_embedding_table = tf.get_variable(name='zm_embedding_table',shape=(122,self.embedding_size),dtype=tf.float32)
            qj_embedding_table = tf.get_variable(name='qj_embedding_table', shape=(20, self.embedding_size),dtype=tf.float32)
            xq_embedding_table = tf.get_variable(name='xq_embedding_table', shape=(25, self.embedding_size),dtype=tf.float32)
            self.zm_index = tf.reduce_mean(tf.slice(self.input_map,[0,0],[-1,1]),axis=-1)
            self.qj_index = tf.reduce_mean(tf.slice(self.input_map,[0,1],[-1,1]),axis=-1)
            self.xq_index = tf.reduce_mean(tf.slice(self.input_map,[0,2],[-1,1]),axis=-1)
            self.embedding_zm = tf.nn.embedding_lookup(zm_embedding_table,self.zm_index)
            self.embedding_qj = tf.nn.embedding_lookup(qj_embedding_table, self.qj_index)
            self.embedding_xq = tf.nn.embedding_lookup(xq_embedding_table, self.xq_index)

        with tf.name_scope('encoder'):
            self.sq_conv1 = self.conv1D(inputs=data_embedding,
                                        kernel_shape=(5, self.embedding_size, self.embedding_size),
                                        strides=1,
                                        kernel_name='conv1',
                                        activation='relu',
                                        dropuot_rate=self.keep_rate)
            self.sq_conv2 = self.conv1D(inputs=self.sq_conv1,
                                        kernel_shape=(5, self.embedding_size, self.embedding_size),
                                        strides=1,
                                        kernel_name='conv2',
                                        activation='relu',
                                        dropuot_rate=self.keep_rate)

        with tf.name_scope('attention'):
            self.atten_zm = self.attention(self.sq_conv2,name='zm')

            self.atten_qj = self.attention(self.sq_conv2,name='xq')

            # self.atten_rule = self.attention(self.sq_conv2,name='rule')

        with tf.name_scope('regular'):
            # self.regular_loss_1 = self.assert_regular(self.rule_atten, self.qj_atten)
            # tf.add_to_collection('loss',self.regular_loss_1)
            # tf.summary.scalar('rule_regular_loss',self.regular_loss_1)
            #
            # self.regular_loss_2 = self.assert_regular(self.zm_atten, self.qj_atten)
            # tf.add_to_collection('loss', self.regular_loss_2)
            # tf.summary.scalar('zm_regular_loss', self.regular_loss_2)
            #
            # self.regular_loss_3 = self.assert_regular(self.xq_atten, self.qj_atten)
            # tf.add_to_collection('loss', self.regular_loss_3)
            # tf.summary.scalar('xq_regular_loss', self.regular_loss_3)

            self.zm_similar = tf.matmul(self.atten_zm,self.embedding_zm,transpose_b=True)
            self.qj_similar = tf.matmul(self.atten_qj,self.embedding_qj,transpose_b=True)
            self.xq_similar = tf.nn.softmax(tf.add(self.zm_similar,self.qj_similar))
            self.xq_map_out = tf.multiply(tf.expand_dims(self.xq_similar,axis=-1),self.embedding_xq)
            self.xq_map_out = tf.reduce_sum(self.xq_map_out,axis=1)
            self.xq_map_out = tf.layers.batch_normalization(self.xq_map_out,training=self.is_trainning)
            # self.zm_map_out = tf.expand_dims(tf.nn.softmax(self.zm_similar),axis=-1)
            # self.zm_map_out = tf.reduce_sum(tf.multiply(self.zm_map_out,self.embedding_zm),axis=1)
            # self.zm_map_out = tf.layers.batch_normalization(self.zm_map_out,training=self.is_trainning)

        with tf.name_scope('classifier'):
            self.sq_rule, self.rule_fw,self.rule_bw = self.Dynamic_init_LSTM(input=self.sq_conv2,
                                                                             keep_rate=self.keep_rate,
                                                                             training=self.is_trainning,
                                                                             name='rule_lstm')
            self.attention_rule = self.attention(inputs=self.sq_rule,name='rule_atten')
            self.rule_dense_1 = self.fully_conacation(self.attention_rule, haddin_size=300,
                                                      training=self.is_trainning,keep_rate=self.keep_rate)
            self.rule_dense_2 = self.fully_conacation(self.rule_dense_1, haddin_size=rule_number,
                                                      training=self.is_trainning, keep_rate=self.keep_rate)

            self.sq_zm, self.zm_fw, self.zm_bw = self.Dynamic_init_LSTM(input=self.sq_conv2,
                                                                        keep_rate=self.keep_rate,
                                                                        training=self.is_trainning,
                                                                        name='zm_lstm',
                                                                        init_hadden_state_fw=self.rule_fw,
                                                                        init_hadden_state_bw=self.rule_bw)
            self.attention_zm = self.attention(inputs=self.sq_zm, name='zm_atten')
            self.zm_dense_1 = self.fully_conacation(self.attention_zm, haddin_size=300,
                                                      training=self.is_trainning, keep_rate=self.keep_rate)
            self.zm_dense_2 = self.fully_conacation(self.zm_dense_1, haddin_size=zm_number,
                                                      training=self.is_trainning, keep_rate=self.keep_rate)

            self.sq_xq, self.xq_fw,self.xq_bw = self.Dynamic_init_LSTM(input=self.sq_conv2,
                                                                        keep_rate=self.keep_rate,
                                                                        training=self.is_trainning,
                                                                        name='xq_lstm',
                                                                        init_hadden_state_fw=self.zm_fw,
                                                                        init_hadden_state_bw=self.zm_bw)
            self.attention_xq = self.attention(inputs=self.sq_xq, name='xq_atten')
            self.all_xq = tf.concat([self.attention_xq,self.xq_map_out],axis=-1)
            self.xq_dense_1 = self.fully_conacation(self.attention_xq, haddin_size=128,
                                                    training=self.is_trainning, keep_rate=self.keep_rate)
            self.xq_dense_2 = self.fully_conacation(self.xq_dense_1, haddin_size=xq_number,
                                                    training=self.is_trainning, keep_rate=self.keep_rate)

        with tf.name_scope('classifier_loss'):
            # self.label_rule = tf.reduce_sum(tf.slice(self.input_label,[0,0],[-1,1],name='rule_slice'),axis=-1)
            # self.label_zm = tf.reduce_sum(tf.slice(self.input_label,[0,1],[-1,1],name='zm_slice'),axis=-1)
            # self.label_xq = tf.reduce_sum(tf.slice(self.input_label,[0,2],[-1,1],name='xq_slice'),axis=-1)

            # self.label_xq_float = tf.cast(self.input_label_xq,dtype=tf.float32)
            self.rule_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_rule,
                                                                                           logits=self.rule_dense_2),axis=-1)
            tf.add_to_collection('loss',self.rule_loss)
            tf.summary.scalar('rule_loss',self.rule_loss)
            # print(self.rule_loss)
            self.zm_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_zm,
                                                                                           logits=self.zm_dense_2),axis=-1)
            tf.add_to_collection('loss',self.zm_loss)
            tf.summary.scalar('zm_loss',self.zm_loss)
            # print(self.zm_loss)
            self.xq_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_xq,
                                                                                           logits=self.xq_dense_2),axis=-1)
            tf.add_to_collection('loss',self.xq_loss)
            tf.summary.scalar('xq_loss',self.xq_loss)
            # print(self.xq_loss)
            self.all_loss = tf.add_n(tf.get_collection('loss'))
            tf.summary.scalar('all_loss',self.all_loss)

        with tf.name_scope('accuracy'):
            self.rule_max_index = tf.argmax(self.rule_dense_2,axis=1)
            # print(self.rule_max_index)
            self.zm_max_index = tf.argmax(self.zm_dense_2,axis=1)
            # print(self.zm_max_index)
            self.xq_max_index = tf.argmax(self.xq_dense_2,axis=1)
            # print(self.xq_max_index)
            self.rule_acc = tf.reduce_mean(tf.cast(tf.equal(self.rule_max_index, self.input_label_rule), dtype=tf.float32),axis=-1)
            tf.summary.scalar('rule_acc',self.rule_acc)
            self.zm_acc = tf.reduce_mean(tf.cast(tf.equal(self.zm_max_index, self.input_label_zm), dtype=tf.float32),axis=-1)
            tf.summary.scalar('zm_acc',self.zm_acc)
            self.xq_acc = tf.reduce_mean(tf.cast(tf.equal(self.xq_max_index, self.input_label_xq), dtype=tf.float32),axis=-1)
            tf.summary.scalar('xq_acc',self.xq_acc)
            # self.all_acc = [self.rule_acc,self.zm_acc,self.xq_loss]
            # print(self.all_acc)
            update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.all_loss)
            self.train_op = tf.group(self.opt_op,update_ops)

    def Dynamic_LSTM(self,input, keep_rate, training, name):
        with tf.variable_scope("lst_" + str(name) + "_1"):
            cell_f_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell_f_1 = tf.nn.rnn_cell.DropoutWrapper(cell_f_1,output_keep_prob=keep_rate,)
            cell_b_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell_b_1 = tf.nn.rnn_cell.DropoutWrapper(cell_b_1,output_keep_prob=keep_rate)
            lstm_output_s1, _ = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1, inputs=input,dtype=tf.float32)

            # lstm_fw_s1, lstm_bw_s1 = lstm_output_s1
            lstm_output = tf.concat(lstm_output_s1, axis=-1)

            # lstm_output_s1 = tf.layers.batch_normalization(inputs=lstm_output_s1, training=training)
            # concat_s1 = tf.concat([input, lstm_output_s1], axis=-1)
            # auto_encoder_1 = self.fully_conacation(concat_s1,hidden_size*3,training=training,keep_rate=keep_rate)
            # auto_encoder_2 = self.fully_conacation(concat_s1, hidden_size,training=training, keep_rate=keep_rate)
            # decoder_s1_1 = self.fully_conacation(auto_encoder_1, haddin_size=concat_s1.shape[-1], keep_rate=keep_rate)
            # loss_encoder_s1 = tf.losses.mean_squared_error(concat_s1, decoder_s1_1)
            # tf.add_to_collection('loss', loss_encoder_s1)
            # tf.summary.scalar('auto_loss' + name, loss_encoder_s1)
            return lstm_output

    def Dynamic_init_LSTM(self, input, keep_rate, training, name,init_hadden_state_fw=None,init_hadden_state_bw=None):
        with tf.variable_scope("lst_" + str(name) + "_1"):
            cell_f_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell_f_1 = tf.nn.rnn_cell.DropoutWrapper(cell_f_1, output_keep_prob=keep_rate)
            cell_b_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
            cell_b_1 = tf.nn.rnn_cell.DropoutWrapper(cell_b_1, output_keep_prob=keep_rate)
            lstm_output_s1, hadden_state = tf.nn.bidirectional_dynamic_rnn(cell_f_1, cell_b_1,
                                                                initial_state_bw=init_hadden_state_bw,
                                                                initial_state_fw=init_hadden_state_fw,
                                                                inputs=input,dtype=tf.float32)
            state_fw,state_bw = hadden_state
            # print(state_fw)
            lstm_output = tf.concat(lstm_output_s1, axis=-1)
            lstm_output = tf.layers.batch_normalization(inputs=lstm_output,training=training)

            return lstm_output,state_fw,state_bw

    def fully_conacation(self,input, haddin_size, training=True, keep_rate=1.0, activation='relu'):
        dense_out = tf.layers.dense(inputs=input, units=haddin_size)
        dense_out = tf.layers.batch_normalization(inputs=dense_out, training=training)
        if activation == 'relu':
            dense_relu = tf.nn.relu(dense_out)
            dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
            return dense_relu
        elif activation == 'leaky_relu':
            dense_relu = tf.nn.leaky_relu(dense_out)
            dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
            return dense_relu
        elif activation == 'sigmoid':
            dense_relu = tf.nn.sigmoid(dense_out)
            dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
            return dense_relu
        elif activation == 'tanh':
            dense_relu = tf.nn.tanh(dense_out)
            dense_relu = tf.nn.dropout(dense_relu, keep_prob=keep_rate)
            return dense_relu
        elif activation == 'None':
            dense_relu = tf.nn.dropout(dense_out, keep_prob=keep_rate)
            return dense_relu

    def assert_regular(self,tensor1,tensor2):
        assert_regular_dot = tf.reduce_sum(tf.multiply(tensor1, tensor2),axis=-1)
        u = tf.sqrt(tf.reduce_sum(tf.square(tensor1),axis=-1))
        v = tf.sqrt(tf.reduce_sum(tf.square(tensor2),axis=-1))
        loss_cos = tf.reduce_mean(assert_regular_dot/tf.multiply(u,v),axis=-1)
        loss_sin = tf.sqrt(1-tf.square(loss_cos))
        return loss_sin

    def attention(self,inputs,name):
        with tf.variable_scope(name+'att'):
            size = inputs.shape[-1].value
            transfor_data = self.fully_conacation(input=inputs,haddin_size=size,training=self.is_trainning,keep_rate=self.keep_rate,activation='tanh')
            u_atten = tf.get_variable(name=name+'att_vocter', shape=(1, size), dtype=tf.float32)
            att_weght = tf.reduce_sum(tf.multiply(transfor_data, u_atten), keep_dims=True, axis=2)
            att_weght = tf.nn.softmax(att_weght,dim=1)
            att_sum = tf.reduce_sum(tf.multiply(inputs, att_weght), axis=1)
        return att_sum

    def conv1D(self, inputs, kernel_shape,strides,kernel_name, padding='VALID',activation='relu', dropuot_rate=1.0):
        with tf.name_scope('conv1d_'+kernel_name):
            kernel = tf.get_variable(dtype=tf.float32, shape=kernel_shape, name=kernel_name)
            conv_output = tf.nn.conv1d(value=inputs, filters=kernel, stride=strides, padding=padding)
            conv_output = tf.layers.batch_normalization(inputs=conv_output, training=self.is_trainning)
            if activation is 'relu':
                conv_output = tf.nn.relu(conv_output)
            elif activation is 'leaky_relu':
                conv_output = tf.nn.leaky_relu(conv_output)
            elif activation is 'sigmoid':
                conv_output = tf.nn.sigmoid(conv_output)
            elif activation is 'tanh':
                conv_output = tf.nn.tanh(conv_output)
            if dropuot_rate is not None:
                conv_output = tf.nn.dropout(conv_output, keep_prob=dropuot_rate)
            return conv_output

# if __name__=="__main__":
#     Model = model()
#     int_op = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(int_op)
#         sess.run(Model.sq_conv2)