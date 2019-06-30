import tensorflow as tf
from config import *


class HAN_model():
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
            self.input_data = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, self.truncature_len),name='input_data')
            self.input_label_rule = tf.placeholder(dtype=tf.int64, shape=(self.batch_size,), name='input_label_rule')
            self.input_label_zm = tf.placeholder(dtype=tf.int64, shape=(self.batch_size,), name='input_label_zm')
            self.input_label_xq = tf.placeholder(dtype=tf.int64, shape=(self.batch_size,), name='input_label_xq')
            self.keep_rate = tf.placeholder(dtype=tf.float32, shape=(None), name='keep')
            self.is_trainning = tf.placeholder(dtype=tf.bool, shape=(None), name='trainning')

        with tf.name_scope('embedding'):
            embedding_table = tf.get_variable(name='embedding_table', shape=(self.vocab_size, self.embedding_size),
                                              dtype=tf.float32)
            self.word_input_data = tf.reshape(self.input_data,shape=(-1,self.len_sq,self.len_word))
            # print(self.word_input_data)
            data_embedding = tf.nn.embedding_lookup(embedding_table, self.word_input_data)
            # print(data_embedding)
            # data_embedding = tf.layers.batch_normalization(data_embedding, training=self.is_trainning)

        with tf.name_scope('encoder'):
            # shape=(3200, 15, 300)
            self.data_word_lavel = tf.reshape(data_embedding, shape=(-1, self.len_word, self.embedding_size))
            # print(data_word_lavel)
            # shape=(3200, 15, 600)
            self.word_encoder = self.Dynamic_LSTM(input=self.data_word_lavel, keep_rate=self.keep_rate,
                                                  training=self.is_trainning, name='encoder1')
            # print(self.word_encoder1)
            # shape=(3200, 600)
            self.word_atten = self.attention(inputs=self.word_encoder, name='word')
            # shape=(128, 25, 600)
            self.data_sq_lavel = tf.reshape(self.word_atten, shape=(-1, self.len_sq, self.hidden_size * 2))
            # shape=(128, 25, 600)
            self.sq_encoder = self.Dynamic_LSTM(input=self.data_sq_lavel, keep_rate=self.keep_rate,
                                                training=self.is_trainning, name='encoder2')
            self.sq_encoder1 = self.Dynamic_LSTM(input=self.sq_encoder, keep_rate=self.keep_rate,
                                                training=self.is_trainning, name='encoder3')
            self.conv = self.conv1D(inputs=self.sq_encoder,kernel_shape=[10,600,300],strides=1,kernel_name='conv_1',padding='SAME')
            print(self.conv)


        with tf.name_scope('attention'):
            self.atten_zm = self.attention(self.sq_encoder1, name='zm')

            self.atten_xq = self.attention(self.sq_encoder1, name='xq')

            self.atten_rule = self.attention(self.sq_encoder1, name='rule')

        with tf.name_scope('classifier'):
            self.rule_dense_1 = self.fully_conacation(name='rule_1',input=self.atten_rule, haddin_size=300,
                                                      training=self.is_trainning, keep_rate=self.keep_rate,activation='tanh')
            self.rule_dense_2 = self.fully_conacation(name='rule_2',input=self.rule_dense_1, haddin_size=rule_number,
                                                      training=self.is_trainning, keep_rate=self.keep_rate,activation='tanh')
            # print(self.rule_dense_2)
            self.zm_dense_1 = self.fully_conacation(name='zm_1',input=self.atten_zm, haddin_size=300,
                                                    training=self.is_trainning, keep_rate=self.keep_rate,activation='tanh')
            self.zm_dense_2 = self.fully_conacation(name='zm_2',input=self.zm_dense_1, haddin_size=zm_number,
                                                    training=self.is_trainning, keep_rate=self.keep_rate,activation='tanh')

            self.xq_dense_1 = self.fully_conacation(name='xq_1',input=self.atten_xq, haddin_size=128,
                                                    training=self.is_trainning, keep_rate=self.keep_rate,activation='tanh')
            self.xq_dense_3 = self.fully_conacation(name='xq_3',input=self.xq_dense_1, haddin_size=xq_number,
                                                    training=self.is_trainning, keep_rate=self.keep_rate,activation='tanh')

        with tf.name_scope('classifier_loss'):
            # self.label_rule = tf.reduce_sum(tf.slice(self.input_label,[0,0],[-1,1],name='rule_slice'),axis=-1)
            # self.label_zm = tf.reduce_sum(tf.slice(self.input_label,[0,1],[-1,1],name='zm_slice'),axis=-1)
            # self.label_xq = tf.reduce_sum(tf.slice(self.input_label,[0,2],[-1,1],name='xq_slice'),axis=-1)

            # self.label_xq_float = tf.cast(self.input_label_xq,dtype=tf.float32)
            self.rule_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_rule,
                                                                                           logits=self.rule_dense_2),axis=-1)
            tf.add_to_collection('loss', self.rule_loss)
            tf.summary.scalar('rule_loss', self.rule_loss)
            # print(self.rule_loss)
            self.zm_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_zm,
                                                                                         logits=self.zm_dense_2),axis=-1)
            tf.add_to_collection('loss', self.zm_loss)
            tf.summary.scalar('zm_loss', self.zm_loss)
            # print(self.zm_loss)
            self.xq_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label_xq,
                                                                                         logits=self.xq_dense_3),axis=-1)
            tf.add_to_collection('loss', self.xq_loss)
            tf.summary.scalar('xq_loss', self.xq_loss)
            # print(self.xq_loss)
            self.all_loss = tf.add_n(tf.get_collection('loss'))
            tf.summary.scalar('all_loss', self.all_loss)

        with tf.name_scope('accuracy'):
            self.rule_max_index = tf.argmax(self.rule_dense_2, axis=1)
            # print(self.rule_max_index)
            self.zm_max_index = tf.argmax(self.zm_dense_2, axis=1)
            # print(self.zm_max_index)
            self.xq_max_index = tf.argmax(self.xq_dense_3, axis=1)
            # print(self.xq_max_index)
            self.rule_acc = tf.reduce_mean(
                tf.cast(tf.equal(self.rule_max_index, self.input_label_rule), dtype=tf.float32), axis=-1)
            tf.summary.scalar('rule_acc', self.rule_acc)
            self.zm_acc = tf.reduce_mean(tf.cast(tf.equal(self.zm_max_index, self.input_label_zm), dtype=tf.float32),
                                         axis=-1)
            tf.summary.scalar('zm_acc', self.zm_acc)
            self.xq_acc = tf.reduce_mean(tf.cast(tf.equal(self.xq_max_index, self.input_label_xq), dtype=tf.float32),
                                         axis=-1)
            tf.summary.scalar('xq_acc', self.xq_acc)
            self.all_acc = [self.rule_acc, self.zm_acc, self.xq_loss]
            # print(self.all_acc)
            update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
            self.opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.all_loss)
            self.train_op = tf.group(self.opt_op, update_ops)

    def Dynamic_LSTM(self, input, keep_rate, training, name,init_hadden_state_fw=None,init_hadden_state_bw=None):
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

            return lstm_output

    def fully_conacation(self,name, input, haddin_size, training=True, keep_rate=1.0, activation='relu'):
        with tf.variable_scope(name):
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

    def assert_regular(self, tensor1, tensor2):
        assert_regular_dot = tf.reduce_sum(tf.multiply(tensor1, tensor2), axis=-1)
        u = tf.sqrt(tf.reduce_sum(tf.square(tensor1), axis=-1))
        v = tf.sqrt(tf.reduce_sum(tf.square(tensor2), axis=-1))
        loss_cos = tf.reduce_mean(assert_regular_dot / tf.multiply(u, v), axis=-1)
        loss_sin = tf.sqrt(1 - tf.square(loss_cos))
        return loss_sin

    def attention(self, inputs, name):
        with tf.variable_scope(name + 'att'):
            transfor_data = self.fully_conacation(name=name,input=inputs, haddin_size=self.hidden_size * 2,
                                                  training=self.is_trainning, keep_rate=self.keep_rate,
                                                  activation='tanh')
            u_atten = tf.get_variable(name=name + 'att_vocter', shape=(1, self.hidden_size * 2), dtype=tf.float32)
            att_weght = tf.reduce_sum(tf.multiply(transfor_data, u_atten), keep_dims=True, axis=2)
            att_weght = tf.nn.softmax(att_weght)
            att_sum = tf.reduce_sum(tf.multiply(inputs, att_weght), axis=1)
        return att_sum

    def conv1D(self, inputs, kernel_shape,strides,kernel_name, padding='VALID',activation='leaky_relu', dropuot_rate=1.0):
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

if __name__=='__main__':
    Model = HAN_model()
    int_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(int_op)
        sess.run(Model.sq_encoder)