import tensorflow as tf
from config import *

# s = tf.constant([[1,2.6,3],[1,2,3],[1,2,3]],dtype=tf.float32)
# ss = tf.to_int32(s+0.5)
# print(ss)
# s = tf.constant([1.],dtype=tf.float32)
# sigma = tf.Variable([0],dtype=tf.float32,name='sigma')
# ss = tf.get_variable(name='att_v',shape=(batch_size,268,1),dtype=tf.float32)
# att_rate_vec=(1/(2*tf.square(sigma))*s+tf.log(sigma))
# # att_rate_vec = tf.reduce_mean(att_rate_vec,axis=-1)
# print(att_rate_vec)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(s))
#     print(sess.run(ss))
#     print(sess.run(att_rate_vec))
import jsonlines as jl
with open('./exercise_contest/data_train.json','r',encoding='utf-8') as file:
    i = 0
    for item in jl.Reader(file):
        i+=1
        print(i)