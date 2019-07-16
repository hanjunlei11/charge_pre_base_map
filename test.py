from data_process import *
from config import *
import tensorflow as tf
train_data,train_label,test_data,test_label,map_int = read_file('./exercise_contest/')
print('1、load data完成')
ckpt = tf.train.get_checkpoint_state('./ckpt_train/')
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
print('2、构造模型完成')
pre_rule,pre_zm,pre_xq = [],[],[]
with tf.Session() as sess:
    saver.restore(sess, ckpt.model_checkpoint_path)
    graph = tf.get_default_graph()
    print('3、初始化完成')
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    # print(tensor_name_list)
    input_data = graph.get_tensor_by_name("input/input_data:0")
    input_label_rule = graph.get_tensor_by_name('input/input_label_rule:0')
    input_label_zm = graph.get_tensor_by_name('input/input_label_zm:0')
    input_label_xq = graph.get_tensor_by_name('input/input_label_xq:0')
    input_map = graph.get_tensor_by_name('input/input_map:0')
    keep_rate = graph.get_tensor_by_name('input/keep:0')
    is_trainning = graph.get_tensor_by_name('input/trainning:0')
    rule_max_index = graph.get_tensor_by_name('accuracy/ArgMax:0')
    zm_max_index = graph.get_tensor_by_name('accuracy/ArgMax_1:0')
    xq_max_index = graph.get_tensor_by_name('accuracy/ArgMax_2:0')
    # tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    # print(tensor_name_list)
    print('4、开始测试')
    test_data_epoch,test_label_epoch,len_of_test = get_epoch(test_data,test_label)
    for i in range(len_of_test):
        batch_data_test, batch_label_test = get_batch_test(test_data_epoch, test_label_epoch,i)
        batch_label_rule_test = batch_label_test[:, 0]
        batch_label_zm_test = batch_label_test[:, 1]
        batch_label_xq_test = batch_label_test[:, 2]
        feed_dic = {input_data: batch_data_test, input_label_rule: batch_label_rule_test,input_map:map_int,
                    input_label_zm: batch_label_zm_test, input_label_xq: batch_label_xq_test,
                    keep_rate: 1.0, is_trainning: False}
        rule, zm, xq = sess.run([rule_max_index, zm_max_index, xq_max_index],feed_dict=feed_dic)
        print(i)
        pre_rule = pre_rule+list(rule)
        pre_zm = pre_zm + list(zm)
        pre_xq = pre_xq + list(xq)
    with open('./pre.txt','w+',encoding='utf_8') as pre:
        for i in range(len(pre_rule)):
            pre.write(str(pre_rule[i])+' '+str(pre_zm[i])+' '+str(pre_xq[i])+'\n')



