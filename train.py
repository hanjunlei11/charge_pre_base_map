from model import *
from data_process import *
from config import *
import tensorflow as tf
train_data,train_label,test_data,test_label,map_int = read_file('./exercise_contest/')
print('1、load data完成')
Model = model()
print('2、构造模型完成')
# # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
# opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(Model.all_loss)
saver = tf.train.Saver(max_to_keep=10)
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer_train = tf.summary.FileWriter("logs/train", sess.graph)
    writer_test = tf.summary.FileWriter("logs/test")
    sess.run(init_op)
    # ckpt = tf.train.get_checkpoint_state('./ckpt/')
    # saver.restore(sess, save_path=ckpt.model_checkpoint_path)
    print('3、初始化完成')
    print('4、开始训练')
    min_loss = 100
    for i in range(100000):
        batch_data_train, batch_label_train = get_batch(train_data,train_label)
        # print(batch_data)
        batch_label_rule_train = batch_label_train[:,0]
        batch_label_zm_train = batch_label_train[:,1]
        batch_label_xq_train = batch_label_train[:,2]
        # print(batch_label_xq)
        # print(batch_label_zm)
        feed_dic = {Model.input_data: batch_data_train,Model.input_label_rule:batch_label_rule_train,Model.input_map:map_int,
                    Model.input_label_zm: batch_label_zm_train, Model.input_label_xq: batch_label_xq_train, Model.keep_rate: 0.7,
                    Model.is_trainning: True}
        _,loss,rule_acc,zm_acc,xq_acc,rs = sess.run([Model.train_op,Model.all_loss,Model.rule_acc,Model.zm_acc,Model.xq_acc,merged],feed_dict=feed_dic)
        writer_train.add_summary(rs, i + 1)
        print(str(i+1)+':次训练 '+'loss: '+str('%.7f'%loss)+' acc: '+str('%.7f'%rule_acc)+' '+str('%.7f'%zm_acc)+' '+str('%.7f'%xq_acc))

        batch_data_test, batch_label_test = get_batch(test_data, test_label)
        batch_label_rule_test = batch_label_test[:, 0]
        batch_label_zm_test = batch_label_test[:, 1]
        batch_label_xq_test = batch_label_test[:, 2]
        feed_dic = {Model.input_data: batch_data_test, Model.input_label_rule: batch_label_rule_test,Model.input_map:map_int,
                    Model.input_label_zm: batch_label_zm_test, Model.input_label_xq: batch_label_xq_test,
                    Model.keep_rate: 1.0, Model.is_trainning: False}
        loss, rule_acc, zm_acc, xq_acc,rs_t = sess.run([Model.all_loss, Model.rule_acc, Model.zm_acc, Model.xq_acc,merged],feed_dict=feed_dic)
        writer_test.add_summary(rs_t, i + 1)
        print(str(i+1)+':次测试 '+'loss: ' + str('%.7f' % loss) + ' acc: ' + str('%.7f' % rule_acc) + ' ' + str('%.7f' % zm_acc) + ' ' + str('%.7f' % xq_acc))
        if (i+1)%100==0:
            saver.save(sess, save_path='ckpt_train/model.ckpt', global_step=i + 1)
        # with open('./data/train_file.txt', 'a+', encoding='utf-8') as train_file:
        #     train_file.write(str(i+1)+':次训练 '+'loss: '+str('%.7f'%loss)+'acc: '+str('%.7f'%acc)+'max:'+str('%.7f'%max_acc) + '\n')
        # if (i+1)%100==0:
        #     all_loss = 0
        #     all_zm_acc = 0
        #     all_xq_acc = 0
        #     all_rule_acc = 0
        #     for j in range(20):
        #         batch_data, batch_label = get_batch(test_data, test_label)
        #         batch_label_rule = batch_label[:, 0]
        #         batch_label_zm = batch_label[:, 1]
        #         batch_label_xq = batch_label[:, 2]
        #         feed_dic = {Model.input_data: batch_data,Model.input_label_rule:batch_label_rule,
        #                     Model.input_label_zm: batch_label_zm, Model.input_label_xq: batch_label_xq,
        #                     Model.keep_rate: 1.0, Model.is_trainning: False}
        #         loss,rule_acc, zm_acc,xq_acc= sess.run([Model.all_loss,Model.rule_acc, Model.zm_acc,Model.xq_acc], feed_dict=feed_dic)
        #         print('loss: ' + str('%.7f' % loss) + ' acc: '+str('%.7f'%rule_acc)+' '+str('%.7f'%zm_acc)+' '+str('%.7f'%xq_acc))
        #         all_loss+=loss
        #         all_zm_acc+=zm_acc
        #         all_xq_acc+=xq_acc
        #         all_rule_acc+=rule_acc
        #         # for s in range(3):
        #         #     all_acc[s]+=acc[s]
        #     all_loss = all_loss/20
        #     all_zm_acc = all_zm_acc/20
        #     all_xq_acc = all_xq_acc/20
        #     all_rule_acc = all_rule_acc/20
        #     # for s in range(3):
        #     #     all_acc[s] = all_acc[s]/20
        #     if all_loss<min_loss:
        #         min_loss = all_loss
        #         saver.save(sess, save_path='ckpt_retrain/model.ckpt', global_step=i + 1)
        #
        #     print(str((i+1)//100)+':次测试 '+'loss: '+str('%.7f'%all_loss)+'acc: '+str('%.7f'%all_rule_acc)+' '+str('%.7f'%all_zm_acc)+' '+str('%.7f'%all_xq_acc))