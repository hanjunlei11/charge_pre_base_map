import jsonlines as jl
from bosonnlp import BosonNLP
import numpy as np
from config import *
import xlrd

def load_data(filepath):
    with open(filepath+'train.json','r',encoding='utf-8') as data_train,open(filepath+'test.json','r',encoding='utf-8') as data_test,open(filepath+'data_train.txt','w+',encoding='utf-8') as train_data,open(filepath+'data_test.txt','w+',encoding='utf-8') as test_data:
        nlp = BosonNLP('sPB-JflO.34520.7EXOGbw_13LD')
        i = 0
        for item in jl.Reader(data_train):
            # sentence = nlp.tag(item['fact'])[0]['word']
            relevant_articles = item["meta"]["relevant_articles"]
            if len(relevant_articles)>=2:
                continue
            else:
                relevant_articles = relevant_articles[0]
            accusation = item['meta']['accusation']
            if len(accusation)>=2:
                continue
            else:
                accusation = accusation[0]
            sentence = nlp.tag(item['fact'])[0]['word']
            imprisonment = item['meta']['term_of_imprisonment']['imprisonment']
            if imprisonment>180:
                continue
            death_penalty = item['meta']['term_of_imprisonment']['death_penalty']
            life_imprisonment = item['meta']['term_of_imprisonment']['life_imprisonment']
            if (death_penalty is True) or (life_imprisonment is True):
                train_data.write(' '.join(sentence) + '     ' + str(relevant_articles) + '     ' + accusation + '     ' + str(400) + '\n')
            else:
                train_data.write(' '.join(sentence) + '     ' + str(relevant_articles) + '     ' + accusation + '     ' + str(imprisonment) + '\n')
            i+=1
            print(i)
        j = 0
        for item in jl.Reader(data_test):
            # sentence = nlp.tag(item['fact'])[0]['word']
            relevant_articles = item["meta"]["relevant_articles"]
            if len(relevant_articles)>=2:
                continue
            else:
                relevant_articles = relevant_articles[0]
            accusation = item['meta']['accusation']
            if len(accusation)>=2:
                continue
            else:
                accusation = accusation[0]
            sentence = nlp.tag(item['fact'])[0]['word']
            imprisonment = item['meta']['term_of_imprisonment']['imprisonment']
            if imprisonment>180:
                continue
            death_penalty = item['meta']['term_of_imprisonment']['death_penalty']
            life_imprisonment = item['meta']['term_of_imprisonment']['life_imprisonment']
            if (death_penalty is True) or (life_imprisonment is True):
                test_data.write(' '.join(sentence) + '     ' + str(relevant_articles) + '     ' + accusation + '     ' + str(400) + '\n')
            else:
                test_data.write(' '.join(sentence) + '     ' + str(relevant_articles) + '     ' + accusation + '     ' + str(imprisonment) + '\n')
            j+=1
            print(j)

def get_dict(filepath):
    with open(filepath+'data_train.txt','r',encoding='utf-8') as data_train,open(filepath+'data_test.txt','r',encoding='utf-8') as data_test,open(filepath+'word2index.txt','w+',encoding='utf-8') as word2index,open(filepath+'train_index.txt','w+',encoding='utf-8') as train_inderx,open(filepath+'test_index.txt','w+',encoding='utf-8') as test_index,open(filepath+'zm2index.txt','w+',encoding='utf-8') as zm2index:
        data_train_lines = data_train.readlines()
        data_test_lines = data_test.readlines()
        word_index = {}
        zm_index = {}
        rule_index = {}

        for line in data_train_lines:
            data = line.strip().split('     ')
            s = data[0].strip().split()
            zm = data[2]
            if zm not in zm_index:
                zm_index[zm] = 1
            else:
                zm_index[zm] += 1
            rule = data[1]
            if rule not in rule_index:
                rule_index[rule] = 1
            else:
                rule_index[rule] += 1
            xq = int(data[3])
            if xq>180 and xq<301:
                continue
            for word in s:
                if word not in stop_word:
                    if word not in word_index:
                        word_index[word] = 1
                    else:
                        word_index[word] += 1
        ss = 0
        for line in data_test_lines:
            data = line.strip().split('     ')
            s = data[0].strip().split()
            ss += len(s)
            # zm = data[2]
            # if zm not in zm_index:
            #     zm_index[zm] = 1
            # else:
            #     zm_index[zm] += 1
            # rule = data[1]
            # if rule not in rule_index:
            #     rule_index[rule] = 1
            # else:
            #     rule_index[rule] += 1
            for word in s:
                if word not in stop_word:
                    if word not in word_index:
                        word_index[word] = 1
                    else:
                        word_index[word] += 1
        word2dict = {}
        word2dict['unk'] = 0
        zm2dict = {}
        rule2dict = {}
        for item in word_index.items():
            if item[1]>=20:
                word2dict[item[0]] = len(word2dict)
        for item in zm_index.items():
            if item[1]>=100:
                zm2dict[item[0]] = len(zm2dict)
        for item in rule_index.items():
            if item[1]>=100:
                rule2dict[item[0]] = len(rule2dict)
        for item in word2dict.items():
            word2index.write(str(item[0])+' '+str(item[1])+'\n')
        for item in zm2dict.items():
            zm2index.write(str(item[0])+' '+str(item[1])+'\n')

        xq2dict = {}
        for line in data_train_lines:
            data = line.strip().split('     ')
            s = data[0].strip().split()
            zm = data[2]
            if zm not in zm2dict:
                continue
            rule = data[1]
            if rule not in rule2dict:
                continue
            xq = int(data[3])
            if xq>180 and xq<301:
                continue
            xq_c = transfor_xq(xq)
            # if str(xq_c) not in xq2dict:
            #     xq2dict[str(xq_c)] = 1
            # else:
            #     xq2dict[str(xq_c)] += 1
            s_index = ['0' for i in range(truncature_len)]
            for i in range(truncature_len):
                if i < len(s):
                    if s[i] in word2dict:
                        s_index[i] = str(word2dict[s[i]])
                    else:
                        s_index[i] = str(word2dict['unk'])
            train_inderx.write(' '.join(s_index)+'    '+str(rule2dict[rule])+'    '+str(zm2dict[zm])+'    '+str(xq_c)+'\n')

        for line in data_test_lines:
            data = line.strip().split('     ')
            s = data[0].strip().split()
            zm = data[2]
            if zm not in zm2dict:
                continue
            rule = data[1]
            if rule not in rule2dict:
                continue
            xq = int(data[3])
            if xq > 180 and xq<301:
                continue
            xq_c = transfor_xq(xq)
            if str(xq_c) not in xq2dict:
                xq2dict[str(xq_c)] = 1
            else:
                xq2dict[str(xq_c)] += 1
            s_index = ['0' for i in range(truncature_len)]
            for i in range(truncature_len):
                if i < len(s):
                    if s[i] in word2dict:
                        s_index[i] = str(word2dict[s[i]])
                    else:
                        s_index[i] = str(word2dict['unk'])
            test_index.write(' '.join(s_index)+'    '+str(rule2dict[rule])+'    '+str(zm2dict[zm])+'    '+str(xq_c)+'\n')

        print(ss/len(data_test_lines))
        print(len(word_index))
        print(len(word2dict))
        print(len(zm2dict))
        print(len(rule2dict))
        print(xq2dict)

def HAN_data_get(filepath):
    with open(filepath + 'data_train.txt', 'r', encoding='utf-8') as data_train, open(filepath + 'data_test.txt', 'r',encoding='utf-8') as data_test,open(filepath+'HAN/word2index.txt','w+',encoding='utf-8') as word2index,open(filepath+'HAN/train_index.txt','w+',encoding='utf-8') as train_inderx,open(filepath+'HAN/test_index.txt','w+',encoding='utf-8') as test_index,open(filepath+'HAN/zm2index.txt','w+',encoding='utf-8') as zm2index:
        data_train_lines = data_train.readlines()
        data_test_lines = data_test.readlines()
        word_index = {}
        zm_index = {}
        rule_index = {}
        s_line = 0
        s_word = 0
        for line in data_train_lines:
            data = line.strip().split('     ')
            s = data[0].strip().split()
            zm = data[2]
            if zm not in zm_index:
                zm_index[zm] = 1
            else:
                zm_index[zm] += 1
            rule = data[1]
            if rule not in rule_index:
                rule_index[rule] = 1
            else:
                rule_index[rule] += 1
            for word in s:
                if word not in stop_word:
                    if word not in word_index:
                        word_index[word] = 1
                    else:
                        word_index[word] += 1
        for line in data_test_lines:
            data = line.strip().split('     ')
            s = data[0].strip().split()
            zm = data[2]
            if zm not in zm_index:
                zm_index[zm] = 1
            else:
                zm_index[zm] += 1
            rule = data[1]
            if rule not in rule_index:
                rule_index[rule] = 1
            else:
                rule_index[rule] += 1
            for word in s:
                if word not in stop_word:
                    if word not in word_index:
                        word_index[word] = 1
                    else:
                        word_index[word] += 1

        word2dict = {}
        word2dict['unk'] = 0
        zm2dict = {}
        rule2dict = {}
        print(zm_index)
        print(rule_index)
        for item in word_index.items():
            if item[1] >= 2:
                word2dict[item[0]] = len(word2dict)
        for item in zm_index.items():
            if item[1] >= 100:
                zm2dict[item[0]] = len(zm2dict)
        # print(len(zm2dict))
        for item in rule_index.items():
            if item[1] >= 100:
                rule2dict[item[0]] = len(rule2dict)
        for item in word2dict.items():
            word2index.write(str(item[0])+' '+str(item[1])+'\n')
        for item in zm2dict.items():
            zm2index.write(str(item[0])+' '+str(item[1])+'\n')

        for line in data_train_lines:
            data,rule,zm,xq = line.strip().split('     ')
            temp = data.strip().split(' ')
            if zm not in zm2dict:
                continue
            if rule not in rule2dict:
                continue
            data_d = [' '.join('0' for k in range(len_of_word)) for i in range(len_of_sq)]
            for j in range(len_of_sq):
                if j < len(temp):
                    sq = temp[j]
                    temp1 = sq.strip().split()
                    s_index = ['0' for i in range(len_of_word)]
                    for i in range(len_of_word):
                        if i < len(temp1):
                            if temp1[i] in word2dict:
                                s_index[i] = str(word2dict[temp1[i]])
                    data_d[j] = ' '.join(s_index)
            train_inderx.write('   '.join(data_d)+'    '+str(rule2dict[rule])+'    '+str(zm2dict[zm])+'    '+str(transfor_xq(int(xq)))+'\n')

        for line in data_test_lines:
            data,rule,zm,xq = line.strip().split('     ')
            temp = data.replace('。','，').replace(',','，').replace('；','，').split('，')
            if zm not in zm2dict:
                continue
            if rule not in rule2dict:
                continue
            data_d = [' '.join('0' for k in range(len_of_word)) for i in range(len_of_sq)]
            for j in range(len_of_sq):
                if j < len(temp):
                    sq = temp[j]
                    temp1 = sq.strip().split()
                    s_index = ['0' for i in range(len_of_word)]
                    for i in range(len_of_word):
                        if i < len(temp1):
                            if temp1[i] in word2dict:
                                s_index[i] = str(word2dict[temp1[i]])
                    data_d[j] = ' '.join(s_index)
            test_index.write('   '.join(data_d) + '    ' + str(rule2dict[rule]) + '    ' + str(zm2dict[zm]) + '    ' + str(transfor_xq(int(xq))) + '\n')
        # print(s_word/s_line)
        # print(s_line/(len(data_train_lines)+len(data_test_lines)))

def read_HAN(filepath):
    with open(filepath+'train_index.txt','r',encoding='utf-8') as train_index,open(filepath+'test_index.txt','r',encoding='utf-8') as test_index,open(filepath+'map.txt','r',encoding='utf-8') as map_of_k:
        train_lines = train_index.readlines()
        test_lines = test_index.readlines()
        map_lines = map_of_k.readlines()
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        zm_dict = {}
        for line in train_lines:
            data = line.strip().split('    ')
            s_list = data[0].split('   ')
            rule = int(data[1])
            zm = int(data[2])
            if data[2] not in zm_dict:
                zm_dict[data[2]] = 1
            else:
                zm_dict[data[2]] += 1
            xq = int(data[3])
            ss_list = []
            for sq in s_list:
                sq_list = list(map(int,sq.strip().split(' ')))
                ss_list.append(sq_list)
            train_data.append(ss_list)
            train_label.append([rule,zm,xq])

        # print(np.asarray(train_data).shape)
        # print(np.asarray(train_label).shape)

        for line in test_lines:
            data = line.strip().split('    ')
            s_list = data[0].split('   ')
            rule = int(data[1])
            zm = int(data[2])
            if data[2] not in zm_dict:
                zm_dict[data[2]] = 1
            else:
                zm_dict[data[2]] += 1
            xq = int(data[3])
            ss_list = []
            for sq in s_list:
                sq_list = list(map(int, sq.strip().split(' ')))
                ss_list.append(sq_list)
            test_data.append(ss_list)
            test_label.append([rule, zm, xq])

        # map_int = []
        # # print(zm_dict)
        # for line in map_lines:
        #     map_line = list(map(int,line.split(' ')))
        #     map_int.append(map_line)
        # print(np.asarray(test_data).shape)
        # print(np.asarray(test_label).shape)

        return train_data,train_label,test_data,test_label

def get_batch(data,label):
    random_int = np.random.randint(0, len(data) - 1, batch_size)
    batch_data = np.asarray(data)[random_int]
    batch_label = np.asarray(label)[random_int]
    return batch_data,batch_label

def transfor_xq(xq):
    xq_c = 0
    if xq==0:
        xq_c = 0
    elif xq>0 and xq<=6:
        xq_c = 1
    elif xq>6 and xq <= 9:
        xq_c = 2
    elif xq>9 and xq<=12:
        xq_c = 3
    elif xq>12 and xq <= 24:
        xq_c = 4
    elif xq>24 and xq <= 36:
        xq_c = 5
    elif xq>36 and xq<=60:
        xq_c = 6
    elif xq>60 and xq <= 84:
        xq_c = 7
    elif xq>84 and xq<= 120:
        xq_c = 8
    elif xq>120 and xq<=300:
        xq_c = 9
    elif xq>300:
        xq_c = 10
    return xq_c

def get_rule(filepath):
    with open(filepath+'HAN/map.txt','w+',encoding='utf-8') as map:
        path = filepath+'map_of_knowlegde.xlsx'
        workbook = xlrd.open_workbook(path)
        data_sheet = workbook.sheets()[0]
        rowNum = data_sheet.nrows  # sheet行数
        colNum = data_sheet.ncols  # sheet列数
        list = []
        for i in range(1,rowNum):
            rowlist = []
            for j in range(colNum):
                rowlist.append(data_sheet.cell_value(i, j))
            list.append(rowlist)
        # print(list)
        charge_dict = {}
        with open(filepath+'HAN/zm2index.txt','r',encoding='utf-8') as zm_dic:
            zm_lines = zm_dic.readlines()
            for line in zm_lines:
                zm2index = line.split(' ')
                zm = zm2index[0]+'罪'
                index = int(zm2index[1])
                charge_dict[zm] = index
        print(len(charge_dict))
        qj_dict = {}
        xq_dict = {}
        zm_dict = {}
        for line in list:
            charge,qj,xq = line
            if charge in charge_dict:
                if qj not in qj_dict:
                    qj_dict[qj] = len(qj_dict)
                if xq not in xq_dict:
                    xq_dict[xq] = len(xq_dict)
                if charge not in zm_dict:
                    zm_dict[charge] = len(zm_dict)
                map.write(str(zm_dict[charge])+' '+str(qj_dict[qj])+' '+str(xq_dict[xq])+'\n')
        print(zm_dict)
        print(qj_dict)
        print(xq_dict)

def read_file(filepath):
    with open(filepath+'train_index.txt','r',encoding='utf-8') as train_index,open(filepath+'test_index.txt','r',encoding='utf-8') as test_index,open(filepath+'map.txt','r',encoding='utf-8') as map_of_k:
        train_lines = train_index.readlines()
        test_lines = test_index.readlines()
        map_lines = map_of_k.readlines()
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        # zm_dict = {}
        # rule_dict = {}
        for line in train_lines:
            data = line.strip().split('    ')
            s = data[0].split(' ')
            rule = int(data[1])
            zm = int(data[2])
            # if data[2] not in zm_dict:
            #     zm_dict[data[2]] = len(zm_dict)
            # if data[1] not in rule_dict:
            #     rule_dict[data[1]] = len(rule_dict)
            xq = int(data[3])
            ss = list(map(int,s))
            train_data.append(ss)
            train_label.append([rule,zm,xq])
        # print(len(zm_dict),len(rule_dict))
        # print(np.asarray(train_data).shape)
        # print(np.asarray(train_label).shape)

        for line in test_lines:
            data = line.strip().split('    ')
            s = data[0].split(' ')
            rule = int(data[1])
            zm = int(data[2])
            xq = int(data[3])
            ss = list(map(int,s))
            test_data.append(ss)
            test_label.append([rule,zm,xq])
        map_int = []
        for line in map_lines:
            map_line = list(map(int,line.split(' ')))
            map_int.append(map_line)
        # print(np.asarray(test_data).shape)
        # print(np.asarray(test_label).shape)

        return train_data,train_label,test_data,test_label,map_int




# if __name__=="__main__":
#     load_data('./exercise_contest/first_stage/')
#     get_dict('./exercise_contest/')
#     get_rule('./exercise_contest/')
#     HAN_data_get('./exercise_contest/')
#     read_HAN('./exercise_contest/HAN/')
#     train_data, train_label, test_data, test_label, map_int = read_file('./exercise_contest/')
#     batch_data,batch_label = get_batch(train_data,train_label)
#     print(np.shape(train_data))