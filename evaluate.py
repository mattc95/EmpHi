#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pandas import DataFrame as df

def evaluate(data):
    res = []
    valid = ['0', '1', '2']

    for i in range(4, 7):
        for index, r in data[:50].iterrows():
            c = r[1]
            
            print('===========================================')
            print('第', index, '个问题：')
            print('文本: ', c)
            print('第1个回复: ', r[3])
            print('第2个回复: ', r[i])
            print('第1个好，输入1，第2个好，输入2，两个质量相同，输入0: ')
            boo = True

            while boo:
                temp = str(input("比如 1/2/0, 请在此输入 ---------->>>>: "))
                num = temp.strip()
                if num in valid:
                    num = int(num)
                    res.append([i-4, num])
                    boo = False
                else:
                    print('Input invalid, please re-enter your evaluation!')
        
    return res

def get_all_elements(res):

    better_num = [0, 0, 0]
    lose_num = [0, 0, 0]
    tie_num = [0, 0, 0]
    for i, num in res:
        if num == 1 :
            better_num[i] += 1
        elif num == 2:
            lose_num[i] += 1
        else:
            tie_num[i] += 1
    
    total_num = better_num[0] + lose_num[0] + tie_num[0]
    Name = ['MIME', 'MoEL', 'Multitask-Trans']
    for i in range(3):
        print('Better than %s, %f' % (Name[i], better_num[i]/total_num))
        print('Worse than %s, %f' % (Name[i], lose_num[i]/total_num))
        print('Same with %s, %f' % (Name[i], tie_num[i]/total_num))

def information():
    print('共情对话生成的任务目标是生成一个相关且具有共情的句子。\n共情的指标有：共情度、相关性与流程性。')
    print('\n共情度是1-5分，共情有三种展示方式（告诉对方理解他的情绪，通过提问了解更多 和 通过描述感受和个人经历展示自己感同身受）评价的方式是：')
    print('1.回复让人感到不舒服或完全没有共情的句子。')
    print('2.回复有一种共情方式，但意义不大。')
    print('3.回复有其中一种或多种共情方式，且较为合适。')
    print('4.回复中有一种或多种共情方式，且让人感到舒服。')
    print('5.回复中有一种或多种共情方式，让人感到心情愉悦、且产生共鸣。')
    print('\n相关性，与其他对话生成指标一致：\n1.回复与文本完全不相关\n2.回复与文本有一点关系\n3.回复与文本较为相关\n4.回复与文本很相关\n5.回复与文本完全相关')
    print('\n流畅度\n1.完全不知所云\n2.语句不通顺，有很多语法错误\n3.语句较为通顺，有语法错误\n4.语句通顺，有点语法错误\n5.语句通顺，无语法错误')
    print('根据以上三个指标综合评测好坏，认为第一个句子好就输入1，认为第二个句子好就输入2，认为两个句子质量一样则输入0')
    
    boo = False
    while not boo:
        temp = str(input("在阅读以上信息后，请输入yes: "))
        if temp == 'yes':
            boo = True

if __name__ == '__main__':
    information()

    data = pd.read_excel('./sampled_data_100_42_5-6_chinese.xlsx')
    
    res = evaluate(data)

    get_all_elements(res)

    print('感谢参与！！！您的参与对我们十分重要。')