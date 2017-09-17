import numpy as np
import pandas as pd
from collections import defaultdict

#df = pd.read_csv('output_zsl_123.csv', names=['inx', '0', '1', 'truth', 'predictions'])
df = pd.read_csv('tr_set1_df_4.csv')
out_df = pd.read_csv('new_oid_4.csv')
out_df = out_df[-2096:].reset_index(drop=True)

start = 0
for a in range(1):
    count = 0
    dict_truth = {}
    dict_pred = {}
    dict_oth = {}
    i=0
    for i in range(651):
        end = start+16
        in_df = df[start:end]
        #print in_df
        #start = end
        #count = count+1
        vals = in_df['1'].tolist()
        #print vals
        inx_list = in_df.index.values.tolist()
        inx = np.argmax(vals)
        #print inx
        #print("Hello")
        #print inx_list
        #print inx_list[inx]
        #print in_df['truth']
        
        begin = (start - (a))%2096
        #begin = start
        
        num = int((in_df['truth'].tolist()[0]))
        #print num
        if(num == 1):
            #row = out_df.iloc[out_df['id'] == in_df['id'][0]]
            row = out_df.iloc[begin]
            if(row['sent2'] not in dict_truth.keys()):
                dict_truth[row['sent2']]=[]
            dict_truth[row['sent2']].append(count)
            #print "hey---------------"
            #print(dict_truth)
        
        else:
            dict_oth[count]=np.max(vals)
            print("YAY")
        #vals = in_df['predictions']
        #inx = np.argmax(vals)
        #select = in_df.iloc[inx]
        #begin = (start - (a))%2096
        #print(begin+inx)
        row = out_df.iloc[begin+inx]
        if(row['sent2'] not in dict_pred.keys()):
            dict_pred[row['sent2']]=[]
        dict_pred[row['sent2']].append(count)
        start = end
        count = count+1
    
    total_tp = 0
    total_pd = 0
    total_rd = 0
    
    #print len(dict_truth.keys())
    #print("ypppppppppiii-----------------")
    #print len(dict_pred.keys())
    #print dict_truth.keys()
    #print("-----------")
    #print dict_pred.keys()
    #s=0
    for each in dict_pred.keys():
        #print each
        a = dict_truth[each]
        b = dict_pred[each]
        tp = list(set(a) & set(b))
        #print a
        #print a
        #print b
        print len(tp)
        #s=s+len(b)
        precision = len(tp)/(1.0*len(b))
        recall = len(tp)/(1.0*len(a))
        if(precision !=0 and recall!=0):
            f1 = (2.0*precision*recall)/(precision+recall)
        #print f1
        total_tp = total_tp + len(tp) 
        total_pd = total_pd + len(b)
        total_rd = total_rd + len(a)
        #print ("\n")

    #print s
    #print total_tp

    tprecision = (total_tp/(1.0*total_pd))
    trecall = (total_tp/(1.0*total_rd))
    tf1 = ((2*tprecision*trecall)/(tprecision+trecall))
    #print('\n')
    #print(tprecision)
    #print(trecall)
    print(tf1)
    #print("---------------------")
    
    #start = start+1

    print("***********")
    print dict_oth
    print('\n')
    #print(np.max(dict_oth.values()))
    #print(np.min(dict_oth.values()))
