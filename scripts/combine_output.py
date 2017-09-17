import os
import sys
sys.path.insert(0, '../../../../scripts/')
import preprocessing as pp
import pandas as pd
import unidecode as ud

flag = 0

out = []
result = []
main_df = pd.DataFrame(columns = ['id', 'sent_1', 'sent_2', 'label', 'stance'])
out_main_df = pd.DataFrame(columns = ['content'])  
for filename in os.listdir(os.getcwd()):
    #print filename
    if "onew_result_1" in filename:
        #print filename
        df_1 = pd.read_csv(filename, names= None)
        main_df = main_df.append(df_1)
    if "onew_result_2" in filename:
        df_2 = pd.read_csv(filename, names=None)
        main_df = main_df.append(df_2)
    if "onew_result_3" in filename:
        df_3 = pd.read_csv(filename, names=None)
        main_df = main_df.append(df_3)
    if "onew_result_4" in filename:
        df_4 = pd.read_csv(filename, names=None)
        main_df = main_df.append(df_4)
    elif "onew_out" in filename:
        out_df = pd.read_csv(filename, names = None)
        out_main_df = out_main_df.append(out_df)


set_df = pd.DataFrame(columns = ['id', 'sent_1', 'sent_2', 'label', 'stance'])
set_df = set_df.append(df_1)
set_df = set_df.append(df_2)
set_df = set_df.append(df_3)

tr_set1_df_4 = df_4.iloc[:-2368]
tst_set1_df_4 = df_4.iloc[-2368:]

tr_set2_df_4 = df_4.iloc[ :-4732].append(df_4.iloc[-2368:])
tst_set2_df_4 = df_4.iloc[-4732:-2368]

tr_set3_df_4 = df_4.iloc[ :-7096].append(df_4.iloc[-4732:])
tst_set3_df_4 = df_4.iloc[-7096:-4732]

tr_set4_df_4 = df_4.iloc[ :-9460].append(df_4.iloc[-7096:])
tst_set4_df_4 = df_4.iloc[-9460:-7096]

tr_set5_df_4 = df_4.iloc[:2368].append(df_4.iloc[4732:])
tst_set5_df_4 = df_4.iloc[2368:4732]


#print main_df
#print len(main_df)
#print len(out_main_df)

'''
main_df = main_df.sample(frac=1).reset_index(drop=True)
df_class0 = main_df.loc[main_df['label'] == 0]
df_class0 = df_class0.sample(frac=1).reset_index(drop=True)
df_class1 = main_df.loc[main_df['label'] == 1]
df_class1 = df_class1.sample(frac=1).reset_index(drop=True)

size = len(df_class1)
df_class0 = df_class0.iloc[:size+1000]
tr_df_123 = df_class1.append(df_class0)
tr_df_123 = tr_df_123.sample(frac=1).reset_index(drop=True)

print len(tr_df_123)
tr_df_123.to_csv('tr-combined-412.csv', index=False)


print len(main_df)
print len(out_main_df)

main_df = main_df.sample(frac=1).reset_index(drop=True)
main_df.to_csv('new-combined-result.csv', index=False)
out_main_df.to_csv('new-combined-out.csv', index=False)

'''
print(len(set_df))
tr_set1_df_4 = tr_set1_df_4.append(set_df)
tr_set1_df_4 = tr_set1_df_4.sample(frac=1).reset_index(drop=True)
df_class0 = tr_set1_df_4.loc[tr_set1_df_4['label'] == 0]
df_class0 = df_class0.sample(frac=1).reset_index(drop=True)
df_class1 = tr_set1_df_4.loc[tr_set1_df_4['label'] == 1]
df_class1 = df_class1.sample(frac=1).reset_index(drop=True)

size = len(df_class1)
df_class0 = df_class0.iloc[:size+1000]
tr_df_4 = df_class1.append(df_class0)
tr_df_4 = tr_df_4.sample(frac=1).reset_index(drop=True)

print(len(tr_df_4))
print(len(tst_set1_df_4))

tr_df_4.to_csv('tr_set1_df_4.csv', index=False)
tst_set1_df_4.to_csv('tst_set1_df_4.csv', index=False)


'''
train_df = main_df.iloc[:-1000]
remain_df = main_df.iloc[-1000:]
remain_df = remain_df.sample(frac=1).reset_index(drop=True)

dev_df = remain_df.iloc[:-500]
test_df = remain_df.iloc[-500:]


#train_df.to_csv('train.csv', index=False)
#dev_df.to_csv('dev.csv', index=False)
#test_df.to_csv('test.csv', index=False)

df_class0 = main_df.loc[main_df['labels'] == 0]
df_class0 = df_class0.sample(frac=1).reset_index(drop=True)
df_class1 = main_df.loc[main_df['labels'] == 1]
df_class1 = df_class1.sample(frac=1).reset_index(drop=True)

train_df_class0 = df_class0.iloc[:-500]

#making class 0 size almost equal to class 1 size in training data
train_df_class0 = train_df_class0.iloc[:4000]

remain_df_class0 = df_class0.iloc[-500:]
dev_df_class0 = remain_df_class0[:-250]
test_df_class0 = remain_df_class0[-250:]

train_df_class1 = df_class1.iloc[:-500]
remain_df_class1 = df_class1.iloc[-500:]
dev_df_class1 = remain_df_class1[:-250]
test_df_class1 = remain_df_class1[-250:]

train_df = train_df_class0.append(train_df_class1)
train_df.sample(frac=1).reset_index(drop=True)
dev_df = dev_df_class0.append(dev_df_class1)
dev_df.sample(frac=1).reset_index(drop=True)
test_df = test_df_class0.append(test_df_class1)
test_df.sample(frac=1).reset_index(drop=True)

print len(train_df)
print len(dev_df)
print len(test_df)

train_df.to_csv('train_balanced.csv', index=False)
dev_df.to_csv('dev_balanced.csv', index=False)
test_df.to_csv('test_balanced.csv', index=False)
'''

'''
text = out_main_df['content'].tolist()
text = " ".join(text)
text = text.decode('ascii', 'ignore')
ans = pp.preprocess_pipeline(text, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
list_df = pd.DataFrame(columns = ['word'])
list_df['word'] = ans

'''
#list_df.to_csv('combined-word-list.csv', index=False)

#print len(ans)
#print ans            
