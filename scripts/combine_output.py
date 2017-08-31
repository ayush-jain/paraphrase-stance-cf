import os
import sys
sys.path.insert(0, '../../../../scripts/')
import preprocessing as pp
import pandas as pd
import unidecode as ud

flag = 0

out = []
result = []
main_df = pd.DataFrame(columns = ['sent_1', 'sent_2', 'labels'])
out_main_df = pd.DataFrame(columns = ['content'])  
for filename in os.listdir(os.getcwd()):
    #print filename
    if "result_" in filename:
        #print filename
        df = pd.read_csv(filename, names= None)
        main_df = main_df.append(df)
    elif "out_" in filename:
        out_df = pd.read_csv(filename, names = None)
        out_main_df = out_main_df.append(out_df)

#print main_df
#print len(main_df)
#print len(out_main_df)

main_df = main_df.sample(frac=1).reset_index(drop=True)
#main_df.to_csv('combined-result.csv', index=False)
#out_main_df.to_csv('combined-out.csv', index=False)
'''
train_df = main_df.iloc[:-1000]
remain_df = main_df.iloc[-1000:]
remain_df = remain_df.sample(frac=1).reset_index(drop=True)

dev_df = remain_df.iloc[:-500]
test_df = remain_df.iloc[-500:]
'''
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
