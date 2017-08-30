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
    if "result_" in filename:
        df = pd.read_csv('result_1.csv', names= None)
        main_df = main_df.append(df)
    else:
        out_df = pd.read_csv('out_1.csv', names = None)
        out_main_df = out_main_df.append(out_df)

#print main_df
#print len(main_df)
#print len(out_main_df)

main_df = main_df.sample(frac=1).reset_index(drop=True)
#main_df.to_csv('combined-result.csv', index=False)
#out_main_df.to_csv('combined-out.csv', index=False)
train_df = main_df.iloc[:-1000]
remain_df = main_df.iloc[-1000:]
remain_df = remain_df.sample(frac=1).reset_index(drop=True)

dev_df = remain_df.iloc[:-500]
test_df = remain_df.iloc[-500:]

train_df.to_csv('train.csv', index=False)
dev_df.to_csv('dev.csv', index=False)
test_df.to_csv('test.csv', index=False)

'''
text = out_main_df['content'].tolist()
text = " ".join(text)
text = text.decode('ascii', 'ignore')
ans = pp.preprocess_pipeline(text, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
list_df = pd.DataFrame(columns = ['word'])
list_df['word'] = ans
list_df.to_csv('combined-word-list.csv', index=False)
'''
#print len(ans)
#print ans            
