import os
import sys
sys.path.insert(0, '../../../../scripts/')
import preprocessing as pp
import pandas as pd

flag = 0
dict={}
with open('../labels/abortion.txt') as f1:
    for line in f1:
        if line=='\n': 
            continue
        if 'p-' in line or 'c-' in line:
            key = line.strip()
            #print key
        else:
            dict[key] = line.strip()
            
print dict

out = []
result = [] 
for filename in os.listdir(os.getcwd()):    
    with open(filename,'r') as f:
        #print("\n\n\n********************"+filename + "\n")
        #if(flag > 0):
        #	print("\n\n\n********************"+filename + "\n")
        #max1 = 0
        #flag = 0
        i=0
        for line in f:
            try:
                if i==0:
	            out.append(line)
	        else:    
	            if "Label##" in line:
                        #print line
                        word = line.split("##")[1].strip()
                        sent2 = dict[word].strip()
                        sent2_list = pp.preprocess_pipeline(sent2, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
                    elif "Line##" in line:
                        word = line.split("##")[1]
                        sent1 = word
                        sent1_list = pp.preprocess_pipeline(sent1, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
            	        result.append(tuple((sent1_list, sent2_list, 1)))
            	        print result[-1]
                        for each in dict.keys():
                            if each is not word:
                                new_sent2 = dict[each].strip()
                                new_sent2_list = pp.preprocess_pipeline(new_sent2, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
                                result.append(tuple((sent1_list, new_sent2_list, 0)))
                           
                i=i+1           
            except:
                continue

        df = pd.DataFrame(result, columns=['sent_1', 'sent_2', 'labels'])
        df.to_csv('result_1.csv', index=False)
        df1 = pd.DataFrame(out, columns=['content'])
        df1.to_csv('out_1.csv', index=False)
