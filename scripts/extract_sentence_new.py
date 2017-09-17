import os
import sys
sys.path.insert(0, '../../../../scripts/')
import preprocessing as pp
import pandas as pd

flag = 0
dict_p={}
dict_c={}
char = 'a'
with open('../labels/abortion.txt') as f1:
    for line in f1:
        if line=='\n': 
            continue
        if 'p-' in line:
            key = line.strip()
            flag = 1
            #print key
        elif 'c-' in line:
            key = line.strip()
            flag = 0
        else:
            if(flag==1):
                dict_p[key] = line.strip()
            else:
                dict_c[key] = line.strip()
            
print len(dict_p)
print len(dict_c)

id_result = []
oth_list = []
out = []
result = [] 
sent2 = []
count = 0
for filename in os.listdir(os.getcwd()):
    if ".rsn" not in filename:
        continue
    with open(filename,'r') as f:
        #print("\n\n\n********************"+filename + "\n")
        #if(flag > 0):
        #	print("\n\n\n********************"+filename + "\n")
        #max1 = 0
        flag = 0
        i=0
        for line in f:
            try:
                oth_flag = 0
                if i==0:
	            out.append(line)
	        else:    
	            if "Label##" in line:
                        #print line
                        word = line.split("##")[1].strip()
                        if 'p-' in word:
                            flag = 1
                            sent2 = dict_p[word].strip()
                            #print "hey"
                        elif 'c-' in word:
                            flag = 0
                            sent2 = dict_c[word].strip()
                            #print "yo"
                        sent2_list = pp.preprocess_pipeline(sent2, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
                    elif "Line##" in line:
                        word_1 = line.split("##")[1]
                        sent1 = word_1
                        sent1_list = pp.preprocess_pipeline(sent1, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
                        if '-other' in word:
                            oth_flag = 1
                        if '-Other' in word:
                            oth_flag = 1
                        inx = char + '{0:05}'.format(count)     
                        #print inx
                        #print sent1
                        if(oth_flag == 0):
                            id_result.append(tuple((inx, sent1, sent2))) 
                            #print id_result[-1]
                            result.append(tuple((inx, sent1_list, sent2_list, 1, flag)))
                            #print result[-1] 
                            count = count+1
                        else:
                            oth_list.append(tuple((inx, sent1, flag)))
                            count = count+1
                            oth_flag = 0

                        for each in dict_p.keys():
                            if '-other' in each:
                                continue
                            if '-Other' in each:
                                continue
                            if each != word:
                                #print each
                                #print len(each)
                                #print word
                                #print len(word)
                                #print "\n"
                                new_sent2 = dict_p[each].strip()
                                new_sent2_list = pp.preprocess_pipeline(new_sent2, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
                                inx = char + '{0:05}'.format(count)
                                id_result.append(tuple((inx, sent1, new_sent2)))
                                result.append(tuple((inx, sent1_list, new_sent2_list, 0, 1)))
                                count = count+1
                                #print result[-1]
                        for each in dict_c.keys():
                            if '-other' in each:
                                continue
                            if '-Other' in each:
                                continue
                            if each != word:
                                #print each
                                #print len(each)
                                #print word
                                #print len(word)
                                #print "\n"
                                new_sent2 = dict_c[each].strip()
                                new_sent2_list = pp.preprocess_pipeline(new_sent2, lang='english', stemmer_type ='WordNetLemmatizer', return_as_list=True, do_remove_stopwords=True)
                                inx = char + '{0:05}'.format(count)
                                id_result.append(tuple((inx, sent1, new_sent2)))
                                result.append(tuple((inx, sent1_list, new_sent2_list, 0, 0)))
                                count = count+1
                                #print result[-1]
                i=i+1
            except:
                print "error"
                print line
                continue

print(len(result))
print(len(oth_list))
df = pd.DataFrame(result, columns=['id', 'sent_1', 'sent_2', 'label','stance'])
df.to_csv('onew_result_1.csv', index=False)
id_df = pd.DataFrame(id_result, columns=['id', 'sent1', 'sent2'])
id_df.to_csv('oid_1.csv', index=False)
df1 = pd.DataFrame(out, columns=['content'])
df1.to_csv('onew_out_1.csv', index=False)
oth_df = pd.DataFrame(oth_list, columns=['id', 'sent1','stance'])
oth_df.to_csv('other_ids_1.csv', index=False)
