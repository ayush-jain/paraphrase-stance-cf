import pandas as pd
import numpy as np
import gensim as gs
import unidecode as ud 

EMBEDDING_DIM = 300

model = gs.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True) 
#embedding_matrix = np.zeros((len(model.wv.vocab) + 1, EMBEDDING_DIM))
headers = ['word', 'count', 'index', 'vector']
df = pd.DataFrame(columns = headers)
i=0
temp={}
temp['word'] = []
temp['count'] = []
temp['index'] = []
temp['vector'] = []
min_val =5000000
for i in range(len(model.wv.vocab)):
    try:
        temp['word'].append(ud.unidecode(model.wv.index2word[i]))
        embedding_vector = model.wv[temp['word'][-1]]
        if embedding_vector is not None:
            word = model.wv.vocab[temp['word'][-1]]
            if (word.count<min_val):
                min_val = word.count
            if (word.count < 100000):
                temp['word'].pop()
                continue
            temp['count'].append(word.count)
            temp['index'].append(word.index)
            print temp['index'][-1]
            temp['vector'].append(np.array2string(embedding_vector, separator=',', precision = 16))
            #print temp['vector'][-1]
            #print embedding_vector
            #print i
            #embedding_matrix[i] = embedding_vector
        
            #new_df = pd.DataFrame.from_dict([temp])
            #df = df.append(new_df, ignore_index=True)
    except:
        temp['word'].pop()
        continue
#free memory
del(model)
df = pd.DataFrame.from_dict(temp)
df.to_csv('word_embedding_mini.csv', index=False)
print min_val
