import numpy as np
import pandas as pd
import re
import itertools

max_len = 15
hits = 0
def getseq(seq, lookup_dict):
    size = len(lookup_dict)
    seq_num = []
    global hits

    for each in seq:
        try:
            seq_num.append(int(lookup_dict[each]))
        except:
            hits = hits+1
            if(each == 'NUM_MARKER'):
                seq_num.append(size)
            else:
                seq_num.append(size+1)
            continue
    return seq_num

def load_data(file_name, lookup_dict):
    df = pd.read_csv(file_name)
    
    x1_batch = []
    x1_lengths = []
    x2_batch = []
    x2_lengths = []
    y_labels = []
    global hits

    for index, row in df.iterrows():
        #print(row)
        temp = eval(row['sent_1'])
        tlen = len(temp)
    
        x1 = getseq(temp, lookup_dict)
        #print(temp)
        #print(x1)        
        if(max_len > tlen):
            x1.extend([0]*(max_len - tlen))
            x1_lengths.append(tlen)
        else:
            x1= x1[:max_len]
            x1_lengths.append(max_len)

        x1_batch.append(x1)
         
        temp = eval(row['sent_2']) 
        tlen = len(temp)

        x2 = getseq(temp, lookup_dict)
        
        if(max_len > tlen):
            x2.extend([0]*(max_len - tlen))
            x2_lengths.append(tlen)
        else:
            x2= x2[:max_len]
            x2_lengths.append(max_len)
        
        x2_batch.append(x2)
        
        label = int(row['labels'])
        y_labels.append(label)
    
    print "Hits are--------------------------------------------------------\n"
    print hits    
    return x1_batch, x1_lengths, x2_batch, x2_lengths, y_labels     

def batch_iter(x1_batch, x1_lengths, x2_batch, x2_lengths, y_labels, batch_size, num_epochs, shuffle=True):
    
    x1_batch = np.array(x1_batch)
    x2_batch = np.array(x2_batch)
    x1_lengths = np.array(x1_lengths)
    x2_lengths = np.array(x2_lengths)
    y_labels = np.array(y_labels)
    
    data_size = len(y_labels)
    
    if(data_size%batch_size == 0):
        num_batches_per_epoch = int(data_size/batch_size)
    else:
        num_batches_per_epoch = int(data_size/batch_size)+1
    
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x1_batch = x1_batch[shuffle_indices]
            x2_batch = x2_batch[shuffle_indices]
            x1_lengths = x1_lengths[shuffle_indices]
            x2_lengths = x2_lengths[shuffle_indices]
            y_labels = y_labels[shuffle_indices]
                    
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)    
            yield x1_batch[start_index:end_index], x1_lengths[start_index:end_index], x2_batch[start_index:end_index], x2_lengths[start_index:end_index], y_labels[start_index:end_index]
            

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    
    # Load data from files
    positive_examples = list(open("./data/rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./data/rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    
    return [x_text, y]
    
def pad_sentences(sentences, padding_word="<PAD/>", max_filter=5):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """

    # Using this might improve accuracy...

    pad_filter = max_filter -1
    sequence_length = max(len(x) for x in sentences) + 2*pad_filter

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence) - pad_filter
        new_sentence = [padding_word]*max_filter + sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)



    
    return padded_sentences

