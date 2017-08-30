from __future__ import print_function
import argparse
import os
import sys
import time
import re
import datetime
import tensorflow as tf
import numpy as np
import data_helpers as dh
import pandas as pd
from SentenceMatchModelGraph import SentenceMatchModelGraph


FLAGS = None
dropout_rate = 0.3
learning_rate = 0.001
best_accuracy = 0.0
init_scale = 0.01
num_classes = 2
optimize_type = 'adam'
lambda_l2 = 1e-5
with_word=True

with_char=False
with_POS=False
with_NER=False
char_lstm_dim=20

context_lstm_dim=200
aggregation_lstm_dim=200
embedding_dim = 300
is_training=True
filter_layer_threshold=0.2,
MP_dim=25
context_layer_num=1
aggregation_layer_num=1
fix_word_vec=True
with_filter_layer=False
with_highway=True

with_lex_features=False
lex_dim=100
word_level_MP_dim=-1
sep_endpoint=False
end_model_combine=False

with_match_highway=True,
with_aggregation_highway=True
highway_layer_num=1

with_lex_decomposition=False
lex_decompsition_dim=-1

with_left_match=True
with_right_match=True
with_full_match=True
with_maxpool_match=True
with_attentive_match=True
with_max_attentive_match=True

training_size = 58008
batch_size = 30
num_epochs = 8
checkpoint_every = 500
evaluate_every = (training_size/batch_size)+1
dev_batch_size = 500

lookup_filename = 'word_embedding_short.csv'
train_filename = 'train.csv'
dev_filename = 'dev.csv'

outpath = 'output.csv'

lookup_frame = pd.read_csv(lookup_filename)
lookup_labels = []

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    #         with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        train_graph = SentenceMatchModelGraph(num_classes, lookup_frame, word_vocab=None, char_vocab=None,
                                              POS_vocab=None, NER_vocab=None,
                                              dropout_rate=dropout_rate, learning_rate=learning_rate,
                                              optimize_type=optimize_type,
                                              lambda_l2=lambda_l2, char_lstm_dim=char_lstm_dim,
                                              context_lstm_dim=context_lstm_dim,
                                              aggregation_lstm_dim=aggregation_lstm_dim, is_training=True,
                                              MP_dim=MP_dim,
                                              context_layer_num=context_layer_num,
                                              aggregation_layer_num=aggregation_layer_num,
                                              fix_word_vec=fix_word_vec,
                                              with_filter_layer=with_filter_layer,
                                              with_highway=with_highway,
                                              word_level_MP_dim=word_level_MP_dim,
                                              with_match_highway=with_match_highway,
                                              with_aggregation_highway=with_aggregation_highway,
                                              highway_layer_num=highway_layer_num,
                                              with_lex_decomposition=with_lex_decomposition,
                                              lex_decompsition_dim=lex_decompsition_dim,
                                              with_left_match=with_left_match,
                                              with_right_match=with_right_match,
                                              with_full_match=with_full_match,
                                              with_maxpool_match=with_maxpool_match,
                                              with_attentive_match=with_attentive_match,
                                              with_max_attentive_match=with_max_attentive_match, embedding_dim=embedding_dim)
    tf.summary.scalar("Training Loss", train_graph.get_loss())  # Add a scalar summary for the snapshot loss.
    
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        valid_graph = SentenceMatchModelGraph(num_classes, lookup_frame, word_vocab=None, char_vocab=None,
                                              POS_vocab=None, NER_vocab=None,
                                              dropout_rate=dropout_rate, learning_rate=learning_rate,
                                              optimize_type=optimize_type,
                                              lambda_l2=lambda_l2, char_lstm_dim=char_lstm_dim,
                                              context_lstm_dim=context_lstm_dim,
                                              aggregation_lstm_dim=aggregation_lstm_dim, is_training=False,
                                              MP_dim=MP_dim,
                                              context_layer_num=context_layer_num,
                                              aggregation_layer_num=aggregation_layer_num,
                                              fix_word_vec=fix_word_vec,
                                              with_filter_layer=with_filter_layer,
                                              with_highway=with_highway,
                                              word_level_MP_dim=word_level_MP_dim,
                                              with_match_highway=with_match_highway,
                                              with_aggregation_highway=with_aggregation_highway,
                                              highway_layer_num=highway_layer_num,
                                              with_lex_decomposition=with_lex_decomposition,
                                              lex_decompsition_dim=lex_decompsition_dim,
                                              with_left_match=with_left_match,
                                              with_right_match=with_right_match,
                                              with_full_match=with_full_match,
                                              with_maxpool_match=with_maxpool_match,
                                              with_attentive_match=with_attentive_match,
                                              with_max_attentive_match=with_max_attentive_match, embedding_dim=embedding_dim)
    
    initializer = tf.global_variables_initializer()
    
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess = tf.Session(config=session_conf)
    
    sess.run(initializer)
    vocab_size = lookup_frame.shape[0]
    
    embedding_matrix = np.zeros((vocab_size+2, embedding_dim), dtype='float32')
    i=0
    for index, row in lookup_frame.iterrows():
        embedding_vector = np.array(eval(row['vector']))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            i=i+1
    embedding_matrix[vocab_size] = np.ones((1, embedding_dim))        
    
    sess.run(train_graph.get_embedding_init(), feed_dict={train_graph.get_embedding_placeholder(): embedding_matrix})

    vars_ = {}
    '''
    for var in tf.global_variables():
        if "word_embedding" in var.name: continue
        #             if not var.name.startswith("Model"): continue
        vars_[var.name.split(":")[0]] = var
    saver = tf.train.Saver(vars_)
    '''
    '''
    #----------------------------------------------------------------------------
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.merge_summary(grad_summaries)
    '''
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    #print("Writing to {}\n".format(out_dir))
    
    '''
    # Summaries for loss and accuracy
    loss_summary = tf.scalar_summary("loss", valid_graph.get_loss())
    acc_summary = tf.scalar_summary("accuracy", valid_graph.get_eval_correct())

    # Train Summaries
    train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
    '''
    
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    #------------------------------------------------------------------
    
    label_id_batch = np.random.randint(0, 2, [30])
    sent1_length_batch = np.random.randint(1, 15, [30])
    sent2_length_batch = np.random.randint(1, 15, [30])

    word_idx_1_batch = np.random.randint(1, 10, [30,15])
    word_idx_2_batch = np.random.randint(1, 10, [30,15])

    feed_dict_1 = {
        train_graph.get_truth(): label_id_batch,
        train_graph.get_question_lengths(): sent1_length_batch,
        train_graph.get_passage_lengths(): sent2_length_batch,
        train_graph.get_in_question_words(): word_idx_1_batch,
        train_graph.get_in_passage_words(): word_idx_2_batch,
        #                          train_graph.get_question_char_lengths(): sent1_char_length_batch,
        #                          train_graph.get_passage_char_lengths(): sent2_char_length_batch,
        #                          train_graph.get_in_question_chars(): char_matrix_idx_1_batch,
        #                          train_graph.get_in_passage_chars(): char_matrix_idx_2_batch,
    }
    
    feed_dict_2 = {
        valid_graph.get_truth(): label_id_batch,
        valid_graph.get_question_lengths(): sent1_length_batch,
        valid_graph.get_passage_lengths(): sent2_length_batch,
        valid_graph.get_in_question_words(): word_idx_1_batch,
        valid_graph.get_in_passage_words(): word_idx_2_batch,
        #                          valid_graph.get_question_char_lengths(): sent1_char_length_batch,
        #                          valid_graph.get_passage_char_lengths(): sent2_char_length_batch,
        #                          valid_graph.get_in_question_chars(): char_matrix_idx_1_batch,
        #                          valid_graph.get_in_passage_chars(): char_matrix_idx_2_batch,
    }
    
    '''
    _, loss_value = sess.run([train_graph.get_train_op(), train_graph.get_loss()], feed_dict=feed_dict_1)
    print(loss_value)
    loss_value = sess.run([valid_graph.get_loss()], feed_dict=feed_dict_2)
    print(loss_value)
    '''
    
    #-----------------------------------------------------------------
    
    def output_probs(probs, lookup_labels):
        out_string = ""
        for i in xrange(probs.size):
            out_string += " {}:{}".format(lookup_labels[i], probs[i])
        return out_string.strip()
        
    def train_step(x1_ba, x1_bl, x2_ba, x2_bl, y_ba):
            """
            A single training step
            """
            feed_dict = {
                train_graph.get_truth(): y_ba,
                train_graph.get_question_lengths(): x1_bl,
                train_graph.get_passage_lengths(): x2_bl,
                train_graph.get_in_question_words(): x1_ba,
                train_graph.get_in_passage_words(): x2_ba,
                #                          train_graph.get_question_char_lengths(): sent1_char_length_batch,
                #                          train_graph.get_passage_char_lengths(): sent2_char_length_batch,
                #                          train_graph.get_in_question_chars(): char_matrix_idx_1_batch,
                #                          train_graph.get_in_passage_chars(): char_matrix_idx_2_batch,
            }
            #print(x1_ba)
            #print(x1_bl)
            #print(x2_ba)
            #print(x2_bl)
            #print(y_ba)
            
            total_tags = len(y_ba)
            _, loss_value, correct_tags, step = sess.run([train_graph.get_train_op(), train_graph.get_loss(), train_graph.get_eval_correct(), train_graph.get_global_step()], feed_dict=feed_dict)
            
            #_, loss_value, correct_tags, step, summaries = sess.run([train_graph.get_train_op(), train_graph.get_loss(), train_graph.get_eval_correct(), train_graph.get_global_step(), train_summary_op], feed_dict=feed_dict)
            #print(correct_tags)
            #print(total_tags)
            accuracy = (float(correct_tags) / float(total_tags)) * 100.0
                
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_value, accuracy))
            #train_summary_writer.add_summary(summaries, step)

    def dev_step(x1_ba, x1_bl, x2_ba, x2_bl, y_ba, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
                valid_graph.get_truth(): y_ba,
                valid_graph.get_question_lengths(): x1_bl,
                valid_graph.get_passage_lengths(): x2_bl,
                valid_graph.get_in_question_words(): x1_ba,
                valid_graph.get_in_passage_words(): x2_ba,
                #                          valid_graph.get_question_char_lengths(): sent1_char_length_batch,
                #                          valid_graph.get_passage_char_lengths(): sent2_char_length_batch,
                #                          valid_graph.get_in_question_chars(): char_matrix_idx_1_batch,
                #                          valid_graph.get_in_passage_chars(): char_matrix_idx_2_batch,
        }
        
        total_tags = len(y_ba)
        loss_value, correct_tags, probs, pred, step = sess.run([valid_graph.get_loss(), valid_graph.get_eval_correct(), valid_graph.get_prob(), valid_graph.get_predictions(), valid_graph.get_global_step()], feed_dict=feed_dict)
        #loss_value, correct_tags, probs, pred, step, summaries = sess.run([valid_graph.get_loss(), valid_graph.get_eval_correct(), valid_graph.get_prob(), valid_graph.get_predictions(), valid_graph.get_global_step(), dev_summary_op], feed_dict=feed_dict)
            
        accuracy = (float(correct_tags) / float(total_tags)) * 100.0
        
        
        #if outpath is not None: outfile = open(outpath, 'at')
        #    for i in xrange(len(y_batch)):
        #        outfile.write(y_batch[i] + "\t" + output_probs(probs[i], lookup_labels) + "\n")
        #if outpath is not None: outfile.close()
        
        
        df = pd.DataFrame(probs)
        df['truth'] = y_ba
        df['predictions'] = pred
        with open(outpath, 'a') as f:
            df.to_csv(outpath, header=True, index=False)        
        
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_value, accuracy))
        #if writer:
        #    writer.add_summary(summaries, step)

        
    
    lookup_dict = dict(zip(lookup_frame.word, lookup_frame.index))
    
    x1, x1_len, x2, x2_len, y = dh.load_data(train_filename, lookup_dict)
    x1_dev, x1_dev_len, x2_dev, x2_dev_len, y_dev = dh.load_data(dev_filename, lookup_dict)
    
    train_data = dh.batch_iter(x1, x1_len, x2, x2_len, y, batch_size, num_epochs, shuffle=True)
    x1_dev_batch, x1_dev_len, x2_dev_batch, x2_dev_len, y_dev_batch = next(dh.batch_iter(x1_dev, x1_dev_len, x2_dev, x2_dev_len, y_dev, dev_batch_size, 1, shuffle=False))
    
    del(lookup_frame)
    del(lookup_dict)
    
    # Training loop
    for x1_batch, x1_blen, x2_batch, x2_blen, y_batch in train_data:
            
        if(len(x1_batch)==0 or len(x2_batch)==0): 
            break
        train_step(x1_batch, x1_blen, x2_batch, x2_blen, y_batch)
        current_step = tf.train.global_step(sess, train_graph.get_global_step())
            
        if current_step % evaluate_every == 0:
            print("\nEvaluation:")
            if(len(x1_dev_batch)==0 or len(x2_dev_batch)==0):
                break    
            dev_step(x1_dev_batch, x1_dev_len, x2_dev_batch, x2_dev_len, y_dev_batch, writer=None)
            #dev_step(x1_dev_batch, x1_dev_len, x2_dev_batch, x2_dev_len, y_dev_batch, writer=dev_summary_writer)
            print("")
    
        #if current_step % checkpoint_every == 0:
            #path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            #print("Saved model checkpoint to {}\n".format(path))
    
    print("\nEvaluation:")
    if(len(x1_dev_batch)!=0 and len(x2_dev_batch)!=0):
        dev_step(x1_dev_batch, x1_dev_len, x2_dev_batch, x2_dev_len, y_dev_batch, writer=None)
        #dev_step(x1_dev_batch, x1_dev_len, x2_dev_batch, x2_dev_len, y_dev_batch, writer=dev_summary_writer)
    print("")

