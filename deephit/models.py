import os
import glob
import sys

import tensorflow as tf
import numpy as np
from deephit.utils import convert_to_graph
from sklearn.model_selection import StratifiedKFold
import sys
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import svm
import pickle

def run_descriptor_based_DNN(smiles_list, C_total, trained_model):   
    learning_rate = 0.01
    dnn_hidden_layers =  4
    dnn_hidden_nodes = 892
    dropout_rate = 0.2
    
    num_cats_features = C_total.shape[1]
    
    C = tf.placeholder(tf.float64, [None, num_cats_features])
    Y_truth = tf.placeholder(tf.float64, [None, 1])
    
    output2 = tf.layers.dense(C, dnn_hidden_nodes, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00240067850001354))
    output2 = tf.layers.dropout(output2, rate=dropout_rate)
    
    for i in range(dnn_hidden_layers-1):
        output2 = tf.layers.dense(output2, dnn_hidden_nodes, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00240067850001354))
        output2 = tf.layers.dropout(output2, rate=dropout_rate)
        
    Y_pred = tf.layers.dense(output2, 1, use_bias=True, activation=tf.nn.sigmoid)

    Y_pred = tf.reshape(Y_pred, shape=[-1])
    Y_truth = tf.reshape(Y_truth, shape=[-1])
    
    predicted = tf.cast(Y_pred > 0.5, dtype=tf.float64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y_truth), dtype=tf.float64))

    cost = -tf.reduce_mean(Y_truth * tf.log(Y_pred) + (1 - Y_truth) * tf.log(1 - Y_pred))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()
    results = {}
    with tf.Session() as sess:     
        sess.run(init)
        saver.restore(sess, trained_model)

        y_pred, predicted = sess.run([Y_pred, predicted], feed_dict={C:C_total})        
        y_predictions = list(y_pred)

        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            prob = y_predictions[i]
            results[smiles] = prob
        
    tf.reset_default_graph()
    return results

def run_fingerprint_based_DNN(smiles_list, C_total, trained_model):   
    learning_rate = 0.01
    dnn_hidden_layers =  3
    dnn_hidden_nodes = 1024
    dropout_rate = 0.5
    
    num_cats_features = C_total.shape[1]
    
    C = tf.placeholder(tf.float64, [None, num_cats_features])
    Y_truth = tf.placeholder(tf.float64, [None, 1])
    
    output2 = tf.layers.dense(C, dnn_hidden_nodes, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0074472915242877))
    output2 = tf.layers.dropout(output2, rate=dropout_rate)
    
    for i in range(dnn_hidden_layers-1):
        output2 = tf.layers.dense(output2, dnn_hidden_nodes, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0074472915242877))
        output2 = tf.layers.dropout(output2, rate=dropout_rate)
        
    Y_pred = tf.layers.dense(output2, 1, use_bias=True, activation=tf.nn.sigmoid)

    Y_pred = tf.reshape(Y_pred, shape=[-1])
    Y_truth = tf.reshape(Y_truth, shape=[-1])
    
    predicted = tf.cast(Y_pred > 0.5, dtype=tf.float64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y_truth), dtype=tf.float64))

    cost = -tf.reduce_mean(Y_truth * tf.log(Y_pred) + (1 - Y_truth) * tf.log(1 - Y_pred))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()
    results = {}
    with tf.Session() as sess:     
        sess.run(init)
        saver.restore(sess, trained_model)

        y_pred, predicted = sess.run([Y_pred, predicted], feed_dict={C:C_total})        
        y_predictions = list(y_pred)

        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            prob = y_predictions[i]
            results[smiles] = prob
        
    tf.reset_default_graph()
    return results
  
def attn_coeffs(_X, _A, _C):
    X1 = tf.transpose(_X, [0, 2, 1])
    X2 = tf.einsum('ij,ajk->aik', _C, X1)
    attn_matrix = tf.matmul(_X, X2)
    attn_matrix = tf.multiply(_A, attn_matrix)
    attn_matrix = tf.nn.tanh(attn_matrix)
    return attn_matrix

def graph_conv(_X, _A, output_dim):
    output = tf.layers.dense(_X, units=output_dim, use_bias=True)
    output = tf.matmul(_A, output)
    output = tf.nn.relu(output)
    return output

def readout_nw(_X, output_dim):
    output = tf.layers.dense(_X, output_dim, use_bias=True)
    output = tf.reduce_sum(output, axis=1)
    output = tf.nn.relu(output)
    return output

def run_graph_based_GCN(smiles_list, X_total, A_total, trained_model):   
    dnn_hidden_layers =  2
    dnn_hidden_nodes = 1024
    gcn_hidden_layers = 3
    gcn_hidden_nodes = 64
    dropout_rate = 0.4
    
    learning_rate = 0.01
            
    num_nodes = X_total.shape[1]
    num_features = X_total.shape[2]
    
    X = tf.placeholder(tf.float64, [None, num_nodes, num_features])
    A = tf.placeholder(tf.float64, [None, num_nodes, num_nodes])
    Y_truth = tf.placeholder(tf.float64, [None, 1])
    
    gconv1 = graph_conv(X, A, gcn_hidden_nodes)
    for i in range(gcn_hidden_layers-1):
        gconv1 = graph_conv(gconv1, A, gcn_hidden_nodes)  
        
    graph_feature = readout_nw(gconv1, dnn_hidden_nodes)

    output1 = tf.layers.dense(graph_feature, dnn_hidden_nodes, use_bias=True, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00001))
    output1 = tf.layers.dropout(output1, rate=dropout_rate)
    
    for i in range(dnn_hidden_layers-1):
        output1 = tf.layers.dense(output1, dnn_hidden_nodes, use_bias=True, activation=tf.nn.tanh, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.00001))
        output1 = tf.layers.dropout(output1, rate=dropout_rate)
        
    Y_pred = tf.layers.dense(output1, 1, use_bias=True, activation=tf.nn.sigmoid)

    Y_pred = tf.reshape(Y_pred, shape=[-1])
    Y_truth = tf.reshape(Y_truth, shape=[-1])

    predicted = tf.cast(Y_pred > 0.5, dtype=tf.float64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y_truth), dtype=tf.float64))

    cost = -tf.reduce_mean(Y_truth * tf.log(Y_pred) + (1 - Y_truth) * tf.log(1 - Y_pred))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    saver = tf.train.Saver()
    
    init = tf.global_variables_initializer()
    results = {}
    with tf.Session() as sess:     
        sess.run(init)
        saver.restore(sess, trained_model)
        y_pred, predicted = sess.run([Y_pred, predicted], feed_dict={X: X_total, A: A_total})
        y_predictions = list(y_pred)
        
        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            prob = y_predictions[i]
            results[smiles] = prob
    tf.reset_default_graph()
    return results