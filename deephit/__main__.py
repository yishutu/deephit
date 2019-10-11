import logging
import os
import time
import pkg_resources
import shutil
import copy
import pandas as pd
import sys
import sklearn
import pickle
import warnings
import os
import glob
import numpy as np
import pandas as pd
from rdkit import RDLogger

from deephit import models
from deephit import utils

import warnings , os

def calculate_features(smiles_file, min_max_scaler_model):
    descriptor_list_file = pkg_resources.resource_filename('deephit', 'data/descriptors.txt')
    smi_total = utils.read_data(smiles_file)    
    
    X_total, A_total, new_smiles_list = utils.convert_to_graph(smi_total)
    
    C_total, new_smiles_list = utils.calculate_des(new_smiles_list, descriptor_list_file)
    F_total, new_smiles_list = utils.calculate_fingerprints(new_smiles_list)
    X_total, A_total, new_smiles_list = utils.convert_to_graph(new_smiles_list)
    
    min_max_scaler = pickle.load(open(min_max_scaler_model, 'rb'))
    C_total = min_max_scaler.transform(C_total)
    return new_smiles_list, X_total, A_total, C_total, F_total

def run_predictions(models, smiles_list, X_total, A_total, C_total, F_total, dnn_descriptor_model, dnn_fingerprint_model, gcn_model):
    DNN_results = models.run_descriptor_based_DNN(smiles_list, C_total, dnn_descriptor_model)
    GCN_results = models.run_graph_based_GCN(smiles_list, X_total, A_total, gcn_model) 
    DNN_F_results = models.run_fingerprint_based_DNN(smiles_list, F_total, dnn_fingerprint_model)
    
    results = {}
    for smiles in DNN_results:
        if smiles in GCN_results and smiles in DNN_F_results:
            results[smiles] = {}
            DNN_prob = DNN_results[smiles]
            GCN_prob = GCN_results[smiles]
            DNN_F_prob = DNN_F_results[smiles]

            results[smiles]['Descriptor-based DNN'] = DNN_prob
            results[smiles]['Graph-based GCN'] = GCN_prob
            results[smiles]['Fingerprint-based DNN'] = DNN_F_prob
    
    df = pd.DataFrame.from_dict(results)
    return df.T

def main():   
    warnings.filterwarnings(action='ignore')
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    warnings.filterwarnings('ignore')
    start = time.time()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    min_max_scaler_model = pkg_resources.resource_filename('deephit', 'data/min_max_model.model')
    
    dnn_descriptor_model = pkg_resources.resource_filename('deephit', 'data/model/DNN.ckpt')
    dnn_fingerprint_model = pkg_resources.resource_filename('deephit', 'data/model/DNN_F.ckpt')
    gcn_model = pkg_resources.resource_filename('deephit', 'data/model/GCN.ckpt')
    
    parser = utils.argument_parser()
    
    options = parser.parse_args()
    smiles_file = options.smiles_file
    output_dir = options.output_dir
        
    try:
        os.mkdir(output_dir)
    except:
        pass
    
    smiles_list, X_total, A_total, C_total, F_total = calculate_features(smiles_file, min_max_scaler_model)
    feature_info = {}
    feature_len = 0
    for i in range(len(smiles_list)):
        smi = smiles_list[i]
        feature = C_total[i]
        feature_info[smi] = feature
        feature_len = len(feature)
    
    df = run_predictions(models, smiles_list, X_total, A_total, C_total, F_total, dnn_descriptor_model, dnn_fingerprint_model, gcn_model)
    df.to_csv(output_dir+'/Prediction_results.tsv', sep='\t')
    
    # label 
    label_info = {}
    with open(smiles_file, 'r') as fp:
        for line in fp:
            sptlist = line.strip().split('\t')
            label = sptlist[0].strip()
            smiles = sptlist[1].strip()
            label_info[smiles] = {}
            label_info[smiles]['label'] = label
    
    outfp = open(output_dir+'/feature.tsv', 'w')
    feature_list = []
    for i in range(feature_len):
        feature_list.append('Feature %s'%(i+1))
        
    outfp.write('%s\t%s\t%s\n'%('Smiles', 'Label', '\t'.join(feature_list)))
    for smi in feature_info:
        feature = feature_info[smi]
        if smi in label_info:
            label = label_info[smi]['label']
            str_feature = [str(val) for val in feature]
            outfp.write('%s\t%s\t%s\n'%(smi, label, '\t'.join(str_feature)))
    outfp.close()
    
    df = pd.DataFrame.from_dict(label_info)
    df.T.to_csv(output_dir+'/label.tsv', sep='\t')
    
    df1 = pd.read_table(output_dir+'/Prediction_results.tsv', index_col=0)
    df2 = pd.read_table(output_dir+'/label.tsv', index_col=0)
    
    new_df = pd.concat([df1, df2], axis=1)
    new_df = new_df[['label', 'Descriptor-based DNN','Fingerprint-based DNN','Graph-based GCN']]
    
    fp = open(output_dir+'/Final_results.txt', 'w')
    fp.write('%s\t%s\n'%('ID', 'Label'))
    for each_row, each_df in new_df.iterrows():
        compound_id = each_df['label']
        descriptor_dnn = each_df['Descriptor-based DNN']
        fingerprint_dnn = each_df['Fingerprint-based DNN']
        graph_gcn = each_df['Graph-based GCN']
        
        max_prob = np.max([descriptor_dnn, fingerprint_dnn, graph_gcn])
        prediction_result = 'hERG non-blocker'
        if max_prob > 0.5:
            prediction_result = 'hERG blocker'
        fp.write('%s\t%s\n'%(compound_id, prediction_result))
    fp.close()
    
    new_df.to_csv(output_dir+'/Raw_data.tsv', sep='\t')
    # shutil.move(output_dir+'/Raw_data.tsv', raw_data_dir+'Raw_data.tsv')
    
    os.remove(output_dir+'/label.tsv')
    os.remove(output_dir+'/feature.tsv')
    os.remove(output_dir+'/Prediction_results.tsv')
    
    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))

if __name__ == '__main__':
    main()