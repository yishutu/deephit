import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from PyBioMed.PyMolecule.cats2d import CATS2D
from PyBioMed.PyMolecule import cats2d
from PyBioMed.PyMolecule.fingerprint import CalculateECFP2Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculateECFP4Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculateECFP6Fingerprint
from PyBioMed.PyMolecule.fingerprint import CalculatePubChemFingerprint
from PyBioMed import Pymolecule
from mordred import Calculator, descriptors
import random
import multiprocessing
import argparse
import copy
import os
import shutil
import pandas as pd

def argument_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-i', '--smiles_file', required=True, help="Input smiles file") 
    return parser

def read_data(filename):
    f = open(filename, 'r')
    contents = f.readlines()

    smiles = []
    labels = []
    features = []   
    error_cnt = 0

    for i in contents:
        smi = i.split()[1]
        
        try:
            iMol = Chem.MolFromSmiles(smi.strip())
            for atom in iMol.GetAtoms():
                atom_feature(atom)
        except:
            continue
            
        smiles.append(smi)
       
    return np.asarray(smiles)

def calculate_fingerprints(smi_total, method='ecfp2'): 
    features = []
    new_smiles = []
    
    for smi in smi_total:
        mol = Chem.MolFromSmiles(smi)
        if method == 'ecfp4':
            mol_fingerprint = CalculateECFP4Fingerprint(mol)
        elif method == 'ecfp2':
            mol_fingerprint = CalculateECFP2Fingerprint(mol)
        elif method == 'ecfp6':
            mol_fingerprint = CalculateECFP6Fingerprint(mol)
        else:
            mol_fingerprint = CalculateECFP4Fingerprint(mol)
        
        pubchem_mol_fingerprint = CalculatePubChemFingerprint(mol)
        
        feature1 = mol_fingerprint[0]
        feature2 = pubchem_mol_fingerprint
        feature = list(feature1)+list(feature2)
        features.append(feature)
        new_smiles.append(smi)
            
    return np.asarray(features), np.asarray(new_smiles)

def calculate_des(smi_total, des_file):        
    feature_info = calculate_mordred_descriptors(smi_total, des_file)
    
    features = []
    new_smiles = []
    
    for smi in feature_info:
        feature = feature_info[smi]
        feature = list(feature)
        features.append(feature)
        new_smiles.append(smi)
    return np.asarray(features), np.asarray(new_smiles)
    
def calculate_mordred_descriptors(smiles_list, des_file):
    descriptor_names = []
    with open(des_file, 'r') as fp:
        for line in fp:
            descriptor_names.append(line.strip())
    
    calc = Calculator(descriptors, ignore_3D = True)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list] 
    
    nproc = 1
    if multiprocessing.cpu_count() >= 4:
        nproc = 4
    else:
        nproc = multiprocessing.cpu_count()
    
    if nproc >= len(smiles_list):
        nproc = len(smiles_list)
        
    df = calc.pandas(mols, nproc=nproc)
    
    new_df = df[descriptor_names]
    #new_df.fillna(0.0)
    new_df = new_df.fillna(0.0)
    
    feature_info = {}
    idx = 0 
    for each_row, each_df in new_df.iterrows():
        smiles = smiles_list[idx]
        feature = each_df.values
        if np.max(feature) < 1000000:
            feature_info[smiles] = feature
        idx+=1
    return feature_info

def convert_to_graph(smiles_list):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50
    cnt = 0
    new_smiles_list = []
    for i in smiles_list:
        cnt+=1
        # Mol
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if( iAdjTmp.shape[0] <= maxNumAtoms):
            # Feature-preprocessing
            iFeature = np.zeros((maxNumAtoms, 65))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append( atom_feature(atom) ) ### atom features only
            iFeature[0:len(iFeatureTmp), 0:65] = iFeatureTmp ### 0 padding for feature-set
            features.append(iFeature)

            # Adj-preprocessing
            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(np.asarray(iAdj))
            new_smiles_list.append(i)
            
    features = np.asarray(features)
    adj = np.asarray(adj)
    return features, adj, new_smiles_list
    
def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                    [atom.GetIsAromatic()] + get_ring_info(atom))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_ring_info(atom):
    ring_info_feature = []
    for i in range(3, 9):
        if atom.IsInRingSize(i):
            ring_info_feature.append(1)
        else:
            ring_info_feature.append(0)
    return ring_info_feature