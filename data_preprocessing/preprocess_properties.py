# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:41:00 2019

@author: ziqi
"""
import sys, os
sys.path.insert(0, os.path.abspath('../model/'))
from properties import similarity, drd2, qed, penalized_logp
import argparse
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from rdkit.Chem import AllChem
import pdb
import pickle

props = None

def calculate_properties(smile_idx):
    smile = smile_idx[0]
    logp = penalized_logp(smile)
    d2 = drd2(smile)
    qd = qed(smile)

    return (smile_idx[1], smile, logp, d2, qd)

def add_diff(pair_idxs):
    idx1, idx2, sim = pair_idxs
    prop1 = props[int(idx1)]
    prop2 = props[int(idx2)]
    smile1, smile2 = prop1[1], prop2[1]
    diff_logp = prop1[2] - prop2[2]
    diff_drd2 = prop1[3] - prop2[3]
    diff_qed = prop1[4] - prop2[4]
    return (smile1, smile2, sim, prop1[2], prop2[2], diff_logp, prop1[4], prop2[4], diff_qed, prop1[3], prop2[3], diff_drd2)
    
parser = argparse.ArgumentParser()
parser.add_argument('--smile', type=str, help="This is the path of file with smile strings, that is, the file used to produce the fingerprint.")
parser.add_argument('--pairpath', type=str, default="../data/pair_with_prop.pkl", help="")
parser.add_argument('--ncpu', type=int, default=mp.cpu_count())
parser.add_argument('--outprop-file', dest="outfile", type=str, default="../data/all_prop.pkl")
parser.add_argument('--outpair-file', dest="pairfile", type=str, default="../data/pair.pkl")
args = parser.parse_args()

files = os.listdir(args.pairpath)
pair_idxs = []
for file_name in files:
    data = pickle.load(open(args.pairpath+file_name, 'rb'))
    tmp_smile_idx = [pair for pair in data]
    pair_idxs.extend(tmp_smile_idx)

lines = open(args.smile, 'r')
smiles = [line.rstrip().split(" ") for line in lines]

with Pool(processes=args.ncpu) as pool:
    props = pool.map(calculate_properties, smiles)

with open(args.outfile, 'wb') as f:
    pickle.dump(props, f, pickle.HIGHEST_PROTOCOL)

with Pool(processes=args.ncpu) as pool:
    pairs = pool.map(add_diff, pair_idxs)

with open(args.pairfile, 'wb') as f:
    pickle.dump(pairs, f, pickle.HIGHEST_PROTOCOL)
