"""
This file is used to compute the pair-wise similarities among the molecules.

The input of this file is the file with fingerprints instead of SMILES strings.

To quickly get the fingerprints with Chemfp, you can use the command below:

    rdkit2fps --morgan --output <your_output_fp_name> <your_smiles_string_file_path>

Each line of the file with smiles strings should be:
    "SMILE_STRING <index>"
, such as "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1 0".
"""
import argparse
import numpy as np
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import time
import chemfp
from contextlib import closing
from functools import partial

def get_pairs(fingerprints, threshold=0.6):
    fp1, fp2 = fingerprints
    pairs = []
    for (query_id, hits) in chemfp.threshold_tanimoto_search(fp1, fp2, threshold=threshold):
        if len(hits) == 0: continue
        tmp = [(query_id, sim[0], sim[1]) for sim in hits.get_ids_and_scores() if sim[0] != query_id]
        pairs.extend(tmp)
    return pairs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', type=str, help="The path of fingerprints")
    parser.add_argument('--ncpu', type=int, default=mp.cpu_count(), help="The number of processes used to compute the similarities within the ")
    parser.add_argument('--split', type=int, default=-1, help="The number here is used to specify the index of batch. " + \
                                                        "Note that to accelerate the computing, we can split the data into multiple batches" + \
                                                        "and run multiple batches on different machines." + \
                                                        "If the dataset to be procesed is small, do not specify the split number and compute ")
                            multiple batches at different machines simultaneously.")
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--outpath', type=str, default=path+"/alldata/pair/")
    parser.add_argument('--threshold', type=float, default=0.6)
    args = parser.parse_args()
    
    fps = chemfp.load_fingerprints(args.fp)
    split = args.split
    batches = []
    start = 0
    output_path = None
    
    if split >= 0:
        start = split * args.batch_size
        if start > len(fps): quit()
        split_query = fps[start: min(len(fps), (split+1) * args.batch_size)]
        output_path = args.outpath+"pair_split%d.pkl" % (split)
    else:
        split_query = fps
        output_path = args.outpath+"pair.pkl"
    
    pool_size = int(len(fps)-start / args.ncpu) + 1
    for idx in range(args.ncpu):
        tmp1 = split_query.copy()
        tmp2 = fps[start + idx * pool_size : min(start+(idx+1) * pool_size, len(fps))]
        batches.append((tmp1, tmp2))    
    
    func = partial(get_pairs, threshold=args.threshold)
    with closing(Pool(processes=args.ncpu)) as pool:
        pairs_list = pool.map(func, batches)
        pool.terminate()
    
    data = [pair for pairs in pairs_list for pair in pairs]
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)