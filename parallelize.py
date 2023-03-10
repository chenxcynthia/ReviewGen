import json
import os
import time
import pickle

import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import concurrent.futures

import importlib

from utils import *

extractor = Extractor('keywords.txt', 'parameters.txt')

workdir = '../'

def extract_file(i, sub_dir='NIPS_2017', output_dir='outputdata/'):
    try:
        
        lst = []

        t0 = time.time()
        paper_dict = {}

        # Check whether content and review info exist, if not, exclude paper
        kind = 'paper'
        fname = workdir + f'dataset/{sub_dir}/{sub_dir}_'+kind+f'/{sub_dir}_'+str(i)+'_'+kind+'.json'
        with open(fname, 'r', encoding='utf8') as f:
            content_dict = json.loads(f.read())

        # Get basic paper information
        paper_dict['title'] = content_dict['title']
        paper_dict['decision'] = content_dict['decision']
        paper_dict['conference'] = content_dict['conference']

        # Get longform paper text and use the extractor to condense
        kind = 'content'
        fname = workdir + f'dataset/{sub_dir}/{sub_dir}_'+kind+f'/{sub_dir}_'+str(i)+'_'+kind+'.json'
        fulltext = get_full_text(fname)
        extraction = extractor.extract(fulltext)
        paper_dict['text'] = extraction

        # Get reviews
        kind = 'review'
        fname = workdir + f'dataset/{sub_dir}/{sub_dir}_'+kind+f'/{sub_dir}_'+str(i)+'_'+kind+'.json'
        with open(fname, 'r', encoding='utf8') as f:
            content_dict = json.loads(f.read())
        reviews = []

        for r in content_dict['reviews']:
            paper_dict_with_review = paper_dict.copy()
            paper_dict_with_review['review'] = r['review']
            lst.append(paper_dict_with_review)
    #         reviews.append(r['review'])
    #         paper_dict['reviews'] = reviews

        with open(workdir + output_dir + f'{sub_dir}_ce_extract_{i}.json', 'w', encoding='utf-8') as f:
            json.dump(lst, f)

        t1 = time.time()
        status = 'finished in ' + str(t1 - t0) + ' seconds'
    except Exception as e:
        status = "EXCEPTION! " + str(e)
        raise("EXCEPTION! " + str(e))
        return (-1, status)

    return 0, status

import multiprocessing

def run_extraction(st, end, sub_dir):
    n_cpu = multiprocessing.cpu_count()
    n_jobs = n_cpu - 2
    
    cur_st = st
    all_results = []
    while cur_st < end:
        inc = min(n_jobs, end-cur_st)
        print(cur_st)
    
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(extract_file, list(range(cur_st, cur_st+inc)), [sub_dir for _ in range(inc)])

        cur_st += inc
        for res in results:
            print(res)