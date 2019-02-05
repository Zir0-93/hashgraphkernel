# Copyright (c) 2017 by Christopher MorrisALL
# Web site: https://ls11-www.cs.uni-dortmund.de/staff/morris
# Email: christopher.morris at udo.edu

from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import dataset_parsers as dp
from graphkernel import hash_graph_kernel as rbk
from graphkernel import shortest_path_kernel_explicit as sp_exp
from graphkernel import wl_kernel as wl
import time
import sys
import os
import traceback
import random
import multiprocessing

def analyze(dataset):
    try:
        print "Processing dataset: " + dataset
        start = time.time()
        # Load ENZYMES data set
        graph_db, classes = dp.read_txt(dataset)

        # Parameters used: 
        # Compute gram matrix: False, 
        # Normalize gram matrix: False
        # Use discrete labels: False
        #kernel_parameters_sp = [False, False, 0]

        # Parameters used: 
        # Compute gram matrix: False, 
        # Normalize gram matrix: False
        # Use discrete labels: False
        # Number of iterations for WL: 3
        #kernel_parameters_wl = [3, False, False, 0]

        # Use discrete labels, too
        kernel_parameters_sp = [False, False, 1]
        kernel_parameters_wl = [3, False, False, 20]


        # Compute gram matrix for HGK-WL
        # 20 is the number of iterations
        #gram_matrix = rbk.hash_graph_kernel(graph_db, sp_exp.shortest_path_kernel, kernel_parameters_sp, 100, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)
        # Normalize gram matrix
        #gram_matrix = aux.normalize_gram_matrix(gram_matrix)

        # Compute gram matrix for HGK-SP
        # 20 is the number of iterations
        gram_matrix = rbk.hash_graph_kernel(graph_db, wl.weisfeiler_lehman_subtree_kernel, kernel_parameters_wl, 1, scale_attributes=True, lsh_bin_width=1.0, sigma=1.0)

        # Normalize gram matrix
        gram_matrix = aux.normalize_gram_matrix(gram_matrix)

        end = time.time()
        # Write out LIBSVM matrix
        dp.write_lib_svm(gram_matrix, classes, dataset + "_gram_matrix", end - start)
    except Exception as e:
        traceback.print_exc()


def main():
    repos = ['FACEBOOK', 'GOOGLE', 'NETFLIX',  'MICROSOFT', 'APACHE', 'SPRING-PROJECTS', 'SQUARE', 'ALL']
    types = ['GENERAL', 'SPECIFIC']
    threads = 5   # Number of threads to create
    for repo in repos:
        for type in types:
            jobs = []
            for i in range(5, 10):
                out_list = list()
		dataset_name = 'DIFFS_' + repo + '_' + str(i) + '_' + type
                analysis = multiprocessing.Process(
			target=analyze,
			args=(dataset_name,)
		)
		jobs.append(analysis)
            # Start the threads (i.e. calculate the random number lists)
            for j in jobs:
                j.start()
            # Ensure all of the threads have finished
            for j in jobs:
                j.join()
            
            

    
if __name__ == "__main__":
    main()
