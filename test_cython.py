#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:56:38 2020

@author: steffen
"""
import os
import numpy as np

os.system('python setup.py build_ext --inplace')
from AIS.func import say_hello_to
from func import cy_affinity
from main_cy import AIRS
import pandas as pd

say_hello_to('Steffen')

nd1 = np.array([1,2,3,4])
nd2 = np.array([1,5,3,5])

aff = cy_affinity(nd1, nd2)

ARRAY_SIZE = 30  # Features number
MAX_ITER = 10  # Max iterations to stop training on a given antigene

# Mutation rate for ARBs
MUTATION_RATE = 0.2

data = pd.read_csv('data/creditcard.csv', nrows=1000)

# Very low nr of fraud cases, upsample cases.

airs = AIRS(hyper_clonal_rate=20,
            clonal_rate=0.8,
            class_number=2,
            mc_init_rate=0.4,
            total_num_resources=10,
            affinity_threshold_scalar=0.8,
            k=6,
            test_size=0.4,
            data=data)
            #input_data_file='data/creditcard.csv')

airs.train()
