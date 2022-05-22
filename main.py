"""
Implementation of AIRS2 (Artificial Immune Recognition System V2), first
applied to the IRIS data set by: Azzoug Aghiles

Forked and modified by Steffen Kjær Jacobsen to handle general dataframe
inputs, not only IRIS dataset.

Log by SKJAC:
    - Optimized affinity threshold calculation, was previously brute-force.


Terms:
(Affinity threshold)
Antigenic recognition is the first pre-requisite for the immune system to be activated
and to mount an immune response. The recognition has to satisfy some criteria. First,
the cell receptor recognises an antigen with a certain affinity, and a binding between the
receptor and the antigen occurs with strength proportional to this affinity. If the affinity
is greater than a given threshold, named affinity threshold, then the immune system is
activated. The nature of antigen, type of recognising cell, and the recognition site also
influence the outcome of an encounter between an antigen and a cell receptor.

- Affinity threshold is where the affinity between an incoming foreign cell
an the immune cell is high enough to activate the immune system and select
the immune cell for multiplication to identify the foreign, invading cells.

Steps:
    Find affinity threshold using all training data.
    Initialize MC population.
    

Implementation notes:
    affinity() ensures that the distance between two vectors is always normalized.
    It is not necessary to normalize all vectors before hand. Computational gain
    would be negligible.
    
    affinity threshold is the average distance between all antigens (cells/rows in training set).
    This is affirmed according to algorithm instructions.
    
    _stimulate() affirmed.


Glossary:
    - Affinity is the distance between two cells. In code, it is the 
    (Normalized!! distance) between two features vectors.
    - Affinity threshold of the memory cells - this is this treshold where the
    immune system is activated.
    
    Stimulation threshold is the mechanism with the use of an
    average stimulation threshold as a criterion for determining when to
    stop training on ag.

    MC: Memory Cells
    AB: Antibodies

Datastructure:
    MC: dict of with two keys [0, 1], denoting class. Values are two separate pandas Dataframes
    AB: dict of with two keys [0, 1], denoting class. Values are two separate pandas Dataframes

Improvements:
    Stimulation function is currently simply 1 - affinity, but could be any
    inversely proportional function. Some room for innovation here, using a
    nonlinear function if it is deemed useful.

    Check that data is properly normalized before use. All affinity and stimulation
    calculations must return a number in [0;1].

Parameters:
    
    AFFINITY_THRESHOLD_SCALAR: The scalar value which modifies the affinity
                               threshold.
    MUTATION_RATE: The fraction of input vector elements who will be assigned
    a new random value.
    
    mc_init_rate: Fraction of memory cells relative to the input training
                  set number of cells (number of rows).

@authors: Steffen Kjær Jacobsen and Azzoug Aghiles.
"""

import random
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np
import tinyarray as ta
import os
#os.system('python setup.py build_ext --inplace')
#from func import cy_affinity
import multiprocessing
from copy import copy

# Substitute AIRS.affinity with cy_affinity

#import pandas_profiling


class ARB:
    """ARB (Artificial Recognition Ball) class
    Args:
        vector (list) : list of features
        _class (integer) : the class of the previous features
    """

    def __init__(self, vector=None, _class=None):
        if vector is None:
            self.vector = [random.random() for _ in range(ARRAY_SIZE)]
        else:
            self.vector = np.array(vector)
        self._class = _class
        self.stimulation = float('-inf')
        self.resources = 0

# =============================================================================
#     def __str__(self):
#         return "ARB : Vector = {} | class = {} | stim = {} | res = {}".format(self.vector, self._class,
#                                                                               self.stimulation, self.resources)
# 
#     def __repr__(self):
#         return "ARB : Vector = {} | class = {} | stim = {} | res = {}".format(self.vector, self._class,
#                                                                               self.stimulation, self.resources)
# =============================================================================

# =============================================================================
#     def stimulate(self, pattern):
#         """
#         Stimulation of the ARB is inversely proportional with the affinity of
#         the cell. Affinity ranges from [0;1] so stimulation ranges from 
#         [1;0] due to start value of stim of 1.
#         
#         Parameters
#         ----------
#         pattern : TYPE
#             DESCRIPTION.
# 
#         Returns
#         -------
#         self.stimulation: float
#             Stimulation value of the cell.
# 
#         """
#         
#         self.stimulation = 1 - cy_affinity(vector1=pattern,
#                                              vector2=self.vector)
#         return self.stimulation
# 
# =============================================================================
# =============================================================================
#     def _mutate(self):
#         """
#         For each element in vector, assign a random number in the range
#         [0.1, 7.1], if the random 'change' value is below MUTATION_RATE.
#         
#         Returns
#         -------
#         ARB: ARB instance
#             A new ARB instance with a new vector with either the same
#             value in element, or new mutated value.
#         mutated: bool
#             Informs us if any mutated vector elements were returned.
#         """
# 
#         # _range = 1 - self.stimulation
#         mutated = False
#         new_vector = []
# 
#         for v in self.vector:
#             change = random.random()
#             change_to = 7 * random.random() + 0.1
# 
#             if change <= AIRS.MUTATION_RATE:
#                 new_vector.append(change_to)
#                 mutated = True
#             else:
#                 new_vector.append(v)
# 
#         return ARB(vector=new_vector, _class=self._class), mutated
# 
# =============================================================================

class Cell:
    """Cell class
    Args:
        vector (list) : list of features
        _class (integer) : the class of the previous features
    """

    def __init__(self, vector=None, _class=None):
        
        assert vector is not None, 'Cannot create cell with no features'
        #    self.vector = [random.random() for _ in range(ARRAY_SIZE)]
        #else:

        self.vector = np.array(vector)  # Vector containing all cell features.
        self._class = _class  # Cell class 
        self.stimulation = float('-inf')

# =============================================================================
#     def __str__(self):
#         return "Cell : Vector = {} | class = {} | stim = {}".format(self.vector, self._class, self.stimulation)
# 
#     def __repr__(self):
#         return "Cell : Vector = {} | class = {} | stim = {}".format(self.vector, self._class, self.stimulation)
# =============================================================================

# =============================================================================
#     def _mutate(self):
#         #_range = 1 - self.stimulation
#         mutated = False
#         new_vector = []
# 
#         for v in self.vector:
#             change = random.random()
#             change_to = random.random()
# 
#             if change <= MUTATION_RATE:
#                 new_vector.append(change_to)
#                 mutated = True
#             else:
#                 new_vector.append(v)
# 
#         return ARB(vector=new_vector, _class=self._class), mutated
# =============================================================================

class AIRS:
    """AIRS (Artificial Immune Recognition System) class
    Main class for this algorithm
    
    This implementation uses negative selection and clonal selection to imitate
    B-Cells in the human immune system.
    
    Parameters
    ------------------
    hyper_clonal_rate: float
        Define the number of clones an ARB is allowed to produce
    clonal_rate: float
        Define the number of ressources an ARB can obtain.
    class_number: int
        The number of classes (2 for fraud detection)
    mc_init_rate: float
        Define the number of training data to be copied in memory cells
    total_num_resources: float
        The total numbers of resources to share between ARBs
    affinity_threshold_scalar: float
        Give a cut-off value for cell replacement
    k: int
        The number of memory cells to use for classification
    test_size: float
        The percentage of global data to take as test data
    """

    def __init__(self,
                 hyper_clonal_rate,
                 clonal_rate,
                 class_number,
                 mc_init_rate,
                 total_num_resources,
                 affinity_threshold_scalar,
                 k,
                 test_size,
                 create_test_train_set=True,
                 train_set=None,
                 test_set=None,
                 data=None,
                 class_col='Class'):
        
        self.HYPER_CLONAL_RATE = hyper_clonal_rate
        self.CLONAL_RATE = clonal_rate
        self.AFFINITY_THRESHOLD = 0
        self.CLASS_NUMBER = class_number
        self.MC_INIT_RATE = mc_init_rate
        self.TOTAL_NUM_RESOURCES = total_num_resources
        self.AFFINITY_THRESHOLD_SCALAR = affinity_threshold_scalar
        self.TEST_SIZE = test_size
        self.K = k
        self.MC = None
        self.AB = None
        self.data = data
        self.class_col = class_col
        self.cols_features = self.data.columns.difference([class_col])
        self.cols_attr = ['stimulation', 'ARB', class_col]
        self.n_features = len(self.cols_features)

        if create_test_train_set:
            self.train_set, self.test_set = self._train_test_split()
        else:
            self.train_set = train_set
            self.test_set = test_set
        
        #self.train_set = self.train_set.append([self.train_set[self.train_set.loc[:, 'Class']==1]], ignore_index=True)


    @staticmethod
    def affinity(vector1, vector2, brute_force=False):
        """
        Compute the affinity (Normalized!! distance) between two features
        vectors.
        
        Parameters
        --------------
        vector1: list
            First features vector
        vector2: list
            Second features vector
        
        Returns
        --------------
            The affinity between the two vectors [0-1]
        """

        if brute_force:
            import math
            euclidian_distance = 0
            d = 0

            for i, j in zip(vector1, vector2):
                d += (i - j)**2
            euclidian_distance = math.sqrt(d)

            return euclidian_distance / (1 + euclidian_distance)
        else:
#            dist = np.sqrt((np.square(vector1 - vector2).sum()))
            #vector1 = np.array(vector1)
            #vector2 = np.array(vector2)
            #assert type(vector1) in (np.ndarray_int, np.ndarray_float), type(vector1)
            #assert type(vector2) in (np.ndarray_int, np.ndarray_float), type(vector2)
            #print(vector1)
            #print(vector2)
            #dist = (vector1 - vector2)**2.0
           # dist = (np.square(vector1 - vector2).sum())**0.5
            #print('vectors')
            #print(vector1.shape)
            #print(np.array(vector2))
            #dist = pdist(ta.array((vector1, np.array(vector2))))
            #dist = pdist(vector1, vector2)
            #dist = sum([(x-y)**2.0 for x, y in zip(vector1, vector2)])**0.5
            dist = np.linalg.norm(vector1 - vector2)
            
            return dist/(1 + dist)


    def _stimulate_apply(self, cell, pattern):
        """
        The higher the affinity, the lower the stimulation.

        Parameters
        ----------
        pattern : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        cell['stimulation'] = 1 - self.affinity(vector1=pattern, vector2=cell[self.cols_features].to_numpy())
        
        return cell['stimulation']

    def _stimulate_val(self, cell, pattern):
        """
        The higher the affinity, the lower the stimulation.

        Parameters
        ----------
        pattern : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
                
        return 1 - self.affinity(vector1=cell[self.cols_features].to_numpy(), vector2=pattern)

    def _mutate(self, cell):
        """
        Mutate the cell. All features will be randomly mutated, at the chance
        of MUTATION_RATE.

        Parameters
        ----------
        vector : TYPE
            DESCRIPTION.
        _class : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        mutated = False
        # Assign random values controlling the mutation cases.
        mutate_array = np.random.rand(1, self.n_features)

        # Array denoting mutation cases
        decision_mask = np.random.rand(1, self.n_features) <= MUTATION_RATE

        n_mutated = np.where(decision_mask, 1, 0).sum()
        
        if cell['ARB'] == 1:
            cell[self.cols_features][decision_mask[0]] = (7 * np.random.rand(1, n_mutated) + 0.1)[0]
        else:
            cell[self.cols_features][decision_mask[0]] = np.random.rand(1, n_mutated)[0]

        if n_mutated > 0:
            mutated = True
        else:
            mutated = False

        return cell, mutated
    
    


    def _train_test_split(self, open_file=True):
        
#=============================================================================
        df = self.data

        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=42)
        
        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)
# =============================================================================

        # cutoff = self.data['Time'].max()*0.7
        # train_set = self.data[self.data['Time'] < cutoff]
        # test_set = self.data[self.data['Time'] >= cutoff]
        
        return train_set, test_set


    def calculate_affinity_threshold(self, use_brute_force=False):
        """
        Calculates euclidian distance between the datapoints.
        Since we can have a large nr. of features and a very large nr. of 
        rows, this is very computationally expensive and should be optimized.
        
        Perhaps use Cython.
        """
        print("Now calculating affinity threshold..")
        if use_brute_force:
            affinity_threshold = 0
            print('Calculating affinity threshold..')
            for i in range(len(self.train_set)):
                for j in range(i + 1, len(self.train_set)):
                    affinity_threshold += self.affinity(self.train_set.iloc[i, :-1], self.train_set.iloc[j, :-1])
                    
            self.AFFINITY_THRESHOLD = affinity_threshold / (len(self.train_set) * (len(self.train_set) - 1) / 2)
        else:
            
            # Affinity threshold must be calculated from a normalized
            # dataframes.
            x = self.train_set.iloc[:, :-1].values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)

            self.AFFINITY_THRESHOLD = np.mean(pdist(df.values))
            
#            self.AFFINITY_THRESHOLD = np.mean(pdist(self.train_set.iloc[:, :-1].values))
            
            print(f"Affinity threshold found as {self.AFFINITY_THRESHOLD}!")


    def init_MC(self, MC):
        """ 
        Initialize the memory set pool. Appended as rows into the empty
        dataframes in MC dict. Two dataframes are seperately composed of fraud
        and nonfraud seed cells to be used as a basis for the subsequent
        calculations.
        
        Parameters
        -----------------
        train_set: dataframe
            The training set
        MC:
            The memory set pool
            
        Returns
        -----------------
        None
        """

        print("Initiating memory set pool..")

        seed_cells = self.train_set.sample(frac=self.MC_INIT_RATE)
        seed_cells['stimulation'] = float('-inf')
        seed_cells['ARB'] = 0
        seed_cells = seed_cells.reindex(columns=seed_cells.columns.difference(['stimulation', self.class_col]).tolist() + ['stimulation', self.class_col])

        fraud_mask = seed_cells[self.class_col] == 1

        # Append seed cells to the correct class
        MC[0] = MC[0].append(seed_cells[~fraud_mask])
        MC[1] = MC[1].append(seed_cells[fraud_mask])

# =============================================================================
#         for _ in range(int(self.train_set.shape[0] * self.MC_INIT_RATE)):
#             # Choose a random row from the train set. Same value can 
#             # chosen multiple times in this loop.
#             seed_cell = random.choice(self.train_set.values)
#             
#             # Create a memory cell and append it to the MC dictionary in the 
#             # Correct class list., i.e., sort input memory cells into fraud
#             # and nonfraud cells.
#             
#             # seed_cell[-1] is the class. Append Cell instance with all
#             # features.
# 
#             MC[int(seed_cell[-1])].append(Cell(vector=seed_cell[0:-1],
#                                                _class=seed_cell[-1]))
# =============================================================================

    def argminARB(self, AB, _class):
        """Get the ARB with the minimum amount of resources
        :param AB: The Artificial Recognition Balls set
        :param _class: the class of the ARBs
        :return: The ARB with the lower amount of resources and its index
        """
        minRes = 1.0
        ab = None
        abIndex = None
        
        for i in range(len(AB[_class])):
            if AB[_class].iloc[i, :].resources <= minRes:
                minRes = AB[_class].iloc[i, :].resources
                ab = AB[_class].iloc[i, :]
                abIndex = i

        return ab, abIndex

    def getMcCandidate(self, AB, _class):
        """Get the higher stimulation ARB to be (eventually) added to the memory cells pool
        :param AB: The Artificial Recognition Balls set
        :param _class: the class of the ARBs
        :return: Higher stimulation ARB of the given class
        """
        #maxStim = 0.0
        #ab = None

        maxStim = AB[_class].stimulation.max()
        #maxStim_index = np.where(AB[_class].stimulation == [maxStim])[0]
        #print(maxStim_index)
        #ab = AB[_class].iloc[maxStim_index, :]
        ab = AB[_class][AB[_class]['stimulation'] == maxStim]

# =============================================================================
#         for i in range(len(AB[_class])):
#             if AB[_class][i].stimulation >= maxStim:
#                 maxStim = AB[_class][i].stimulation
#                 ab = AB[_class][i]
# =============================================================================
        
        #c = Cell(vector=ab.vector, _class=ab._class)
        #c.stimulation = ab.stimulation
        
        return ab

    # def classify(self, antigene):
    #     if (self.MC is None) or (self.AB is None):
    #         raise Exception("AIRS must be trained first")

    #     vote_array = []
    #     for c in self.MC.keys():
    #         for index, ab in self.MC.get(c).iterrows():
    #             #self._stimulate_apply(cell=ab,
    #             #                pattern=antigene)

    #             ab['stimulation'] = self._stimulate_val(cell=ab,
    #                             pattern=antigene)

    #             vote_array.append(ab)

    #     vote_array = list(sorted(vote_array, key=lambda cell: -cell.stimulation))
    #     v = {0: 0, 1: 0}
    #     self.K = min(self.K, len(vote_array))
    #     for x in vote_array[:self.K]:
    #         v[x._class] += 1

    #     maxVote = 0
    #     _class = 0
    #     for x in v.keys():
    #         if v[x] > maxVote:
    #             maxVote = v[x]
    #             _class = x
    #     return _class

    def classify(self, antigene):
        
        df_vote = pd.DataFrame({})

        for c in self.MC.keys():
            df_vote_c = self.MC.get(c).copy()
            df_vote_c['stimulation'] = df_vote_c.apply(self._stimulate_val, pattern=antigene, axis=1)

            df_vote = df_vote.append(df_vote_c)

        df_vote.sort_values(by='stimulation', ascending=False)
        
        self.K = min(self.K, df_vote.shape[0])

        df_allowed_voters = df_vote.iloc[:self.K, :].copy()
        df_allowed_voters['class_count'] = df_allowed_voters.groupby('Class')['stimulation'].transform('count')

        df_result = df_allowed_voters.drop_duplicates('Class')
        df_result = df_result[df_result['class_count'].max() == df_result['class_count']]

        if df_result.shape[0]:
            return df_result['Class']
        else:
            return np.nan


    def train(self):
        """Train AIRS on the training dataset"""
        start = time.time()

        # Calculate the affinity threshold of the memory cells
        self.calculate_affinity_threshold()
        # The actual affinity threshold is modified by the input value,
        # AFFINITY_THRESHOLD_SCALAR
        self.AFFINITY_THRESHOLD_SCALED = self.AFFINITY_THRESHOLD * self.AFFINITY_THRESHOLD_SCALAR

        # Create two dicts for the Memory Cells and Antibodies, with two
        # empty class slots, nonfraud and fraud, labelled 0 and 1.
        MC = {_class_int: pd.DataFrame() for _class_int in range(self.CLASS_NUMBER)}
        AB = {_class_int: pd.DataFrame() for _class_int in range(self.CLASS_NUMBER)}

        # MC Initialisation
        self.init_MC(MC)

        #for row in self.train_set:
        self.train_set.reset_index(drop=True, inplace=True)
        size = self.train_set.shape[0]
        for index, row in self.train_set.iterrows():
            print(f'Training row {index} out of {size}')
            # Split into featureset (antigene) and target (class)
            antigene, _class = np.array(row[:-1]), int(row[-1])
            # MC Identification
            mc_match = None
            
            if len(MC[_class]) == 0:
                # If this is the first row in dataset
               # mc_match = Cell(vector=antigene, _class=_class)
                mc_match = row
                mc_match['stimulation'] = float('-inf')
                mc_match['ARB'] = 0
                MC[_class] = MC[_class].append(mc_match)
            else:
                best_stim = 0
                mc_match = row

                #print(pd.DataFrame(MC[_class]))
                #for c in MC[_class]:
                    #print(c)
                    #print(c.stimulation)
                #print(MC)
                #mc_match = MC[MC[_class].stimulation == MC[_class].stimulation.max()][_class]
                #best_stim = mc_match.stimulation


                    #print(best_stim)
                    #if c.stimulation >= best_stim:
                max_stim = MC[_class].stimulation.max()
                if max_stim > best_stim:
                    best_stim = max_stim
                    mc_match = MC[_class][MC[_class].stimulation == max_stim]

                    if mc_match.shape[0] > 1:
                        mc_match = mc_match.iloc[0, :]

# =============================================================================
#                 for c in MC[_class]:
#                     if c.stimulation >= best_stim:
#                         best_stim = c.stimulation
#                         mc_match = c
# =============================================================================

            # ARB Generation
#            AB[_class].append(ARB(vector=mc_match.vector, _class=mc_match._class))  # add the mc_match to ARBs
            mc_match['ARB'] = 1
            mc_match['stimulation'] = self._stimulate_val(cell=mc_match,
                                                            pattern=antigene)
            AB[_class] = AB[_class].append(mc_match)  # add the mc_match to ARBs

            #stim = self._stimulate_apply(cell=mc_match,
            #                       pattern=antigene)
            stim = mc_match['stimulation']
            iterations = 0
            while True:

            # =================================================================
            #              INITIATE CLONING
            # =================================================================

                iterations += 1
                MAX_CLONES = int(self.HYPER_CLONAL_RATE * self.CLONAL_RATE *stim)
                num_clones = 0

                while num_clones < MAX_CLONES:
                    clone, mutated = self._mutate(cell=mc_match)

                    if mutated:
                        AB[_class] = AB[_class].append(clone)
                        num_clones += 1

                # =============================================================
                #                 Competition for resources
                # =============================================================
                if len(MC[_class]) == 0:
                    avgStim = 0
                    min_stim_ab = 0
                    max_stim_ab = 0
                else:
                    AB[_class]['stimulation'] = AB[_class].apply(self._stimulate_val, pattern=antigene, axis=1)
                    avgStim = AB[_class]['stimulation'].sum() / AB[_class].shape[0]

                    try:
                        stim_values_max = [AB[_class_num].stimulation.max() for _class_num in AB.keys() if 'stimulation' in AB[_class_num].columns]
                        stim_values_min = [AB[_class_num].stimulation.min() for _class_num in AB.keys() if 'stimulation' in AB[_class_num].columns]
                            
                        max_stim_ab = max(stim_values_max)
                        min_stim_ab = min(stim_values_min)
                    except AttributeError:
                        print('Stimulation not determined')
                        min_stim_ab = 0
                        max_stim_ab = 0                        #max_stim_ab = AB[0].stimulation.max()
                        #min_stim_ab = AB[0].stimulation.min()

              #  avgStim = sum([self._stimulate(cell=x,
              #                                 pattern=antigene) for x in AB[_class]]) / AB[_class].shape[0]

                MIN_STIM = 1.0
                MAX_STIM = 0.0

                # BUG! MIN_STEM and MAX_STEM can appear as same value at end of loop.
                # Error appears if stim is 0<stim<1.
                # This must be an unexpected value, indicating a possible error.

# =============================================================================
#                 for _class in AB.keys():
#                     for index, ab in self.AB[_class].iterrows():                  
# 
#                         stim = self._stimulate(cell=ab,
#                                                pattern=antigene)
#                         if stim < MIN_STIM:
#                             MIN_STIM = stim
#                         if stim > MAX_STIM:
#                             MAX_STIM = stim
# =============================================================================

                MIN_STIM = min([MIN_STIM, min_stim_ab])
                MAX_STIM = min([MAX_STIM, max_stim_ab])

                if MIN_STIM == MAX_STIM:
                 #   print(f"OBS! 0<stim({stim})<1, keeping MIN_STEM and MAX_STEM at default values")
                    MIN_STIM = 1.0
                    MAX_STIM = 0.0                    

# =============================================================================
#                 for _class in AB.keys():
#                     for index, ab in self.AB[_class].iterrows():   
#                         ab.stimulation = (ab.stimulation - MIN_STIM) / (MAX_STIM - MIN_STIM)
#                         ab.resources = ab.stimulation * self.CLONAL_RATE
# =============================================================================
                if len(MC[_class]) != 0:

                    for c in AB.keys():
                        
                        if len(AB[c].index) > 0:

                            AB[c]['stimulation'] = (AB[c]['stimulation'] - MIN_STIM) / (MAX_STIM - MIN_STIM)
                            AB[c]['resources'] = AB[c]['stimulation'] * self.CLONAL_RATE

                #print([x.resources for x in AB[_class]])
                    resAlloc = AB[_class].resources.sum()
                else:
                    resAlloc = 0

                #resAlloc = sum([x.resources for x in AB[_class]])
                #resAlloc = [*map(np.sum, [[x.resources for x in AB[_class]]])][0]

                numResAllowed = self.TOTAL_NUM_RESOURCES
                while resAlloc > numResAllowed:
                    numResRemove = resAlloc - numResAllowed
                    abRemove, abRemoveIndex = self.argminARB(AB=AB, _class=_class)
                    
                    if abRemove.resources <= numResRemove:
                        AB[_class].reset_index(drop=True, inplace=True)
                        AB[_class] = AB[_class].drop(index=abRemoveIndex)
                        resAlloc -= abRemove.resources
                    else:
                        AB[_class].iloc[abRemoveIndex, :].resources -= numResRemove
                        resAlloc -= numResRemove
               # print(f"avgStim is {avgStim}")
                if (avgStim > self.AFFINITY_THRESHOLD) or (iterations >= MAX_ITER):
                    break

            mc_candidate = self.getMcCandidate(AB=AB, _class=_class)

            if mc_candidate['stimulation'].get_values()[0] > float(mc_match.stimulation):
                # if 'Class' in mc_candidate.columns:    
                #     candidate_antigene = np.array(mc_candidate.drop('Class', axis=1))
                # else:
                #     candidate_antigene = np.array(mc_candidate)
                # if 'Class' in mc_match.columns:    
                #     mc_match_antigene = np.array(mc_match.drop('Class', axis=1))
                # else:
                #     mc_match_antigene = np.array(mc_match)#.drop('Class'))
                
                candidate_antigene = np.array(mc_candidate)
                mc_match_antigene = np.array(mc_match)#.drop('Class'))

#                if cy_affinity(vector1=candidate_antigene[0], vector2=mc_match_antigene[0]) < self.AFFINITY_THRESHOLD_SCALED:
                #if self.affinity(vector1=candidate_antigene[0], vector2=mc_match_antigene[0]) < self.AFFINITY_THRESHOLD_SCALED:
                    # The mc candidate replaces the mc match
                    #MC[_class] = MC[_class][not MC[_class].Time.equals(mc_match.Time)]
                # Add the mc_match to MC pool
                MC[_class] = MC[_class].append(mc_candidate)

        self.MC = MC
        self.AB = AB

        n_correct = 0
        
        df_pred = pd.DataFrame({})

        df_pred['y_pred'] = [x.values[0] for x in [self.classify(np.array(x)) for x in self.test_set.iloc[:, :-1].values]]

        #print(*map(np.array, self.test_set.iloc[:, :-1].values))
        #pool_obj = multiprocessing.Pool(processes=8)

#        df_pred['y_pred'] = [*map(self.classify, [*map(np.array, self.test_set.iloc[:, :-1].values)])]
        #df_pred['y_pred'] = pool_obj.map(self.classify, pool_obj.map(np.array, self.test_set.iloc[:, :-1]))
#        df_pred['y_pred'] = pool_obj.map(self.classify, pool_obj.map(np.array, self.test_set.iloc[:, :-1].values))
        print(df_pred['y_pred'].shape)
        print(self.test_set.iloc[:, -1].shape)
        df_pred['y_true'] = self.test_set.iloc[:, -1].values
        #n_correct = [*map(np.sum, [df_pred.y_pred == df_pred.y_true])][0]
        n_correct = np.sum([df_pred.y_pred == df_pred.y_true])

        #for row in self.test_set.values:
        #    ag = row[:-1]
        #    _class = row[-1]
        #    if self.classify(ag) == reverseMapping[_class]:
        #        n_correct += 1
        #for ag, _class in (self.test_set.iloc[:, :-1], self.test_set.iloc[:, -1]):

        print("Execution time : {:2.4f} seconds".format(time.time() - start))
        print("Accuracy : {:2.2f} %".format(n_correct * 100 / self.test_set.shape[0]))
        print(f"Nonfraud fract : {sum(self.test_set.iloc[:, -1] == 0) / self.test_set.shape[0] * 100:.2f} %")
        print(f"Fraud fract : {sum(self.test_set.iloc[:, -1] == 1) / self.test_set.shape[0] * 100:.2f} %")
        print(f"Confusion matrix is {confusion_matrix(df_pred.y_true, df_pred.y_pred)}")

        return self.MC, self.train_set, self.test_set


if __name__ == '__main__':

    # =========================================================================
    # USE-CASE CREDIT CARD FRAUD
    # =========================================================================
    import matplotlib.pyplot as plt
    ARRAY_SIZE = 30  # Features number
    MAX_ITER = 5  # Max iterations to stop training on a given antigene

    # Mutation rate for ARBs
    MUTATION_RATE = 0.2

    n = 284808
    s = 160000 #desired sample size
    skip = sorted(random.sample(range(1,n+1),n-s)) #

    data = pd.read_csv('data/creditcard.csv', skiprows=skip)
    data_1 = data[data['Class']==1].copy()
    
    data = pd.concat([data.iloc[:1000, :], data_1], axis=0)
    plt.figure()
    data.Class.hist(bins=50)
    plt.show()

    # Very low nr of fraud cases, upsample cases.

    airs = AIRS(hyper_clonal_rate=20,
                clonal_rate=0.8,
                class_number=2,
                mc_init_rate=0.4,
                total_num_resources=10,
                affinity_threshold_scalar=0.8,
                k=7,
                test_size=0.3,
                data=data)
                #input_data_file='data/creditcard.csv')

    mc, df_train, df_test = airs.train()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3)