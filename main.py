"""
Implementation of AIRS2 (Artificial Immune Recognition System V2), first
applied to the IRIS data set by: Azzoug Aghiles

Forked and modified by Steffen Kjær Jacobsen to handle general dataframe
inputs, not only IRUS dataset.

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


Glossary:
    - Affinity is the distance between two cells. In code, it is the 
    (Normalized!! distance) between two features vectors.
    - Affinity threshold of the memory cells - this is this treshold where the
    immune system is activated.

    MC: Memory Cells
    AB: Antibodies


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
from func import cy_affinity
import multiprocessing

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
        self.stimulation = float('inf')
        self.resources = 0

    def __str__(self):
        return "ARB : Vector = {} | class = {} | stim = {} | res = {}".format(self.vector, self._class,
                                                                              self.stimulation, self.resources)

    def __repr__(self):
        return "ARB : Vector = {} | class = {} | stim = {} | res = {}".format(self.vector, self._class,
                                                                              self.stimulation, self.resources)

    def stimulate(self, pattern):
        """
        Stimulation of the ARB is inversely proportional with the affinity of
        the cell. Affinity ranges from [0;1] so stimulation ranges from 
        [1;0] due to start value of stim of 1.
        
        Parameters
        ----------
        pattern : TYPE
            DESCRIPTION.

        Returns
        -------
        self.stimulation: float
            Stimulation value of the cell.

        """
        
        self.stimulation = 1 - cy_affinity(vector1=pattern,
                                             vector2=self.vector)
        return self.stimulation

    def _mutate(self):
        """
        For each element in vector, assign a random number in the range
        [0.1, 7.1], if the random 'change' value is below MUTATION_RATE.
        
        Returns
        -------
        ARB: ARB instance
            A new ARB instance with a new vector with either the same
            value in element, or new mutated value.
        mutated: bool
            Informs us if any mutated vector elements were returned.
        """

        # _range = 1 - self.stimulation
        mutated = False
        new_vector = []

        for v in self.vector:
            change = random.random()
            change_to = 7 * random.random() + 0.1

            if change <= AIRS.MUTATION_RATE:
                new_vector.append(change_to)
                mutated = True
            else:
                new_vector.append(v)

        return ARB(vector=new_vector, _class=self._class), mutated


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
        self.stimulation = float('inf')

    def __str__(self):
        return "Cell : Vector = {} | class = {} | stim = {}".format(self.vector, self._class, self.stimulation)

    def __repr__(self):
        return "Cell : Vector = {} | class = {} | stim = {}".format(self.vector, self._class, self.stimulation)

    def stimulate(self, pattern):
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
        
        self.stimulation = 1 - cy_affinity(vector1=pattern, vector2=self.vector)
        return self.stimulation

    def _mutate(self):
        #_range = 1 - self.stimulation
        mutated = False
        new_vector = []

        for v in self.vector:
            change = random.random()
            change_to = random.random()

            if change <= MUTATION_RATE:
                new_vector.append(change_to)
                mutated = True
            else:
                new_vector.append(v)

        return ARB(vector=new_vector, _class=self._class), mutated

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
                 data=None):
        
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

        if create_test_train_set:
            self.train_set, self.test_set = self._train_test_split()
        else:
            self.train_set = train_set
            self.test_set = test_set
        
        self.train_set = self.train_set.append([self.train_set[self.train_set.loc[:, 'Class']==1]]*10, ignore_index=True)


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
            
            dist = pdist(ta.array((vector1, vector2)))
            #dist = sum([(x-y)**2.0 for x, y in zip(vector1, vector2)])**0.5
           # dist = np.linalg.norm(vector1 - vector2)
            
            return dist/(1 + dist)
    
    
    def _train_test_split(self, open_file=True):
        
# =============================================================================
#         df = self.data
# 
#         y = df.iloc[:, -1]
#         X = df.iloc[:, :-1]
#         
#         X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                             test_size=0.33,
#                                                             random_state=42)
#         
#         train_set = pd.concat([X_train, y_train], axis=1)
#         test_set = pd.concat([X_test, y_test], axis=1)
# =============================================================================

        cutoff = self.data['Time'].max()*0.7
        train_set = self.data[self.data['Time'] < cutoff]
        test_set = self.data[self.data['Time'] >= cutoff]
        print(train_set.Class.value_counts())
        print(train_set.shape)
        print(test_set.shape)
        
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
        Initialize the memory set pool
        
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

        for _ in range(int(self.train_set.shape[0] * self.MC_INIT_RATE)):
            # Choose a random row from the train set. Same value can 
            # chosen multiple times in this loop.
            seed_cell = random.choice(self.train_set.values)
            
            # Create a memory cell and append it to the MC dictionary in the 
            # Correct class list., i.e., sort input memory cells into fraud
            # and nonfraud cells.
            
            # seed_cell[-1] is the class. Append Cell instance with all
            # features.

            MC[int(seed_cell[-1])].append(Cell(vector=seed_cell[0:-1],
                                               _class=seed_cell[-1]))

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
            if AB[_class][i].resources <= minRes:
                minRes = AB[_class][i].resources
                ab = AB[_class][i]
                abIndex = i

        return ab, abIndex

    def getMcCandidate(self, AB, _class):
        """Get the higher stimulation ARB to be (eventually) added to the memory cells pool
        :param AB: The Artificial Recognition Balls set
        :param _class: the class of the ARBs
        :return: Higher stimulation ARB of the given class
        """
        maxStim = 0.0
        ab = None
#        print("AB", AB)
#        print("AB class", AB[_class])
        
#        print("len of AB class", len(AB[_class]))
        
        for i in range(len(AB[_class])):
            if AB[_class][i].stimulation >= maxStim:
                maxStim = AB[_class][i].stimulation
                ab = AB[_class][i]
        
        c = Cell(vector=ab.vector, _class=ab._class)
        c.stimulation = ab.stimulation
        return c

    def classify(self, antigene):
        if (self.MC is None) or (self.AB is None):
            raise Exception("AIRS must be trained first")

        vote_array = []
        for c in self.MC.keys():
            for ab in self.MC.get(c):
                ab.stimulate(antigene)
                vote_array.append(ab)

        vote_array = list(sorted(vote_array, key=lambda cell: -cell.stimulation))
        v = {0: 0, 1: 0}
        self.K = min(self.K, len(vote_array))
        for x in vote_array[:self.K]:
            v[x._class] += 1

        maxVote = 0
        _class = 0
        for x in v.keys():
            if v[x] > maxVote:
                maxVote = v[x]
                _class = x
        return _class

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
        MC = {_class: [] for _class in range(self.CLASS_NUMBER)}
        AB = {_class: [] for _class in range(self.CLASS_NUMBER)}

        # MC Initialisation
        self.init_MC(MC)

        for row in self.train_set.values:
            
            # Split into featureset (antigene) and target (class)
            antigene, _class = np.array(row[:-1]), row[-1]
            # MC Identification
            mc_match = None
            
            print('MC[_class]', len(MC[_class]))
            if len(MC[_class]) == 0:
                # If this is the first row in dataset
                mc_match = Cell(vector=antigene, _class=_class)
                MC[_class].append(mc_match)
            else:
                best_stim = 0
                #print(pd.DataFrame(MC[_class]))
                #for c in MC[_class]:
                    #print(c)
                    #print(c.stimulation)
                #print(MC)
                #mc_match = MC[MC[_class].stimulation == MC[_class].stimulation.max()][_class]
                #best_stim = mc_match.stimulation


                    #print(best_stim)
                    #if c.stimulation >= best_stim:
                
                for c in MC[_class]:
                    if c.stimulation >= best_stim:
                        best_stim = c.stimulation
                        mc_match = c

            # ARB Generation
            AB[_class].append(ARB(vector=mc_match.vector, _class=mc_match._class))  # add the mc_match to ARBs
            stim = mc_match.stimulate(np.array(antigene))

            iterations = 0
            while True:
                iterations += 1
                MAX_CLONES = int(self.HYPER_CLONAL_RATE * self.CLONAL_RATE * stim)
                num_clones = 0
                while num_clones < MAX_CLONES:
                    clone, mutated = mc_match._mutate()

                    if mutated:
                        AB[_class].append(clone)
                        num_clones += 1

                # Competition for resources
                print('2 MC[_class]', len(MC[_class]))
   
                avgStim = sum([x.stimulate(antigene) for x in AB[_class]]) / len(AB[_class])
             #   avgStim = [*map(np.sum, [[x.stimulate(antigene) for x in AB[_class]]])][0] / len(AB[_class])

                MIN_STIM = 1.0
                MAX_STIM = 0.0
                
                # BUG! MIN_STEM and MAX_STEM can appear as same value at end of loop.
                # Error appears if stim is 0<stim<1.
                # This must be an unexpected value, indicating a possible error.
                print('AB.keys()', AB.keys())
                for c in AB.keys():
                    for ab in AB.get(c):
                        stim = ab.stimulate(antigene)
                        if stim < MIN_STIM:
                            MIN_STIM = stim
                        if stim > MAX_STIM:
                            MAX_STIM = stim

                if MIN_STIM == MAX_STIM:
                 #   print(f"OBS! 0<stim({stim})<1, keeping MIN_STEM and MAX_STEM at default values")
                    MIN_STIM = 1.0
                    MAX_STIM = 0.0                    

                for c in AB.keys():
                    for ab in AB.get(c):
                        ab.stimulation = (ab.stimulation - MIN_STIM) / (MAX_STIM - MIN_STIM)
                        ab.resources = ab.stimulation * self.CLONAL_RATE
                
                #print([x.resources for x in AB[_class]])
                resAlloc = sum([x.resources for x in AB[_class]])
                #resAlloc = [*map(np.sum, [[x.resources for x in AB[_class]]])][0]
                numResAllowed = self.TOTAL_NUM_RESOURCES
                while resAlloc > numResAllowed:
                    numResRemove = resAlloc - numResAllowed
                    abRemove, abRemoveIndex = self.argminARB(AB=AB, _class=_class)
                    if abRemove.resources <= numResRemove:
                        AB[_class].remove(abRemove)
                        resAlloc -= abRemove.resources
                    else:
                        AB[_class][abRemoveIndex].resources -= numResRemove
                        resAlloc -= numResRemove
               # print(f"avgStim is {avgStim}")
                if (avgStim > self.AFFINITY_THRESHOLD) or (iterations >= MAX_ITER):
                    break

            mc_candidate = self.getMcCandidate(AB=AB, _class=_class)

            if mc_candidate.stimulation > mc_match.stimulation:
                if cy_affinity(mc_candidate.vector,
                                 mc_match.vector) < self.AFFINITY_THRESHOLD_SCALED:
                    # The mc candidate replaces the mc match
                    MC[_class].remove(mc_match)
                # Add the mc_match to MC pool
                MC[_class].append(mc_candidate)

        self.MC = MC
        self.AB = AB

        n_correct = 0
        
        df_pred = pd.DataFrame({})

#        df_pred['y_pred'] = [self.classify(np.array(x)) for x in self.test_set.iloc[:, :-1].values]
        #print(*map(np.array, self.test_set.iloc[:, :-1].values))
        pool_obj = multiprocessing.Pool(processes=8)

#        df_pred['y_pred'] = [*map(self.classify, [*map(np.array, self.test_set.iloc[:, :-1].values)])]
        df_pred['y_pred'] = pool_obj.map(self.classify, pool_obj.map(np.array, self.test_set.iloc[:, :-1].values))
        print(df_pred['y_pred'].shape)
        print(self.test_set.iloc[:, -1].shape)
        df_pred['y_true'] = self.test_set.iloc[:, -1].values
        n_correct = [*map(np.sum, [df_pred.y_pred == df_pred.y_true])][0]
        #n_correct = np.sum([df_pred.y_pred == df_pred.y_true])

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
        return n_correct / len(self.test_set)


if __name__ == '__main__':

    # =========================================================================
    # USE-CASE CREDIT CARD FRAUD
    # =========================================================================

    ARRAY_SIZE = 30  # Features number
    MAX_ITER = 5  # Max iterations to stop training on a given antigene

    # Mutation rate for ARBs
    MUTATION_RATE = 0.2

    n = 284808
    s = 1000 #desired sample size
    skip = sorted(random.sample(range(1,n+1),n-s)) #

    data = pd.read_csv('data/creditcard.csv', skiprows=skip)
    data.Time.hist(bins=50)
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
