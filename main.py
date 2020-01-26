"""
Implementation of AIRS2 (Artificial Immune Recognition System V2) applied to the IRIS data set
@author : Azzoug Aghiles
"""

import random
import time
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from scipy.spatial.distance import pdist
import numpy as np


class AIRS:
    """AIRS (Artificial Immune Recognition System) class
    Main class for this algorithm
    Params:
        hyper_clonal_rate (float) : Define the number of clones an ARB is allowed to produce
        clonal_rate (float) : Define the number of ressources an ARB can obtain
        class_number (int) : The number of classes (3 in this case)
        mc_init_rate (float) : Define the number of training data to be copied in memory cells
        total_num_resources (float) : The total numbers of resources to share between ARBs
        affinity_threshold_scalar  (float) : Give a cut-off value for cell replacement
        k (int) : The number of memory cells to use for classification
        test_size (float) : The percentage of global data to take as test data

    This implementation uses negative selection and clonal selection to imitate
    B-Cells in the human immune system.
    
    Optimizations:
        - Calculation of distance to all neighbors must be reordered into a 
        more effective knn algorithm, horribly slow at the moment.
    Glossary
    ---------------------------------------------------------------------------
    
    MC: Memory Cells
    
    """

    def __init__(self, hyper_clonal_rate, clonal_rate, class_number,
                 mc_init_rate, total_num_resources, affinity_threshold_scalar,
                 k, test_size, create_test_train_set=True, train_set=None,
                 test_set=None, input_data_file=None, data=None):
        
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
        self.input_data_file = input_data_file
        self.data = data

        if create_test_train_set:
            self.train_set, self.test_set = self.train_test_split()
        else:
            self.train_set = train_set
            self.test_set = test_set
        

    @staticmethod
    def affinity(vector1, vector2):
        """Compute the affinity (Normalized !! distance) between two features vectors
        :param vector1: First features vector
        :param vector2: Second features vector
        :return: The affinity between the two vectors [0-1]
        """
        
        #euclidian_distance = 0
        #d = 0
        
        dist = np.linalg.norm(vector1 - vector2)
        
        #for i, j in zip(vector1, vector2):
        #    d += (i - j) ** 2
        #euclidian_distance = math.sqrt(d)
        #return euclidian_distance / (1 + euclidian_distance)
        
        return dist/(1 + dist)
    
    
    def train_test_split(self, open_file=True):
        
        df = self.data #pd.read_csv(self.input_data_file)

        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.33,
                                                            random_state=42)

        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)

        return train_set[:20000], test_set[:6660]

       # if open_file:
       #     with open(self.input_data_file, "r") as data:
       #         content = data.readlines()
       #         ret = [([float(x.split(",")[i]) for i in range(4)],
       #                  mapping[x.split(",")[4][:-1]]) for x in content]
       #         random.shuffle(ret)
       #     return ret[:int((1 - self.TEST_SIZE) * len(ret))], ret[int((1 - self.TEST_SIZE) * len(ret)):]
        
       # else:
            
       #     y = self.data[:, -1]
       #     X = self.data[:, :-1]

       #     X_train, X_test, y_train, y_test = train_test_split(X, y,
       #                                                         test_size=0.33,
       #                                                         random_state=42)
            
       #     train_set = pd.concat(X_train, y_train, axis=1)
       #     test_set = pd.concat(X_test, y_test, axis=1)

       #     return train_set, test_set



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
            
            # Affinity threshold should be normalized
            x = self.train_set.iloc[:, :-1].values  # returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            df = pd.DataFrame(x_scaled)
 
            self.AFFINITY_THRESHOLD = np.mean(pdist(df.values))
            
#            self.AFFINITY_THRESHOLD = np.mean(pdist(self.train_set.iloc[:, :-1].values))
            
            print(f"Affinity threshold found as {self.AFFINITY_THRESHOLD}!")


    def init_MC(self, MC):
        """ Init the memory set pool
        :param train_set: the training set
        :param MC: The memory set pool
        """
        print("Initiating memory set pool..")
        for _ in range(int(len(self.train_set) * self.MC_INIT_RATE)):
            seed_cell = random.choice(self.train_set.values)
          #  print("seed", seed_cell)
          #  print("MC", MC)
          #  print("CELL", Cell(vector=seed_cell[0], _class=seed_cell[-1]))
            MC[int(seed_cell[-1])].append(Cell(vector=seed_cell[0], _class=seed_cell[-1]))

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

    def train(self):
        """Train AIRS on the training dataset"""
        start = time.time()

        # Calculate the affinity threshold of the memory cells
        self.calculate_affinity_threshold()
        MC = {_class: [] for _class in range(self.CLASS_NUMBER)}
        AB = {_class: [] for _class in range(self.CLASS_NUMBER)}

        # MC Initialisation
        self.init_MC(MC)
#        print(self.train_set.iloc[:, :-1].values)
        for row in self.train_set.values:
            antigene, _class = row[:-1], row[-1]
            # MC Identification
            mc_match = None
            if len(MC[_class]) == 0:
                mc_match = Cell(vector=antigene, _class=_class)
                MC[_class].append(mc_match)
            else:
                best_stim = 0
                for c in MC[_class]:
                    if c.stimulation >= best_stim:
                        best_stim = c.stimulation
                        mc_match = c

            # ARB Generation
            AB[_class].append(ARB(vector=mc_match.vector, _class=mc_match._class))  # add the mc_match to ARBs
            stim = mc_match.stimulate(antigene)

            iterations = 0
            while True:
                iterations += 1
                MAX_CLONES = int(self.HYPER_CLONAL_RATE * self.CLONAL_RATE * stim)
                num_clones = 0
                while num_clones < MAX_CLONES:
                    clone, mutated = mc_match.mutate()

                    if mutated:
                        AB[_class].append(clone)
                        num_clones += 1

                # Competition for resources
                avgStim = sum([x.stimulate(antigene) for x in AB[_class]]) / len(AB[_class])

                MIN_STIM = 1.0
                MAX_STIM = 0.0
                
                # BUG! MIN_STEM and MAX_STEM can appear as same value at end of loop.
                # Error appears if stim is 0<stim<1.
                # This must be an unexpected value, indicating a possible error.
                for c in AB.keys():
                    for ab in AB.get(c):
                        stim = ab.stimulate(antigene)
                        if stim < MIN_STIM:
                            MIN_STIM = stim
                        if stim > MAX_STIM:
                            MAX_STIM = stim

                if MIN_STIM == MAX_STIM:
                    print(f"OBS! 0<stim({stim})<1, keeping MIN_STEM and MAX_STEM at default values")
                    MIN_STIM = 1.0
                    MAX_STIM = 0.0                    

                for c in AB.keys():
                    for ab in AB.get(c):
                        ab.stimulation = (ab.stimulation - MIN_STIM) / (MAX_STIM - MIN_STIM)
                        ab.resources = ab.stimulation * self.CLONAL_RATE

                resAlloc = sum([x.resources for x in AB[_class]])
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
                if AIRS.affinity(mc_candidate.vector,
                                 mc_match.vector) < self.AFFINITY_THRESHOLD * self.AFFINITY_THRESHOLD_SCALAR:
                    # The mc candidate replaces the mc match
                    MC[_class].remove(mc_match)
                # Add the mc_match to MC pool
                MC[_class].append(mc_candidate)

        self.MC = MC
        self.AB = AB

        n_correct = 0
        
        for row in self.test_set.values:
            ag = row[:-1]
            _class = row[-1]
            if self.classify(ag) == reverseMapping[_class]:
                n_correct += 1
        #for ag, _class in (self.test_set.iloc[:, :-1], self.test_set.iloc[:, -1]):


        print("Execution time : {:2.4f} seconds".format(time.time() - start))
        print("Accuracy : {:2.2f} %".format(n_correct * 100 / len(self.test_set)))
        return n_correct / len(self.test_set)

    def classify(self, antigene):
        if (self.MC is None) or (self.AB is None):
            raise Exception("AIRS must be trained first")

        vote_array = []
        for c in self.MC.keys():
            for ab in self.MC.get(c):
                ab.stimulate(antigene)
                vote_array.append(ab)

        vote_array = list(sorted(vote_array, key=lambda cell: -cell.stimulation))
        v = {0: 0, 1: 0, 2: 0}
        self.K = min(self.K, len(vote_array))
        for x in vote_array[:self.K]:
            v[x._class] += 1

        maxVote = 0
        _class = 0
        for x in v.keys():
            if v[x] > maxVote:
                maxVote = v[x]
                _class = x
        return reverseMapping[_class]


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
            self.vector = vector
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
        
        self.stimulation = 1 - AIRS.affinity(vector1=pattern, vector2=self.vector)
        return self.stimulation

    def mutate(self):
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
        if vector is None:
            self.vector = [random.random() for _ in range(ARRAY_SIZE)]
        else:
            self.vector = vector
        self._class = _class
        self.stimulation = float('inf')

    def __str__(self):
        return "Cell : Vector = {} | class = {} | stim = {}".format(self.vector, self._class, self.stimulation)

    def __repr__(self):
        return "Cell : Vector = {} | class = {} | stim = {}".format(self.vector, self._class, self.stimulation)

    def stimulate(self, pattern):
        self.stimulation = 1 - AIRS.affinity(vector1=pattern, vector2=self.vector)
        return self.stimulation

    def mutate(self):
        _range = 1 - self.stimulation
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


if __name__ == '__main__':

    # =========================================================================
    # USE-CASE EXAMPLE ON IRIS DATASET
    # =========================================================================

    ARRAY_SIZE = 4  # Features number
    MAX_ITER = 5  # Max iterations to stop training on a given antigene

    # Mapping classes to integers
    mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    reverseMapping = {0: "Iris-setosa", 1: "Iris-versicolor",
                      2: "Iris-virginica"}

    # Mutation rate for ARBs
    MUTATION_RATE = 0.2
    
    data = pd.read_csv('data/creditcard.csv')
    
    # Very low nr of fraud cases, upsample cases.
    data = data.append([data[data.loc[:, 'Class']==1]]*100, ignore_index=True)
    
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
