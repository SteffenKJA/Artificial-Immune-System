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
    (Normalized!! distance) between two features vectors. Essentially, the higher the affinity, the lower the
    relation between the cells. 
    NOTE: Counterintuitive, as small affinity values (closer to 0) indicate strong affinity between antibody and antigen. 
    This is not an error, but the official definition in AIRS V2 paper.
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
    
    affinity_threshold_scalar: The scalar value which modifies the affinity
                               threshold.
    MUTATION_RATE: The fraction of input vector elements who will be assigned
    a new random value.
    
    mc_init_rate: Fraction of memory cells relative to the input training
                  set number of cells (number of rows).


TODO:
    - Unittest _affinity(), to ensure that it is always between 0 and 1.

@authors: Steffen Kjær Jacobsen and Azzoug Aghiles.
"""
# %%
import random
import sqlite3
import time

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, ClassifierMixin
from skopt import BayesSearchCV
# parameter ranges are specified by one of below
from skopt.space import Real, Integer

from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from sqlite_conn import sqlite_db

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#os.system('python setup.py build_ext --inplace')
#from func import cy_affinity
# Substitute AIRS.affinity with cy_affinity


class AIRS(BaseEstimator, ClassifierMixin):
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
        The number of classes (e.g, 2 for fraud detection)
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
                 class_number: int,
                 column_names: list,
                 hyper_clonal_rate: float=20,
                 clonal_rate: float=0.8,
                 max_iter: int=5,
                 mutation_rate: float=0.2,
                 mc_init_rate: float=0.4,
                 total_num_resources: float=10,
                 affinity_threshold_scalar: float=0.8,
                 k: int=3,
                 class_col: str='Class'):
        
        self.hyper_clonal_rate = hyper_clonal_rate
        self.clonal_rate = clonal_rate
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.affinity_threshold = 0
        self.class_number = class_number
        self.mc_init_rate = mc_init_rate
        self.total_num_resources = total_num_resources
        self.affinity_threshold_scalar = affinity_threshold_scalar
        self.k = k
        self.MC = None
        self.AB = None
        self.class_col = class_col
        self.cols_attr = ['stimulation', 'ARB', class_col]
        self.column_names = column_names
        self.n_features = len(self.column_names)

    def _affinity(self, vector1: np.array, vector2: np.array) -> float:
        """
        Compute the affinity (normalized distance) between two features
        vectors.
        Lower affinity values corresponds to stronger affinity bond (somewhat counterintuitively).
        
        Parameters
        --------------
        vector1:
            First features vector
        vector2:
            Second features vector
        
        Returns
        --------------
            The affinity scalar value between the two vectors [0-1]
        """
        
        # Distance, normalized to max equally to the unit vector in train set space
        dist = np.linalg.norm((vector1 - vector2)/self.train_set_max_vector_mag)
        
        # Scaled to 0 to 1 values.
        return dist


    def _stimulate(self, cell: pd.DataFrame, pattern: np.array) -> float:
        """
        The higher the affinity value, the lower the stimulation.

        Parameters
        ----------
        pattern : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        affinity_val = self._affinity(vector1=cell[self.column_names].to_numpy(), vector2=pattern)

        if pd.isna(affinity_val):
            affinity_val = 1

        return 1 - affinity_val

    def _mutate(self, cell: pd.DataFrame) -> Tuple:#[Union[pd.DataFrame, bool]]:
        """
        Mutate the cell. All features will be randomly mutated, at the chance
        of self.mutation_rate.

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
        _range = 1 - cell['stimulation'].values[0]
        mutated = False
        # Assign random values controlling the mutation cases.
        mutate_array = np.random.rand(1, self.n_features)

        # Array denoting mutation cases
        decision_mask = mutate_array <= self.mutation_rate

        n_mutated = np.where(decision_mask, 1, 0).sum()
        features_to_mutate = np.array(self.column_names)[decision_mask[0]]
        normalization_vals = self.feature_maxs[decision_mask[0]]
        change_to_vals = np.random.rand(1, n_mutated)
        mutated_values_bottom = np.array([val / normalization_val - _range/2.0 for val, normalization_val in zip(cell.loc[:, features_to_mutate].values[0], normalization_vals)])
        mutated_values_bottom = np.array([val if val >= 0 else 0 for val in mutated_values_bottom])

        change_to_vals = change_to_vals*_range + mutated_values_bottom 
        change_to = np.array([val if val <= 1 else 1 for val in change_to_vals[0]])

        if n_mutated > 0:
            mutated = True
            cell.loc[:, features_to_mutate] = change_to*normalization_vals
        else:
            mutated = False

        return cell, mutated


    def _calculate_affinity_threshold(self) -> None:
        """
        Calculates euclidian distance between the datapoints.
        Since we can have a large nr. of features and a very large nr. of 
        rows, this is very computationally expensive and should be optimized.
        """
        print("Now calculating affinity threshold..")
            
        # Affinity threshold must be calculated from a normalized
        # dataframe.
        #x = self.train_set.iloc[:, :-1].values  # returns a numpy array

        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(self.X)
        df = pd.DataFrame(x_scaled)

        # Normalize all pairwise distance values to the unit vector length
        # Note that both individual features as well as the total vector length has been normalized between 0 and 1. 
        self.affinity_threshold = np.mean(pdist(df.values, metric='euclidean')/np.sqrt(self.n_features))

        assert (0 <= self.affinity_threshold <= 1)
        print(f"Affinity threshold found as {self.affinity_threshold}!")


    def _init_MC(self, MC: Dict) -> None:
        """ 
        Initialize the memory set pool. Appended as rows into the empty
        dataframes in MC dict. Two dataframes are seperately composed of the
        two classes of seed cells to be used as a basis for the subsequent
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
        np_train_set = np.column_stack((self.X, self.y))
        train_set = pd.DataFrame(np_train_set, columns=self.column_names + [self.class_col])
        seed_cells = train_set.sample(frac=self.mc_init_rate)
        seed_cells['stimulation'] = 0#float('-inf')
        seed_cells['ARB'] = 0  # Denotes if the cell is an ARB or not.
        # TODO: verify that col order is not scrambled below
        seed_cells = seed_cells.reindex(columns=seed_cells.columns.difference(['stimulation', 'ARB', self.class_col]).tolist() + ['stimulation', 'ARB', self.class_col])

        for _class in range(self.class_number):
        
            class_mask = seed_cells[self.class_col] == _class

            # Append seed cells to the correct class
            MC[_class] = pd.concat([MC[_class], seed_cells[class_mask]].copy(), axis=0)


    def _min_ressource_arb(self, AB: Dict, _class: int) -> Tuple:#[Union[pd.Series, int]]:
        """Get the ARB with the minimum amount of resources
        :param AB: The Artificial Recognition Balls set
        :param _class: the class of the ARBs
        :return: The ARB with the lower amount of resources and its index
        """
        min_res = 1.0
        arb = None
        arb_index = None
        # TODO: Change to simple min selection in pandas 
        for i in range(len(AB[_class])):
            if AB[_class].iloc[i, :].resources <= min_res:
                min_res = AB[_class].iloc[i, :].resources
                arb = AB[_class].iloc[i, :]
                arb_index = i

        return arb, arb_index

    def _get_mc_candidate(self, AB: Dict, _class: int) -> pd.DataFrame:
        """Get the higher stimulation ARB to be (eventually) added to the memory cells pool
        :param AB: The Artificial Recognition Balls set
        :param _class: the class of the ARBs
        :return: Higher stimulation ARB of the given class
        """

        max_stim = AB[_class].stimulation.max()
        arb = AB[_class][AB[_class]['stimulation'] == max_stim]

        if arb.shape[0] > 1:
            arb = pd.DataFrame(arb.iloc[0, :]).T

        assert isinstance(arb, pd.DataFrame)
        
        return arb

    def _classify(self, antigene: pd.DataFrame) -> float:
        
        df_knn = self.MC_mass.drop(['ARB', 'stimulation', self.class_col, 'resources'], axis=1)
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(df_knn.to_numpy())
        distances, indices = nbrs.kneighbors([antigene])
        df_allowed_voters = self.MC_mass.iloc[indices[0], :].copy()
        df_allowed_voters['class_count'] = df_allowed_voters.groupby(self.class_col)['stimulation'].transform('count')
        df_result = df_allowed_voters.drop_duplicates(self.class_col)
        df_result = df_result[df_result['class_count'].max() == df_result['class_count']]

        if df_result.shape[0]:
            return df_result[self.class_col]
        else:
            return np.nan

    def fit(self, X: np.array, y: np.array) -> Tuple[pd.DataFrame]:
        """Train AIRS on the training dataset"""
        start = time.time()
        self.X = X
        self.y = y
        self.feature_mins = self.X.min(axis=0) # numpy notation axis 0 is columns
        self.feature_maxs = self.X.max(axis=0) 
        #self.feature_mins = self.train_set.drop([class_col], axis=1).min().to_numpy() 
        #self.feature_maxs = self.train_set.drop([class_col], axis=1).max().to_numpy() 
        self.feature_dimension_scales = self.feature_maxs - self.feature_mins
        self.train_set_max_vector_mag = np.linalg.norm(self.feature_dimension_scales)

        # Calculate the affinity threshold of the memory cells
        self._calculate_affinity_threshold()
        # The actual affinity threshold is modified by the input value,
        # affinity_threshold_scalar
        self.AFFINITY_THRESHOLD_SCALED = self.affinity_threshold * self.affinity_threshold_scalar

        # Create two dicts for the Memory Cells and Artificial Recognition Balls (ARB or AB), with two
        # empty class slots, nonfraud and fraud, labelled 0 and 1.
        MC = {_class_int: pd.DataFrame() for _class_int in range(self.class_number)}
        AB = {_class_int: pd.DataFrame() for _class_int in range(self.class_number)}
        self.remove_mc_cell_aff_threshold = {_class_int: 0 for _class_int in range(self.class_number)}

        # MC Initialisation
        self._init_MC(MC)
        idx = 1
        
        # Iterate over featureset (antigene) and target (class)
        for antigene, _class in zip(X, y):
            print(f'Training row {idx} out of {X.shape[0]}')
            
            # ----------------------------- MC Identification ---------------------------- #
            mc_match = None
            
            if len(MC[_class]) == 0:
                # If this is the first row in dataset
                mc_match = pd.DataFrame(antigene).T
                mc_match.columns = self.column_names
                mc_match['stimulation'] = 0.0 #float('-inf')
                mc_match['ARB'] = 0
                MC[_class] = pd.concat([MC[_class], mc_match.copy()], axis=0)
                MC[_class] = MC[_class].reset_index(drop=True)
            else:
                # Select a MC candidate.
                # NOTE: We choose the MC with the highest stimulation as a starting point.
                # This stimulation relates to the nearest mc to any previous antigene, not the closests
                # previous antigene to the current antigene, so we could start quite far away in parameter space.
                # The mutated clones will randomly fan out from this positions, but will tend to
                # be in the mass center, since it is a random antigene as reference point (or rather, the
                # mc closests to any of the reference antigene). This mass center tendency means that we effectively
                # remove outliers and concentrate our MCs of a class near the density center of the antigenes of a given class.

                # NOTE: Change algo here, appears to a bug from original repo. 
                # We should find the max stimulation to the current antigen, not use old stimulation values, relating to
                # past antigens.
                MC[_class]['stimulation'] = MC[_class].apply(self._stimulate, pattern=antigene, axis=1)  # NOTE: New algo line
                
                max_stim = MC[_class].stimulation.max()
                MC[_class] = MC[_class].reset_index(drop=True)
                mc_match = pd.DataFrame(MC[_class][MC[_class].stimulation == max_stim])

                if mc_match.shape[0] > 1:
                    mc_match = pd.DataFrame(mc_match.iloc[0, :]).T

            mc_match['ARB'] = 1
            # The stimulation between MC candidate and the incoming antigene.
            #mc_match['stimulation'] = self._stimulate(cell=mc_match, pattern=antigene)
            AB[_class] = pd.concat([AB[_class], mc_match.copy()], axis=0)  # add the mc_match to ARBs

            stim = mc_match['stimulation']
            iterations = 0

            while True:

            # =================================================================
            #              INITIATE CLONING
            # =================================================================

                iterations += 1
                MAX_CLONES = int(self.hyper_clonal_rate * self.clonal_rate * stim)
                num_clones = 0

                while num_clones < MAX_CLONES:
                    clone, mutated = self._mutate(cell=mc_match.copy())

                    if mutated:
                        AB[_class] = pd.concat([AB[_class], clone.copy()], axis=0)
                        num_clones += 1

                # =============================================================
                #                 Competition for resources
                # =============================================================
                if len(MC[_class]) == 0:
                    avgStim = 0
                    min_stim_ab = 0
                    max_stim_ab = 0
                else:
                    AB[_class]['stimulation'] = AB[_class].apply(self._stimulate, pattern=antigene, axis=1)
                    avgStim = AB[_class]['stimulation'].sum() / AB[_class].shape[0]

                    try:
                        stim_values_min = [AB[_class_num].stimulation.min() for _class_num in AB.keys() if 'stimulation' in AB[_class_num].columns]
                        stim_values_max = [AB[_class_num].stimulation.max() for _class_num in AB.keys() if 'stimulation' in AB[_class_num].columns]
                            
                        min_stim_ab = min(stim_values_min)
                        max_stim_ab = max(stim_values_max)
                    except AttributeError:
                        print('Stimulation not determined')
                        min_stim_ab = 0
                        max_stim_ab = 0

                MIN_STIM = 1.0
                MAX_STIM = 0.0

                # BUG! MIN_STEM and MAX_STEM can appear as same value at end of loop.
                # Will happen if only one AB of a single class exists.
                # Error appears if stim is 0<stim<1.
                # This must be an unexpected value, indicating a possible error.
                
                # NOTE: Odd default values. Stim of 1.0 equals zero distance (min affinity)
                # Appears to be set so we get negative stimulation if we do not have any ABs.    

                MIN_STIM = min([MIN_STIM, min_stim_ab])
                MAX_STIM = max([MAX_STIM, max_stim_ab])

                if (sum([AB[c].shape[0] for c in AB.keys()]) < 2) or (MIN_STIM==MAX_STIM):
                    MIN_STIM = 1.0
                    MAX_STIM = 0.0                    

                for c in AB.keys():
                    
                    if len(AB[c].index) > 0:

                        AB[c]['stimulation'] = (AB[c]['stimulation'] - MIN_STIM) / (MAX_STIM - MIN_STIM)
                        AB[c]['resources'] = AB[c]['stimulation'] * self.clonal_rate

                if AB[_class].shape[0] > 0:
                    resAlloc = AB[_class].resources.sum()
                else:
                    resAlloc = 0

                numResAllowed = self.total_num_resources
                while resAlloc > numResAllowed:
                    numResRemove = resAlloc - numResAllowed
                    abRemove, abRemoveIndex = self._min_ressource_arb(AB=AB, _class=_class)
                    AB[_class].reset_index(drop=True, inplace=True)
                    
                    if abRemove.resources <= numResRemove:
                        AB[_class] = AB[_class].drop(index=abRemoveIndex)
                        resAlloc -= abRemove.resources
                    else:
                        AB[_class].loc[AB[_class].index == abRemoveIndex, 'resources'] -= numResRemove
                        resAlloc -= numResRemove
                if (avgStim > self.affinity_threshold) or (iterations >= self.max_iter):
                    break

            mc_candidate = self._get_mc_candidate(AB=AB, _class=_class)

            # get_values()[0]
            if mc_candidate['stimulation'].iloc[0] > float(mc_match.stimulation):
                mc_candidate_pattern = mc_candidate[self.column_names].to_numpy() 
                mc_match_pattern = mc_match[self.column_names].to_numpy() 
                
                # If the mc_candidate and the mc_match (parent mc) are within a threshold distance of each other, remove
                # the mc_match, since mc_candidate in this case is closer to the antigene.
                if self._affinity(vector1=mc_candidate_pattern, vector2=mc_match_pattern) < self.affinity_threshold*self.affinity_threshold_scalar:
                    self.remove_mc_cell_aff_threshold[_class] += 1
                    
                    len_before_removal = MC[_class].shape[0] 
                    MC[_class] = MC[_class].drop([mc_match.index[0]])
                    len_after_removal = MC[_class].shape[0]
                    
                    assert len_after_removal < len_before_removal
                    # mc_class_compare = MC[_class][self.column_names].to_numpy()
                    # #mc_class_compare = mc_class_compare.reindex(sorted(mc_class_compare.columns), axis=1)

                    # mc_match_compare = mc_match[self.column_names].to_numpy()
                    # #mc_match_compare = mc_match_compare.reindex(sorted(mc_match_compare.columns), axis=1)

                    # mc_match_index = np.unique(np.where(mc_class_compare==mc_match_compare)[0]).tolist()
                    
                    # # need to drop mc match here
                    # # MC[_class].drop(np.where(MC[_class].iloc(4) == mc_match.iloc[4]), axis=0)
                    # MC[_class] = MC[_class].reset_index(drop=True).drop(mc_match_index, axis=0)
                mc_candidate.index += MC[_class].index.max() + 1
                MC[_class] = pd.concat([MC[_class], mc_candidate.copy()], axis=0)

            idx += 1

        self.MC = MC
        self.AB = AB

        # Define the MC space of containing MCs of all classes

        self.MC_mass = pd.concat([MC.get(c) for c in MC.keys()], axis=0).reset_index(drop=True)
        self.execution_time = time.time() - start
        
        # Use other classifier
        self.external_classifier = LogisticRegression()

        if 'resources' in self.MC_mass.columns:
            drop_cols = [self.class_col, 'ARB', 'stimulation', 'resources']
        else:
            drop_cols = [self.class_col, 'ARB', 'stimulation']
        self.MC_mass_train = self.MC_mass.drop(drop_cols, axis=1)
        self.MC_mass_train = self.MC_mass_train.fillna(self.MC_mass_train.mean())
        self.external_classifier.fit(self.MC_mass_train, self.MC_mass[self.class_col])
        
        print("Execution time : {:2.4f} seconds".format(self.execution_time))

        print("================")
        print("DATA REDUCTION")
        print("----------------")
        for _class in range(self.class_number):
            print(f"Class {_class}: Train rows of {sum(y==_class)} reduced to MC of {self.MC[_class].shape[0]} rows")
            print(f"Class {_class}: {self.MC[_class].ARB.sum()} mutated cells added to MC")
            print(f"Class {_class}: {self.remove_mc_cell_aff_threshold[_class]} removed cells from MC due to crossed affinity threshold")
        print("================")
        
    def get_evaluation(self, y_true: np.array):
        
        #self.df_pred['y_true'] = y_true  # self.test_set.iloc[:, -1].values
        self.n_correct = np.sum(self.df_pred.y_pred.to_numpy() == pd.Series(y_true))
        
        print("Accuracy : {:2.2f} %".format(self.n_correct * 100 / len(y_true)))
        print(f"Confusion matrix is {confusion_matrix(y_true, self.df_pred.y_pred)}")
        #print(f"Reference confusion matrix is {confusion_matrix(y_true, self.df_pred.y_pred_ref)}")
        print(f"KNN confusion matrix is {confusion_matrix(y_true, self.df_pred.y_pred_knn)}")

        print("Classification report")
        print(classification_report(y_true, self.df_pred.y_pred))  

    def predict(self, X: np.array):
        
        self.n_correct = 0
        self.df_pred = pd.DataFrame({})

        self.df_pred['y_pred_knn'] = [cell.values[0] for cell in [self._classify(np.array(cell)) for cell in X]]
    
        preds = self.external_classifier.predict(X)
        self.df_pred['y_pred'] = preds 
        
        # # Train reference model
        # logisticRegr = LogisticRegression()
        # logisticRegr.fit(self.train_set.drop([self.class_col], axis=1), self.train_set[self.class_col])
        # self.df_pred['y_pred_ref'] = logisticRegr.predict(self.test_set.drop([self.class_col], axis=1))

        return preds

    def predict_proba(self, X: np.array):

        return self.external_classifier.predict_proba(X) 

    def score(self, X: np.array, y: np.array):
        # Wrapper for BayesSearchCV
        return self.external_classifier.score(X, y)

    def get_mc_set(self):
        return self.MC_mass


def create_train_test_split(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame]:
    
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)
    
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    return X_train, X_test, y_train, y_test, train_set, test_set


if __name__ == '__main__':

    # %%
    iris = True
    plt_test_set_mc = False
    hyper_par_tuning = False
    write_to_db = False

    if iris:
        data = pd.read_csv('data/iris.csv', names=['V1', 'V2', "V3", "V4", 'Class'])#, skiprows=skip)

        mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        data = data.replace({"Class": mapping})
        n_classes = 3
    else:
        # Credit card fraud dataset
        n = 284808
        s = 200000 #desired sample size
        skip = sorted(random.sample(range(1, n+1), n - s)) #

        data = pd.read_csv('data/creditcard.csv')#, skiprows=skip)
        data_1 = data[data['Class']==1].copy()#.iloc[:300, :]
        data_0 = data[data['Class']==0].copy()
        
        data = pd.concat([data_0.sample(data_1.shape[0]*1), data_1], axis=0)
        n_classes = 2

    plt.figure()
    data.Class.hist(bins=50)
    plt.show()

    X_train, X_test, y_train, y_test, train_set, test_set = create_train_test_split(df=data, test_size=0.5)

    if hyper_par_tuning:

        opt = BayesSearchCV(
            AIRS(class_number=n_classes,
                column_names=data.drop(['Class'], axis=1).columns.tolist()),
            {
                'hyper_clonal_rate': Integer(20, 50),
                'clonal_rate': Real(0.1, 1, prior='uniform'),
                'mutation_rate': Real(0.1, 1, prior='uniform'),
                'max_iter': Integer(5,10),
                'mc_init_rate': Real(0.1, 0.4, prior='uniform'),
                'total_num_resources': Integer(10, 30),
                'affinity_threshold_scalar': Real(0.05, 0.1, prior='uniform'),
            },
            cv=3,
            n_iter=4,
            random_state=0
        )

        # executes bayesian optimization
        _ = opt.fit(X_train.to_numpy(), y_train.to_numpy())

        # model can be saved, used for predictions or scoring
        print("Optimal bayes hyperpars score", opt.score(X_test, y_test))

        hyper_pars = opt.best_params_
        print(f"Best hyperpars found to be {hyper_pars}")
    else:
        hyper_pars = {
            'hyper_clonal_rate': 100,
            'clonal_rate': 0.8,
            'mutation_rate': 0.6,
            'max_iter': 10,
            'mc_init_rate': 0.2,
            'total_num_resources': 30,
            'affinity_threshold_scalar': 0.05
        }
        
    airs = AIRS(class_number=n_classes,
                column_names=data.drop(['Class'], axis=1).columns.tolist(),
                **hyper_pars)
    
    airs.fit(X=X_train.to_numpy(), y=y_train.to_numpy())

    preds = airs.predict(X=X_test.to_numpy())

    airs.get_evaluation(y_true=y_test)

    mc = airs.get_mc_set()
    
    if write_to_db:
        
        sql_db = sqlite_db(path='ais.db')
        sql_db.conn_to_db()
        
        # ---------------------------- Write MC set to db ---------------------------- #
    
        mc.to_sql('mc_index_1', sql_db.conn, if_exists='replace', index=False)
 
        # --------------------------- Write hyperpars to db -------------------------- #

        hyper_opt_table_schema = {
            'mc_set_index': 'INTEGER',
            'hyper_clonal_rate': 'REAL',
            'clonal_rate': 'REAL',
            'mutation_rate': 'REAL',
            'max_iter': 'INTEGER',
            'mc_init_rate': 'REAL',
            'total_num_resources': 'INTEGER',
            'affinity_threshold_scalar': 'REAL'
        }
        hyper_pars.update({'mc_set_index': 1})

        sql_db.create_db_if_exists('airs_hyper_pars', schema=hyper_opt_table_schema) 
        sql_db.append_row_to_table(input_dict=hyper_pars, table_name='airs_hyper_pars')

        # -------------------- Write performance evaluation to db -------------------- #
        
        # ---------------------------- Close db connection --------------------------- #
        sql_db.close_conn()

    # ---------------------------------------------------------------------------- #
    #                             MC SET VISUALIZATION                             #
    # ---------------------------------------------------------------------------- #
    # %%
    mc.loc[:, 'Class'] += n_classes
    df_plot_train = pd.concat([train_set, mc], axis=0).reset_index(drop=True)
    df_plot = pd.concat([test_set, mc], axis=0).reset_index(drop=True)
    plot_cols = data.columns
    
    df_plot['Class'] = df_plot['Class'].astype(int)
    
    # %%
    df_plot_train['Class'] = df_plot_train['Class'].astype(int)
    for _class in range(n_classes):
        df_plot_tmp = df_plot_train[df_plot_train.Class.isin([_class, _class + n_classes])].copy()
        df_plot_tmp['Class'] = df_plot_tmp['Class'].map('Class{}'.format)

        g = sns.PairGrid(df_plot_tmp[plot_cols], hue="Class") 
        g.map_upper(sns.kdeplot, shade=True, thresh=0.05, alpha=0.7)
        g.map_lower(sns.scatterplot, alpha=0.7) 
        g.map_diag(sns.distplot)
        g.fig.suptitle('MC vs train set', y=0.99) # y= some height>1
        plt.legend()
        plt.show()
    
    # %%
    if plt_test_set_mc:
        for _class in range(n_classes):
            df_plot_tmp = df_plot[df_plot.Class.isin([_class, _class + n_classes])].copy()
            df_plot_tmp['Class'] = df_plot_tmp['Class'].map('Class{}'.format)

            g = sns.PairGrid(df_plot_tmp[plot_cols], hue="Class") 
            g.map_upper(sns.kdeplot, shade=True, thresh=0.05, alpha=0.7)
            g.map_lower(sns.scatterplot, alpha=0.7) 
            g.map_diag(sns.distplot)
            g.fig.suptitle('MC vs test set', y=0.99) # y= some height>1
            plt.legend()
            plt.show()
