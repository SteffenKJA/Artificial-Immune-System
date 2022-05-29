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
    
    AFFINITY_THRESHOLD_SCALAR: The scalar value which modifies the affinity
                               threshold.
    MUTATION_RATE: The fraction of input vector elements who will be assigned
    a new random value.
    
    mc_init_rate: Fraction of memory cells relative to the input training
                  set number of cells (number of rows).

@authors: Steffen Kjær Jacobsen and Azzoug Aghiles.
"""
# %%
from ctypes import Union
import random
import time

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

#os.system('python setup.py build_ext --inplace')
#from func import cy_affinity
# Substitute AIRS.affinity with cy_affinity


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

    def _affinity(self, vector1: np.array, vector2: np.array) -> float:
        """
        Compute the affinity (Normalized!! distance) between two features
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

        dist = np.linalg.norm(vector1 - vector2)
        
        return dist/(1.0 + dist)


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
        affinity_val = self._affinity(vector1=cell[self.cols_features].to_numpy(), vector2=pattern)

        if pd.isna(affinity_val):
            affinity_val = 1

        return 1 - affinity_val

    def _mutate(self, cell: pd.DataFrame) -> Tuple:#[Union[pd.DataFrame, bool]]:
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
        decision_mask = mutate_array <= MUTATION_RATE

        n_mutated = np.where(decision_mask, 1, 0).sum()
        features_to_mutate = np.array(self.cols_features)[decision_mask[0]]

        # if cell['ARB'].iloc[0] == 1:
        #    # cell[self.cols_features][decision_mask[0]] = (7 * np.random.rand(1, n_mutated) + 0.1)[0]
        #     #cell.iloc[:, decision_mask[0]] = (7 * np.random.rand(1, n_mutated) + 0.1)[0]
        #     cell.loc[:, features_to_mutate] = (7 * np.random.rand(1, n_mutated) + 0.1)[0]
        # else:
        #     #cell[self.cols_features][decision_mask[0]] = np.random.rand(1, n_mutated)[0]
        #     #cell.iloc[:, decision_mask[0]] = np.random.rand(1, n_mutated)[0]
        cell.loc[:, features_to_mutate] = np.random.rand(1, n_mutated)[0]

        if n_mutated > 0:
            mutated = True
        else:
            mutated = False

        return cell, mutated


    def _train_test_split(self) -> Tuple[pd.DataFrame]:
        
        df = self.data

        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            random_state=42)
        
        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)
        
        return train_set, test_set


    def _calculate_affinity_threshold(self) -> None:
        """
        Calculates euclidian distance between the datapoints.
        Since we can have a large nr. of features and a very large nr. of 
        rows, this is very computationally expensive and should be optimized.
        
        Perhaps use Cython.
        """
        print("Now calculating affinity threshold..")
            
        # Affinity threshold must be calculated from a normalized
        # dataframe.
        x = self.train_set.iloc[:, :-1].values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)

        self.AFFINITY_THRESHOLD = np.mean(pdist(df.values))
        
        print(f"Affinity threshold found as {self.AFFINITY_THRESHOLD}!")


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

        seed_cells = self.train_set.sample(frac=self.MC_INIT_RATE)
        seed_cells['stimulation'] = 0#float('-inf')
        seed_cells['ARB'] = 0  # Denotes if the cell is an ARB or not.
        seed_cells = seed_cells.reindex(columns=seed_cells.columns.difference(['stimulation', self.class_col]).tolist() + ['stimulation', self.class_col])

        class_mask = seed_cells[self.class_col] == 1

        # Append seed cells to the correct class
        MC[0] = MC[0].append(seed_cells[~class_mask])
        MC[1] = MC[1].append(seed_cells[class_mask])


    def _min_ressource_arb(self, AB: Dict, _class: int) -> Tuple:#[Union[pd.Series, int]]:
        """Get the ARB with the minimum amount of resources
        :param AB: The Artificial Recognition Balls set
        :param _class: the class of the ARBs
        :return: The ARB with the lower amount of resources and its index
        """
        min_res = 1.0
        arb = None
        arb_index = None
        
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
        nbrs = NearestNeighbors(n_neighbors=self.K, algorithm='ball_tree').fit(df_knn.to_numpy())
        distances, indices = nbrs.kneighbors([antigene])
        df_allowed_voters = self.MC_mass.iloc[indices[0], :].copy()
        df_allowed_voters['class_count'] = df_allowed_voters.groupby(self.class_col)['stimulation'].transform('count')
        print('class_count unique is', df_allowed_voters['class_count'].unique())
        df_result = df_allowed_voters.drop_duplicates(self.class_col)
        df_result = df_result[df_result['class_count'].max() == df_result['class_count']]

        if df_result.shape[0]:
            return df_result[self.class_col]
        else:
            return np.nan

    def train(self) -> Tuple[pd.DataFrame]:
        """Train AIRS on the training dataset"""
        start = time.time()

        # Calculate the affinity threshold of the memory cells
        self._calculate_affinity_threshold()
        # The actual affinity threshold is modified by the input value,
        # AFFINITY_THRESHOLD_SCALAR
        self.AFFINITY_THRESHOLD_SCALED = self.AFFINITY_THRESHOLD * self.AFFINITY_THRESHOLD_SCALAR

        # Create two dicts for the Memory Cells and Antibodies, with two
        # empty class slots, nonfraud and fraud, labelled 0 and 1.
        MC = {_class_int: pd.DataFrame() for _class_int in range(self.CLASS_NUMBER)}
        AB = {_class_int: pd.DataFrame() for _class_int in range(self.CLASS_NUMBER)}

        # MC Initialisation
        self._init_MC(MC)

        #for row in self.train_set:
        self.train_set.reset_index(drop=True, inplace=True)
        size = self.train_set.shape[0]

        for index, row in self.train_set.iterrows():
            print(f'Training row {index} out of {size}')
            # Split into featureset (antigene) and target (class)
            antigene, _class = np.array(row[:-1]), int(row[-1])
            
            # ----------------------------- MC Identification ---------------------------- #
            mc_match = None
            
            if len(MC[_class]) == 0:
                # If this is the first row in dataset
                mc_match = pd.DataFrame(row).T
                mc_match['stimulation'] = 0.0 #float('-inf')
                mc_match['ARB'] = 0
                MC[_class] = pd.concat([MC[_class], mc_match], axis=0)
            else:
                # Select a MC candidate.
                # NOTE: We choose the MC with the highest stimulation as a starting point.
                # This stimulation relates to the nearest mc to any previous antigene, not the closests
                # previous antigene to the current antigene, so we could start quite far away in parameter space.
                # The mutated clones will randomly fan out from this positions, but will tend to
                # be in the mass center, since it is a random antigene as reference point (or rather, the
                # mc closests to any of the reference antigene). This mass center tendency means that we effectively
                # remove outliers and concentrate our MCs of a class near the density center of the antigenes of a given class.
                best_stim = 0
                #mc_match = pd.DataFrame(row).T

                max_stim = MC[_class].stimulation.max()
                if (max_stim >= best_stim): # or (MC[_class].shape[0] == 1)
                    best_stim = max_stim
                    mc_match = pd.DataFrame(MC[_class][MC[_class].stimulation == max_stim])

                    if mc_match.shape[0] > 1:
                        mc_match = pd.DataFrame(mc_match.iloc[0, :]).T

            mc_match['ARB'] = 1
            # The stimulation between MC candidate and the incoming antigene.
            mc_match['stimulation'] = self._stimulate(cell=mc_match, pattern=antigene)
            AB[_class] = pd.concat([AB[_class], mc_match], axis=0)  # add the mc_match to ARBs

            stim = mc_match['stimulation']
            iterations = 0

            while True:

            # =================================================================
            #              INITIATE CLONING
            # =================================================================

                iterations += 1
                MAX_CLONES = int(self.HYPER_CLONAL_RATE * self.CLONAL_RATE * stim)
                num_clones = 0

                while num_clones < MAX_CLONES:
                    clone, mutated = self._mutate(cell=mc_match)

                    if mutated:
                        AB[_class] = pd.concat([AB[_class], clone], axis=0)
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

                if sum([AB[c].shape[0] for c in AB.keys()]) < 2:
                    MIN_STIM = 1.0
                    MAX_STIM = 0.0                    

                for c in AB.keys():
                    
                    if len(AB[c].index) > 0:

                        AB[c]['stimulation'] = (AB[c]['stimulation'] - MIN_STIM) / (MAX_STIM - MIN_STIM)
                        AB[c]['resources'] = AB[c]['stimulation'] * self.CLONAL_RATE

                if AB[_class].shape[0] > 0:
                    resAlloc = AB[_class].resources.sum()
                else:
                    resAlloc = 0

                numResAllowed = self.TOTAL_NUM_RESOURCES
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
                if (avgStim > self.AFFINITY_THRESHOLD) or (iterations >= MAX_ITER):
                    break

            mc_candidate = self._get_mc_candidate(AB=AB, _class=_class)

            # get_values()[0]
            if mc_candidate['stimulation'].iloc[0] > float(mc_match.stimulation):
                
                mc_candidate_pattern = np.array(mc_candidate.drop([self.class_col, 'stimulation', 'resources'], axis=1))
                if 'resources' in mc_match.columns:
                    mc_match_pattern = np.array(mc_match.drop([self.class_col, 'stimulation', 'resources'], axis=1)) 
                else: 
                    mc_match_pattern = np.array(mc_match.drop([self.class_col, 'stimulation'], axis=1)) 
                
                # If the mc_candidate and the mc_match (parent mc) are within a threshold distance of each other, remove
                # the mc_match, since mc_candidate in this case is closer to the antigene.
                if self._affinity(vector1=mc_candidate_pattern, vector2=mc_match_pattern) < self.AFFINITY_THRESHOLD*self.AFFINITY_THRESHOLD_SCALAR:

                    mc_class_compare = MC[_class].drop([self.class_col, 'ARB', 'stimulation'], axis=1)
                    mc_class_compare = mc_class_compare.reindex(sorted(mc_class_compare.columns), axis=1)

                    mc_match_compare = mc_match.drop([self.class_col, 'ARB', 'stimulation'], axis=1)
                    mc_match_compare = mc_match_compare.reindex(sorted(mc_match_compare.columns), axis=1)

                    mc_match_index = np.unique(np.where(mc_class_compare.to_numpy()==mc_match_compare.to_numpy())[0]).tolist()
                    
                    # need to drop mc match here
                    # MC[_class].drop(np.where(MC[_class].iloc(4) == mc_match.iloc[4]), axis=0)
                    MC[_class] = MC[_class].reset_index(drop=True).drop(mc_match_index, axis=0)
                
                MC[_class] = pd.concat([MC[_class], mc_candidate], axis=0)

        self.MC = MC
        self.AB = AB

        # Define the MC space of containing MCs of all classes

        self.MC_mass = pd.concat([MC.get(c) for c in MC.keys()], axis=0).reset_index(drop=True)

        n_correct = 0
        
        df_pred = pd.DataFrame({})

        df_pred['y_pred_knn'] = [x.values[0] for x in [self._classify(np.array(x)) for x in self.test_set.iloc[:, :-1].values]]
        
        # Use other classifier
        logisticRegr = LogisticRegression()
        self.MC_mass_train = self.MC_mass.drop([self.class_col, 'ARB', 'stimulation', 'resources'], axis=1)
        self.MC_mass_train = self.MC_mass_train.fillna(self.MC_mass_train.mean())
        logisticRegr.fit(self.MC_mass_train, self.MC_mass[self.class_col])
        df_pred['y_pred'] = logisticRegr.predict(self.test_set.drop([self.class_col], axis=1))
        
        # Train reference model
        logisticRegr = LogisticRegression()
        logisticRegr.fit(self.train_set.drop([self.class_col], axis=1), self.train_set[self.class_col])
        df_pred['y_pred_ref'] = logisticRegr.predict(self.test_set.drop([self.class_col], axis=1))

        print(df_pred['y_pred'].shape)
        print(self.test_set.iloc[:, -1].shape)
        df_pred['y_true'] = self.test_set.iloc[:, -1].values
        n_correct = np.sum([df_pred.y_pred == df_pred.y_true])

        print("Execution time : {:2.4f} seconds".format(time.time() - start))
        print("Accuracy : {:2.2f} %".format(n_correct * 100 / self.test_set.shape[0]))
        print(f"Nonfraud fract : {sum(self.test_set.iloc[:, -1] == 0) / self.test_set.shape[0] * 100:.2f} %")
        print(f"Fraud fract : {sum(self.test_set.iloc[:, -1] == 1) / self.test_set.shape[0] * 100:.2f} %")
        print(f"Confusion matrix is {confusion_matrix(df_pred.y_true, df_pred.y_pred)}")
        print(f"Reference confusion matrix is {confusion_matrix(df_pred.y_true, df_pred.y_pred_ref)}")
        print(f"KNN confusion matrix is {confusion_matrix(df_pred.y_true, df_pred.y_pred_knn)}")

        return self.MC_mass, self.train_set, self.test_set


if __name__ == '__main__':

    # %%
    ARRAY_SIZE = 30  # Features number
    MAX_ITER = 5  # Max iterations to stop training on a given antigene

    # Mutation rate for ARBs
    MUTATION_RATE = 0.2
    iris = True

    if iris:
        data = pd.read_csv('data/iris.csv', names=['V1', 'V2', "V3", "V4", 'Class'])#, skiprows=skip)

        mapping = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        data = data.replace({"Class": mapping})
        n_classes = 3
    else:
        # Credit card fraud dataset
        n = 284808
        s = 200000 #desired sample size
        skip = sorted(random.sample(range(1,n+1), n-s)) #

        data = pd.read_csv('data/creditcard.csv')#, skiprows=skip)
        data_1 = data[data['Class']==1].copy()#.iloc[:300, :]
        data_0 = data[data['Class']==0].copy()
        
        data = pd.concat([data_0.sample(data_1.shape[0]*1), data_1], axis=0)
        n_classes = 2

    plt.figure()
    data.Class.hist(bins=50)
    plt.show()

    # Very low nr of fraud cases, upsample cases.

    airs = AIRS(hyper_clonal_rate=30,
                clonal_rate=0.8,
                class_number=n_classes,
                mc_init_rate=0.4,
                total_num_resources=10, #10
                affinity_threshold_scalar=0.8,
                k=3,
                test_size=0.3,
                data=data)
    # %%
    mc, df_train, df_test = airs.train()
    # %%
    mc.loc[:, 'Class'] += 6
    df_test.loc[:, 'Class'] += 3
    df_plot = pd.concat([df_test, mc], axis=0)

    tsne = TSNE(n_components=n_classes, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df_plot.drop(['ARB', 'stimulation', 'Class', 'resources'], axis=1))

    df_plot['tsne-2d-one'] = tsne_results[:,0]
    df_plot['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure()
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="Class",
        palette=sns.color_palette("husl", n_classes*2),
        data=df_plot,
        alpha=0.3)
    plt.title('Memory cells')
    plt.show()
    # %%
    print('Done')
    tsne = TSNE(n_components=n_classes, verbose=1, perplexity=5, n_iter=1000, learning_rate=20)
    tsne_results = tsne.fit_transform(mc.drop(['ARB', 'stimulation', 'Class', 'resources'], axis=1))

    mc['tsne-2d-one'] = tsne_results[:,0]
    mc['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure()
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="Class",
        palette=sns.color_palette("hls", n_classes),
        data=mc,
        alpha=0.3)
    plt.title('Memory cells')
    plt.show()
    # %%
    plt.figure()
    sns.scatterplot(
        x="V1",
        y="V2",
        hue="Class",
        palette=sns.color_palette("hls", n_classes*2),
        data=df_plot,
        alpha=0.3)
    plt.title('Memory cells')
    plt.show()
    # %%
    tsne = TSNE(n_components=n_classes, verbose=1, perplexity=20, n_iter=1000, learning_rate=5)
    tsne_results = tsne.fit_transform(data)

    data['tsne-2d-one'] = tsne_results[:,0]
    data['tsne-2d-two'] = tsne_results[:,1]
    
    plt.figure()
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="Class",
        palette=sns.color_palette("hls", n_classes),
        data=data,
        alpha=0.3)
    plt.title('Raw data')
    plt.show()
