import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import logging

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self,
                 all_word_sequences: dict,
                 all_word_Xlengths: dict,
                 this_word: str,
                 n_constant=3,
                 min_n_components=2,
                 max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """
    Abbreviations:
        - BIC - Baysian Information Criterion
        - CV - Cross-Validation

    About BIC:
        - Maximises the likelihood of data whilst penalising large-size models
        - Used to scoring model topologies by balancing fit
          and complexity within the training set for each word
        - Avoids using CV by instead using a penalty term

    BIC Equation:  BIC = -2 * log L + p * log N
        (re-arrangment of Equation (12) in Reference [0])

        - where "L" is likelihood of "fitted" model
          where "p" is the qty of free parameters in model (aka model "complexity"). Reference [2][3]
          where "p * log N" is the "penalty term" (increases with higher "p"
             to penalise "complexity" and avoid "overfitting")
          where "N" is qty of data points (size of data set)

        Notes:
          -2 * log L    -> decreases with higher "p"
          p * log N     -> increases with higher "p"
          N > e^2 = 7.4 -> BIC applies larger "penalty term" in this case

    Selection using BIC Model:
        - Lower the BIC score the "better" the model.
        - SelectorBIC accepts argument of ModelSelector instance of base class
          with attributes such as: this_word, min_n_components, max_n_components,
        - Loop from min_n_components to max_n_components
        - Find the lowest BIC score as the "better" model.

    References:
        [0] - http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
        [1] - https://en.wikipedia.org/wiki/Hidden_Markov_model#Architecture
        [2] - https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
        [3] - https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/8
        [4] - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    """

    # High model complexity (i.e. containing many paths) is penalised in BIC and is defined as the number of
    # parameters yet to be estimated (aka free parameters) in the model that are used to fit specific data set
    #
    # p = num_free_params = ("transition probs" == n*n) + means(n*f) + covars(n*f).
    #
    #  where num_free_params means "number of parameters yet to be estimated"
    #  where n means number of model states
    #  where f means number of data points (aka features) used to train the model (i.e. len(self.X[0]))
    #  where probs is an abbreviation for probabilities
    #
    # References:
    # - Discussion about calculating number of free parameters - https://discussions.udacity.com/t/understanding-better-model-selection/232987/7
    #
    # p = num_free_params = init_state_occupation_probs + transition_probs + emission_probs
    #                     = ( num_states ) +
    #                       ( num_states * (num_states - 1) ) +
    #                       ( num_states * num_data_points * 2 )
    #
    #                     Simplifying becomes:
    #                     = ( num_states ** 2 ) + ( 2 * num_states * num_data_points )
    #
    #  where init_state_occupation_probs = num_states
    #  where num_states = possible states a hidden variable at time t may be in
    #  where transition_probs = transition_params = num_states * (num_states - 1)
    #    References: https://en.wikipedia.org/wiki/Hidden_Markov_model#Architecture
    #  where emission_probs (aka output_probs)  = num_states * num_data_points * 2
    #                                           = num_means + num_covars
    #  where num_means and num_covars are number of means and covars calculated
    #  (one of each for each state and feature
    #
    # References:
    # - https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12
    # - https://hmmlearn.readthedocs.io/en/latest/tutorial.html
    #
    # p = num_free_params * (num_states - 1) - num_zeros_in_transiton_matrix
    #
    # References:
    # - https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/11
    # - https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
    # def calc_num_free_params(self, num_states, num_features,num_data_points):
    def calc_num_free_params(self, num_states, num_features):
        # Sai: Reviewer said 'though, to be fair, the definition of p can change based on the problem
        # (for example, most other HMM models do not calculate the initial probabilities)'
        return ( num_states ** 2 ) + ( 2 * num_features * num_states) - 1

    # Sai: Reviewer gave me an advanced BIC tip:
    # You can add a hyperparameter alpha to the free parameters to provide a weight to the free parameters,
    # so the penalty term will become alpha * p * logN.
    # This regularization hyperparameter can then be varied to further improve the result 👍🏽
    def get_model_score(self, num_states, alpha):

        # part 1
        hmm_model = self.base_model(num_states)
        log_likelihood = hmm_model.score(self.X, self.lengths)

        # part 2
        num_free_params = self.calc_num_free_params(num_states, hmm_model.n_features)

        # part 3
        num_data_points = len(self.X)
        logN = np.log(num_data_points)

        # p = = n^2 + 2*d*n - 1
        bic_score = (-2 * log_likelihood) + (alpha * num_free_params * logN)
        return bic_score, hmm_model

    def select(self):
        """ Select best model for self.this_word based on BIC score
        for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_model = None
            best_score = float("Inf")

            for num_states in range(self.min_n_components, self.max_n_components + 1):

                # Reviewer: A good range of alpha is to start with is [0, 0.1, 0.2, ... 1] 😊.
                # You are basically weighting the penalty term less in this way.
                model_score, hmm_model = self.get_model_score(num_states, 0.05)

                # In BIC, the smaller the BIC score the better the model.
                if model_score < best_score:
                    best_score, best_model = model_score, hmm_model

            return best_model

        except:
            return self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    """
    Abbreviations:
        - DIC - Discriminative Information Criterion

    About DIC:
        - In DIC we need to find the number of components where the difference is largest.
        The idea of DIC is that we are trying to find the model that gives a
        high likelihood (small negative number) to the original word and
        low likelihood (very big negative number) to the other words
        - In order to get an optimal model for any word, we need to run the model on all
        other words so that we can calculate the formula
        - DIC is a scoring model topology that scores the ability of a
        training set to discriminate one word against competing words.
        It provides a "penalty" if model likelihoods
        for non-matching words are too similar to model likelihoods for the
        correct word in the word set (rather than using a penalty term for
        complexity like in BIC)
        - Task-oriented model selection criterion adapts well to classification
        problems
        - Classification task accounts for Goal of model  (differs from BIC)

    DIC Equation:

        DIC = log(P(X(i)) - 1/(M - 1) * sum(log(P(X(all but i))

        (Equation (17) in Reference [0]. Assumes all data sets are same size)

            = log likelihood of the data belonging to model
              - avg of anti log likelihood of data X and model M

            = log(P(original word)) - average(log(P(other words)))

        where anti log likelihood means likelihood of data X and model M belonging to competing categories
        where log(P(X(i))) is the log-likelihood of the fitted model for the current word
        (in terms of hmmlearn it is the model's score for the current word)
        where where "L" is likelihood of data fitting the model ("fitted" model)
        where X is input training data given in the form of a word dictionary
        where X(i) is the current word being evaluated
        where M is a specific model

        Note:
            - log likelihood of the data belonging to model
            - anti_log_likelihood of data X vs model M

    Selection using DIC Model:
        - Higher the DIC score the "better" the model.
        - SelectorDIC accepts argument of ModelSelector instance of base class
          with attributes such as: this_word, min_n_components, max_n_components,
        - Loop from min_n_components to max_n_components
        - Find the highest BIC score as the "better" model.

    References:
        [0] - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    """
    def get_model_score(self, num_states):
        """
            Return the dic score based on likehood
        """
        hmm_model = self.base_model(num_states)
        log_likelihood_other_words_scores = []
        for word, (X, lengths) in self.hwords.items():
            # Excluding the current word from the list of words.
            if word != self.this_word:
                log_likelihood_other_words_scores.append(hmm_model.score(X, lengths))

        log_likelihood_original_word_scores = hmm_model.score(self.X, self.lengths)
        return log_likelihood_original_word_scores - np.mean(log_likelihood_other_words_scores), hmm_model


    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score = float("-Inf")
            best_model = None
            for num_states in range(self.min_n_components, self.max_n_components+1):
                score, hmm_model = self.get_model_score(num_states)
                # In DIC, the greater the DIC score the better the model.
                if score > best_score:
                    best_score = score
                    best_model = hmm_model
            return best_model

        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    """
    Abbreviations:
        - CV - Cross-Validation

    About CV:
        - Scoring the model simply using Log Likelihood calculated from
        feature sequences it trained on, we expect more complex models
        to have higher likelihoods, but doesn't inform us which would
        have a "better" likelihood score on unseen data. The model will
        likely overfit as complexity is added.
        - Estimate the "better" Topology model using only training data
        by comparing scores using Cross-Validation (CV).
        - CV technique includes breaking-down the training set into "folds",
        rotating which fold is "left out" of the training set.
        The fold that is "left out" is scored for validation.
        Use this as a proxy method of finding the
        "best" model to use on "unseen data".
        e.g. Given a set of word sequences broken-down into three folds
        using scikit-learn Kfold class object.
        - CV useful to limit over-validation

    CV Equation:

    Selection using CV Model:
        - Higher the CV score the "better" the model.
        - Select "best" model based on average log Likelihood
        of cross-validation folds
        - Loop from min_n_components to max_n_components
        - Find the higher score(logL), the higher the better.
        - Score that is "best" for SelectorCV is the
          average Log Likelihood of Cross-Validation (CV) folds.

    References:
        [0] - http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        [1] - https://www.r-bloggers.com/aic-bic-vs-crossvalidation/
    """

    def get_model_score(self, num_states):

        scores = []
        kf = KFold(n_splits=2) # Our split_method

        # CV loop of breaking-down the sequence (training set) into "folds" where a fold
        # rotated out of the training set is tested by scoring for Cross-Validation (CV)
        for cv_train_idx, cv_test_idx in kf.split(self.sequences):

            # Training sequences split using KFold are recombined
            self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
            hmm_train_model = self.base_model(num_states)

            # Test sequences split using KFold are recombined
            X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
            log_likelihood = hmm_train_model.score(X_test, lengths_test)

            # Sai: We need to keep these sufficient log_likelihood in an array so that you can find the mean.
            scores.append(log_likelihood)
        return np.mean(scores)

    def select(self):
        # logging.debug("Sequences: %r" % self.sequences)

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:

            best_model = None
            best_score = float("-Inf")

            for num_states in range(self.min_n_components, self.max_n_components + 1):

                hmm_model = self.base_model(num_states)
                model_score = self.get_model_score(num_states)
                if model_score > best_score:
                    best_model = hmm_model
                    best_score = model_score
            return best_model

        except Exception as e:
            return self.base_model(self.n_constant)
