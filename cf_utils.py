# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Ben Lindsay <benjlindsay@gmail.com>

from sklearn.metrics.pairwise import cosine_similarity
import gc
import numpy as np
import pandas as pd


class SimpleAverageModel():
    """A very simple model that just uses the average of the ratings in the
    training set as the prediction for the test set.

    Attributes
    ----------
    mean : float
        Average of the training set ratings
    """

    def __init__(self):
        pass

    def fit(self, X):
        """Given a ratings dataframe X, compute the mean rating
        
        Parameters
        ----------
        X : pandas dataframe, shape = (n_ratings, >=3)
            User, item, rating dataframe. Only the 3rd column is used.
        
        Returns
        -------
        self
        """
        self.mean = X.iloc[:, 2].mean()
        return self

    def predict(self, X):
        return np.ones(len(X)) * self.mean


class AverageByIdModel():
    """Simple model that predicts based on average ratings for a given Id
    (movieId or userId) from training data
    
    Parameters
    ----------
    id_column : string
        Name of id column (i.e. 'itemId', 'userId') to average by in
        dataframe that will be fitted to

    Attributes
    ----------
    averages_by_id : pandas Series, shape = [n_ids]
        Pandas series of rating averages by id
    overall_average : float
        Average rating over all training samples
    """
    def __init__(self, id_column):
        self.id_column = id_column

    def fit(self, X):
        """Fit training data.

        Parameters
        ----------
        X : pandas dataframe, shape = (n_ratings, >=3)
            User, item, rating dataframe. Columns beyond 3 are ignored

        Returns
        -------
        self : object
        """
        rating_column = X.columns[2]
        X = X[[self.id_column, rating_column]].copy()
        X.columns = ['id', 'rating']
        self.averages_by_id = (
            X
            .groupby('id')['rating']
            .mean()
            .rename('average_rating')
        )
        self.overall_average = X['rating'].mean()
        return self

    def predict(self, X):
        """Return rating predictions

        Parameters
        ----------
        X : pandas dataframe, shape = (n_ratings, >=3)
            Array of n_ratings movieIds or userIds

        Returns
        -------
        y_pred : numpy array, shape = (n_ratings,)
            Array of n_samples rating predictions
        """
        rating_column = X.columns[2]
        X = X[[self.id_column, rating_column]].copy()
        X.columns = ['id', 'rating']
        X = X.join(self.averages_by_id, on='id')
        X['average_rating'].fillna(self.overall_average, inplace=True)
        return X['average_rating'].values


def get_rating_matrix(X):
    """Function to generate a ratings matrx and mappings for
    the user and item ids to the row and column indices

    Parameters
    ----------
    X : pandas.DataFrame, shape=(n_ratings,>=3)
        First 3 columns must be in order of user, item, rating.

    Returns
    -------
    rating_matrix : 2d numpy array, shape=(n_users, n_items)
    user_map : pandas Series, shape=(n_users,)
        Mapping from the original user id to an integer in the range [0,n_users)
    item_map : pandas Series, shape=(n_items,)
        Mapping from the original item id to an integer in the range [0,n_items)
    """
    user_col, item_col, rating_col = X.columns[:3]
    rating = X[rating_col]
    user_map = pd.Series(
        index=np.unique(X[user_col]),
        data=np.arange(X[user_col].nunique()),
        name='user_map',
    )
    item_map = pd.Series(
        index=np.unique(X[item_col]),
        data=np.arange(X[item_col].nunique()),
        name='columns_map',
    )
    user_inds = X[user_col].map(user_map)
    item_inds = X[item_col].map(item_map)
    rating_matrix = (
        pd.pivot_table(
            data=X,
            values=rating_col,
            index=user_inds,
            columns=item_inds,
        )
        .fillna(0)
        .values
    )
    return rating_matrix, user_map, item_map


def dcg(top_k_matrix):
    """Compute discounted cumulative gain (DCG) for each row (user) in matrix.
    This measures how good the k recommendations for each user are, with
    decreasing weight placed on items farther down the list. DCG needs to be
    normalized before comparing between users (see normalized discounted
    cumulative gain, or NDCG).
    Links:
        https://link.springer.com/article/10.1007/s11704-015-4584-1
        https://gist.github.com/bwhite/3726239
        https://opensourceconnections.com/blog/2018/02/26/ndcg-scorer-in-quepid
            #cg-dcg-idcg-and-ndcg

    Parameters
    ----------
    top_k_matrix : 2d numpy array, shape = (n_users, k)
        Each row should have the top k ratings for each user from a rating
        matrix in descending order.

    Returns
    -------
    1d numpy array, shape=(n_users,)
        Array of DCG values for each user
    """
    return np.sum(
        top_k_matrix
        /
        np.log2(np.arange(2, top_k_matrix.shape[1]+2))[None, :],
        axis=1
    )


def ndcg(pred_k_matrix, actual_k_matrix):
    """Calculate normalized discounted cumulative gain (NDCG) for each user
    (each row). This is simply the DCG divided by the maximum possible DCG for
    each user. NDCG ranges from 0 to 1, where 1 means movies were chosen that
    actually received the highest k ratings.

    Parameters
    ----------
    pred_k_matrix : 2d numpy array, shape = (n_users, k)
        A matrix of the *actual* ratings of the k movies chosen by the
        recommender system for each user
    actual_k_matrix : 2d numpy array, shape = (n_users, k)
        A matrix of the *actual* ratings of the k movies from the test set
        which the user gave the highest ratings to.

    Returns
    -------
    ndcg_array : 1d numpy array, shape = (n_users,)
        Array of NDCG values for each user
    """
    max_dcg_array = dcg(actual_k_matrix)
    dcg_array = dcg(pred_k_matrix)
    return dcg_array / max_dcg_array

def ndcg_from_df(df, pred, k):
    """Calculate NDCG for each user in the passed dataframe given predicted
    scores and a number of movies to recommend

    Parameters
    ----------
    df : pandas dataframe, shape = (n_ratings, >=3)
        User, item, rating dataframe. All columns after first 3 are ignored
    pred : 1d array-like, shape = (n_ratings,)
        List/array/series of predicted ratings for each user/item pair in df
    k : int
        Number of movies per user to recommend
        
    Returns
    -------
    user_map : pandas series, shape = (n_users,)
        Index = original user ids, value = mapped integer corresponding to
        position in ndcg_array for that user
    ndcg_array : 1d numpy array, shape = (n_users)
        Array of NDCG scores in range (0, 1]
    """
    df = df.iloc[:, :3].copy()
    df.columns = ['user', 'item', 'rating']
    df['pred'] = pred
    pred_matrix, user_map, item_map = get_rating_matrix(df[['user', 'item', 'pred']])
    n_items = len(item_map)
    inds = pred_matrix.argsort(axis=1)[:, :n_items-1-k:-1]
    del pred_matrix
    gc.collect()
    actual_matrix, _, _ = get_rating_matrix(df[['user', 'item', 'rating']])
    pred_k_matrix = actual_matrix[np.arange(len(actual_matrix))[:, None], inds]
    inds = actual_matrix.argsort(axis=1)[:, :n_items-1-k:-1]
    actual_k_matrix = actual_matrix[np.arange(len(actual_matrix))[:, None], inds]
    ndcg_array = ndcg(pred_k_matrix, actual_k_matrix)
    return user_map, ndcg_array

class DampedUserMovieBaselineModel():
    """Baseline model that of the form mu + b_u + b_i,
    where mu is the overall average, b_u is a damped user
    average rating residual, and b_i is a damped item (movie)
    average rating residual. See eqn 2.1 of
    http://files.grouplens.org/papers/FnT%20CF%20Recsys%20Survey.pdf

    Parameters
    ----------
    damping_factor : float, default=0
        Factor to bring residuals closer to 0. Must be positive.

    Attributes
    ----------
    mu : float
        Average rating over all training samples
    b_u : pandas Series, shape = [n_users]
        User residuals
    b_i : pandas Series, shape = [n_movies]
        Movie residuals
    damping_factor : float, default=0
        Factor to bring residuals closer to 0. Must be >= 0.
    """
    def __init__(self, damping_factor=0):
        self.damping_factor = damping_factor

    def fit(self, X):
        """Fit training data.

        Parameters
        ----------
        X : DataFrame, shape = [n_samples, >=3]
            User, movie, rating dataFrame. Columns beyond 3 are ignored

        Returns
        -------
        self : object
        """
        X = X.iloc[:, :3].copy()
        X.columns = ['user', 'item', 'rating']
        self.mu = np.mean(X['rating'])
        user_counts = X['user'].value_counts()
        movie_counts = X['item'].value_counts()
        b_u = (
            X[['user', 'rating']]
            .groupby('user')['rating']
            .sum()
            .subtract(user_counts * self.mu)
            .divide(user_counts + self.damping_factor)
            .rename('b_u')
        )
        X = X.join(b_u, on='user')
        X['item_residual'] = X['rating'] - X['b_u'] - self.mu
        b_i = (
            X[['item', 'item_residual']]
            .groupby('item')['item_residual']
            .sum()
            .divide(movie_counts + self.damping_factor)
            .rename('b_i')
        )
        self.b_u = b_u
        self.b_i = b_i
        return self

    def predict(self, X):
        """Return rating predictions

        Parameters
        ----------
        X : DataFrame, shape = (n_ratings, 2)
            User, item dataframe

        Returns
        -------
        y_pred : numpy array, shape = (n_ratings,)
            Array of n_samples rating predictions
        """
        X = X.iloc[:, :2].copy()
        X.columns = ['user', 'item']
        X = X.join(self.b_u, on='user').fillna(0)
        X = X.join(self.b_i, on='item').fillna(0)
        return (self.mu + X['b_u'] + X['b_i']).values


class KNNRecommender():
    """User-based or Item-based collaborative filtering model that operates on
    dataframes with at least a user-like, item-like, and a rating-like column

    Parameters
    ----------
    mode : str, ['item | 'user'], default='item'
        Tells model whether to use item-based or user-based collaborative filtering
    k : int, default=20
        Number of most similar items or users to average for prediction
    basline_algo : object, optional
        Algorithm used to predict baseline scores for each rating. If not provided,
        the mean of all training ratings is used as the baseline. If provided,
        the object must have a fit(X) method and a predict(X) method
    similarity_func : function, default=cosine_similarity
        Function must take a numpy array M of shape (m,n) and return a numpy array
        of shape (m,m) where each element i,j represents the similarity between row
        i and row j of M.
    loop_predict : boolean, default=True
        If True, the model will loop over all user-item pairs in test set and compute
        prediction individually. If False, the model will compute all ratings
        simultaneously. With sparse matrices, looping is typically faster.

    Attributes
    ----------
    train_mean : float
        Mean of the training data ratings. Used if baseline_algo is None.
    rating_matrix : 2d numpy array, shape=(n_users, n_items)
        Rating matrix minus baselines
    user_map : pandas Series, shape=(n_users,)
        Mapping from the original user id to an integer in the range [0,n_users)
    item_map : pandas Series, shape=(n_items,)
        Mapping from the original item id to an integer in the range [0,n_items)
    knn_indices : 2d numpy array, shape=([n_users|n_items], k)
        Element i,j represents the index of the jth closet [user|item] to i
    knn_similarities : 2d numpy array, shape=([n_users|n_items], k)
        Element i,j represents the similarity between the jth closest [user|item] to i
    """
    def __init__(self, mode='item', k=20, baseline_algo=None,
                 similarity_func=cosine_similarity, loop_predict=True):
        if not mode in ['user', 'item']:
            raise ValueError("'mode' must be either 'user' or 'item', not '{}'!".format(mode))

        self.mode = mode
        # force k to be int in case it accidentally comes in as float
        self.k = int(np.round(k))
        self.baseline_algo = baseline_algo
        self.similarity_func = similarity_func
        self.loop_predict = loop_predict
        

        self.train_mean = None
        self.rating_matrix = None
        self.user_map = None
        self.item_map = None
        self.knn_indices = None
        self.knn_similarities = None

    def _get_knn_indices_and_similarities(self, rating_matrix):
        """Private function to find indices and similarities of k nearest
        neighbors for each user or item

        Parameters
        ----------
        rating_matrix : 2d numpy array, shape=(n_users, n_items)
            Matrix of ratings minus baselines

        Returns
        -------
        knn_indices : 2d numpy array, shape=([n_users|n_items], k)
            Element i,j represents the index of the jth closet [user|item] to i
        knn_similarities : 2d numpy array, shape=([n_users|n_items], k)
            Element i,j represents the similarity between the jth closest [user|item] to i
        """
        if self.mode == 'item':
            n_users_or_items = rating_matrix.shape[1]
        else:
            n_users_or_items = rating_matrix.shape[0]
        if self.k > n_users_or_items:
            new_k = n_users_or_items - 1
            print(
                "Warning: k = {} > # {}s = {}! Setting k to {}"
                .format(self.k, n_users_or_items, self.mode, new_k)
            )
            self.k = new_k
        if self.mode == 'item':
            similarity_matrix = self.similarity_func(rating_matrix.T)
        else:
            similarity_matrix = self.similarity_func(rating_matrix)
        np.fill_diagonal(similarity_matrix, -1)
        knn_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :self.k]
        # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/neighbors/base.py#L373
        sample_range = np.arange(len(knn_indices))[:, None]
        knn_similarities = similarity_matrix[sample_range, knn_indices]
        return knn_indices, knn_similarities

    def fit(self, X):
        """Fit model to training data X. Sets the knn_indices, knn_similarities,
        rating_matrix, user_map, and item map variables.

        Parameters
        ----------
        X : pandas DataFrame, shape=(n_ratings, >=3)
            First 3 columns must correspond to user, item, and rating in that order

        Returns
        -------
        self
            This allows chaining like `preds = KNNRecommender().fit(X_train).predict(X_test)`
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col, rating_col = X.columns[:3]
        if self.baseline_algo is None:
            self.train_mean = X[rating_col].mean()
            X['rating_baseline'] = self.train_mean
        else:
            self.baseline_algo.fit(X)
            X['rating_baseline'] = self.baseline_algo.predict(X[[user_col, item_col]])
        X['rating_diff'] = X[rating_col] - X['rating_baseline']
        nodiff_rating_matrix, _, _ = get_rating_matrix(X[[user_col, item_col, rating_col]])
        self.knn_indices, self.knn_similarities = self._get_knn_indices_and_similarities(
            nodiff_rating_matrix
        )
        gc.collect()
        self.rating_matrix, self.user_map, self.item_map = get_rating_matrix(
            X[[user_col, item_col, 'rating_diff']]
        )
        return self

    def _predict_1_ui_pair(self, user, item):
        """Predict rating (minus baseline) for 1 user-item pair. Must add
        baseline to get the rating in the original rating scale.

        Parameters
        ----------
        user : int
            Must be in range [0, n_users)
        item : int
            Must be in range [0, n_items)

        Returns
        -------
        rating_pred : float
            Predicted ratings
        """
        if self.mode == 'item':
            inds_i = self.knn_indices[item, :]
            sims_i = self.knn_similarities[item, :]
            # https://stackoverflow.com/a/35696047/2680824
            numerator = np.sum(self.rating_matrix[user, inds_i] * sims_i)
            denominator = np.sum(np.abs(sims_i))
            with np.errstate(divide='ignore', invalid='ignore'):
                rating_pred = numerator / denominator
        else:
            inds_u = self.knn_indices[user, :]
            sims_u = self.knn_similarities[user, :]
            # https://stackoverflow.com/a/35696047/2680824
            numerator = np.sum(self.rating_matrix[inds_u, item] * sims_u)
            denominator = np.sum(np.abs(sims_u))
            with np.errstate(divide='ignore', invalid='ignore'):
                rating_pred = numerator / denominator
        return rating_pred

    def predict(self, X):
        """Predict ratings for each user-item pair in X

        Parameters
        ----------
        X : pandas DataFrame, shape=(n_ratings, >=2)
            First 2 columns of X must correspond to user and item.

        Returns
        -------
        pandas Series, shape=(n_ratings,)
            Ratings for each user-item pair in X. No restriction on the data type
            for the user and item ids, other than they must match the training indices.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col = X.columns[:2]
        if self.baseline_algo is None:
            X['rating_baseline'] = self.train_mean
        else:
            X['rating_baseline'] = self.baseline_algo.predict(X)
        X['rating'] = 0
        known_user_and_item_mask = (
            X[user_col].isin(self.user_map.index) & X[item_col].isin(self.item_map.index)
        )
        X_known = X[known_user_and_item_mask]
        user_inds = X_known[user_col].map(self.user_map)
        item_inds = X_known[item_col].map(self.item_map)
        if self.loop_predict:
            rating_pred = np.array([
                self._predict_1_ui_pair(u_ind, i_ind)
                for u_ind, i_ind in zip(user_inds, item_inds)
            ])
        else:
            stacked_ratings = self.rating_matrix[
                self.knn_indices[:, :, None],
                np.arange(self.rating_matrix.shape[1])[None, None, :]
            ]
            numerator_matrix = np.sum(
                stacked_ratings * self.knn_similarities[:, :, None],
                axis=1
            )
            denominator_matrix = np.sum(
                (stacked_ratings != 0) * self.knn_similarities[:, :, None],
                axis=1
            )
            # https://stackoverflow.com/a/35696047/2680824
            with np.errstate(divide='ignore', invalid='ignore'):
                rating_pred_matrix = numerator_matrix / denominator_matrix
            rating_pred = rating_pred_matrix[user_inds, item_inds]
        rating_pred[np.isnan(rating_pred)] = 0
        X.loc[known_user_and_item_mask, 'rating'] = rating_pred
        return X['rating'] + X['rating_baseline']


class ALSRecommender():
    """Recommender based on Alternating Least Squares algorithm.
    
    Parameters
    ----------
    k : int, default=5
        Number of latent features
    lmbda : float, default=0.1
        Regularization parameter
    max_epochs : int, default=15
        Max number of iterations to run
    baseline_algo : object
        Object with fit(X) and 
    """
    def __init__(self, k=5, lmbda=0.1, max_epochs=15, baseline_algo=None, error_metric='mae',
                 verbose=True):
        # Force integer in case it comes in as float
        self.k = int(np.round(k))
        self.lmbda = lmbda
        self.max_epochs = max_epochs
        self.baseline_algo = baseline_algo
        self.error_metric = error_metric
        self.verbose = verbose

        self.U = None
        self.I = None
        self.initialized = False

    def _calc_train_error(self, U, I, R, R_selector=None, error_metric='mae'):
        if R_selector is None:
            R_selector = (R > 0)
        R_hat = np.dot(U.T, I)
        if error_metric == 'mae':
            error = np.sum(R_selector * np.abs(R_hat - R)) / np.sum(R_selector)
        else:
            raise ValueError("{} is an unsupported error metric".format(metric))
        return error

    def _fit_init(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col, rating_col = X.columns[:3]
        if self.baseline_algo is None:
            self.train_mean = X[rating_col].mean()
        else:
            self.baseline_algo.fit(X)
        self.R, self.user_map, self.item_map = get_rating_matrix(X)
        n_users, n_items = self.R.shape
        self.U = 3 * np.random.rand(self.k, n_users)
        self.I = 3 * np.random.rand(self.k, n_items)
        self.I[0, :] = self.R[self.R != 0].mean(axis=0) # Avg. rating for each movie
        self.E = np.eye(self.k) # (k x k)-dimensional idendity matrix
        self.epoch = 0
        self.train_errors = []
        self.initialized = True

    def fit(self, X, n_epochs=None):
        """Fit model to training data X. If at least one iteration has already been run,
        then the model will continue from its most recent state.

        Parameters
        ----------
        X : pandas DataFrame, shape=(n_ratings, >=3)
            First 3 columns must correspond to user, item, and rating in that order
        n_epochs : int, optional
            Number of iterations to run. If not provided, will run for self.max_epochs

        Returns
        -------
        self
            This allows chaining like `ALSRecommender().fit(X_train).predict(X_test)`
        """
        # Allow continuation from previous state if n_epochs is given. Otherwise start from scratch.
        if n_epochs is None:
            self.initialized = False
        if not self.initialized:
            self._fit_init(X)

        epoch_0 = self.epoch
        if n_epochs is None:
            n_epochs = self.max_epochs - epoch_0

        n_users, n_items = self.R.shape

        # Run n_epochs iterations
        for i_epoch in range(n_epochs):
            if self.epoch >= self.max_epochs:
                print("max_epochs = {}".format(self.max_epochs))
                break
            # Fix I and estimate U
            for i, Ri in enumerate(self.R):
                nui = np.count_nonzero(Ri) # Number of items user i has rated
                if (nui == 0): nui = 1 # Be aware of zero counts!
                # Get array of nonzero indices in row Ii
                Ri_nonzero_selector = np.nonzero(Ri)[0]
                # Select subset of I associated with movies reviewed by user i
                I_Ri = self.I[:, Ri_nonzero_selector]
                # Select subset of row R_i associated with movies reviewed by user i
                Ri_nonzero = self.R[i, Ri_nonzero_selector]
                Ai = np.dot(I_Ri, I_Ri.T) + self.lmbda * nui * self.E
                Vi = np.dot(I_Ri, Ri_nonzero.T)
                self.U[:, i] = np.linalg.solve(Ai, Vi)
            # Fix U and estimate I
            for j, Rj in enumerate(self.R.T):
                nmj = np.count_nonzero(Rj) # Number of users that rated item j
                if (nmj == 0): nmj = 1 # Be aware of zero counts!
                # Get array of nonzero indices in row Ij
                Rj_nonzero_selector = np.nonzero(Rj)[0]
                # Select subset of P associated with users who reviewed movie j
                U_Rj = self.U[:, Rj_nonzero_selector]
                # Select subset of column R_j associated with users who reviewed movie j
                Rj_nonzero = self.R[Rj_nonzero_selector, j]
                Aj = np.dot(U_Rj, U_Rj.T) + self.lmbda * nmj * self.E
                Vj = np.dot(U_Rj, Rj_nonzero)
                self.I[:, j] = np.linalg.solve(Aj, Vj)
            error = self._calc_train_error(self.U, self.I, self.R)
            self.train_errors.append(error)
            if self.verbose:
                print("[Epoch {}/{}] train error: {}".format(self.epoch, self.max_epochs, error))
            self.epoch += 1
        return self

    def predict(self, X):
        """Generate predictions for user/item pairs
        
        Parameters
        ----------
        X : pandas dataframe, shape = (n_pairs, 2)
            User, item dataframe
            
        Returns
        -------
        rating_pred : 1d numpy array, shape = (n_pairs,)
            Array of rating predictions for each user/item pair
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col = X.columns[:2]
        if self.baseline_algo is None:
            X['rating_baseline'] = self.train_mean
        else:
            X['rating_baseline'] = self.baseline_algo.predict(X)
        X['rating'] = 0
        known_user_and_item_mask = (
            X[user_col].isin(self.user_map.index) & X[item_col].isin(self.item_map.index)
        )
        X_known, X_unknown = X[known_user_and_item_mask], X[~known_user_and_item_mask]
        user_inds = X_known[user_col].map(self.user_map)
        item_inds = X_known[item_col].map(self.item_map)
        rating_pred = np.array([
            np.sum(self.U[:, u_ind] * self.I[:, i_ind])
            for u_ind, i_ind in zip(user_inds, item_inds)
        ])
        X.loc[known_user_and_item_mask, 'rating'] = rating_pred
        X.loc[~known_user_and_item_mask, 'rating'] = self.baseline_algo.predict(X_unknown)
        min_rating = np.min(self.R[np.nonzero(self.R)])
        max_rating = np.max(self.R)
        X.loc[X['rating'] < min_rating, 'rating'] = min_rating
        X.loc[X['rating'] > max_rating, 'rating'] = max_rating
        return X['rating'].values


class SGDRecommender():
    """Stochastic Gradient Descent recommender.
    
    Parameters
    ----------
    k : int, default=5
        Number of latent features
    learning_rate : float, default=0.1
        Speed at which to descend down gradient
    max_epochs : int, default=15
        Max number of iterations to run
    error_metric : string, default='mae'
        Error metric to use
    user_reg : float, default=0.0
        Regularization parameter for the latent feature weights in U, >=0
    item_reg : float, default=0.0
        Regularization parameter for the latent feature weights in I, >=0
    user_bias_reg : float, default=0.0
        Regularization parameter for the b_u terms, >=0
    item_bias_reg : float, default=0.0
        Regularization parameter for the b_i terms, >=0
    damping_factor : float, default=25
        Damping factor to be used in the baseline algorithm
    minibatch_size : int, default=1
        Number of user/item pairs to evaluate at a time during training
    verbose : boolean, default=True
        If True, print progress.
    """
    def __init__(self, k=5, learning_rate=0.1, max_epochs=15, error_metric='mae',
                 user_reg=0.0, item_reg=0.0, user_bias_reg=0.0, item_bias_reg=0.0,
                 damping_factor=25, minibatch_size=1, verbose=True):
        self.k = k
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.error_metric = error_metric
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_bias_reg = user_bias_reg
        self.item_bias_reg = item_bias_reg
        self.damping_factor = damping_factor
        self.minibatch_size = minibatch_size
        self.verbose = verbose

        self.U = None
        self.I = None
        self.initialized = False

    def _calc_train_error(self, U, I, mu, b_u, b_i, R, R_selector=None):
        if R_selector is None:
            R_selector = (R > 0)
        R_hat = np.dot(U, I.T) + mu + b_u[:, None] + b_i[None, :]
        if self.error_metric == 'mae':
            error = np.sum(R_selector * np.abs(R_hat - R)) / np.sum(R_selector)
        else:
            raise ValueError("{} is an unsupported error metric".format(metric))
        return error

    def _fit_init(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        user_col, item_col, rating_col = X.columns[:3]
        self.baseline_algo = DampedUserMovieBaselineModel(damping_factor=self.damping_factor)
        self.baseline_algo.fit(X)
        self.mu = X[rating_col].mean()
        self.b_u, self.b_i = self.baseline_algo.b_u.values, self.baseline_algo.b_i.values
        self.R, self.user_map, self.item_map = get_rating_matrix(X)
        n_users, n_items = self.R.shape
        self.U = np.random.normal(scale=1.0/self.k, size=(n_users, self.k))
        self.I = np.random.normal(scale=1.0/self.k, size=(n_items, self.k))
        self.epoch = 0
        self.train_errors = []
        self.initialized = True

    def fit(self, X, n_epochs=None):
        """Fit model to training data X. If at least one iteration has already been run,
        then the model will continue from its most recent state.

        Parameters
        ----------
        X : pandas DataFrame, shape=(n_ratings, >=3)
            First 3 columns must correspond to user, item, and rating in that order
        n_epochs : int, optional
            Number of iterations to run. If not provided, will run for self.max_epochs

        Returns
        -------
        self
            This allows chaining like `SGDRecommender().fit(X_train).predict(X_test)`
        """
        X = X.copy()
        # Allow continuation from previous state if n_epochs is given. Otherwise start from scratch.
        if n_epochs is None:
            self.initialized = False
        if not self.initialized:
            self._fit_init(X)
        X.iloc[:, 0] = X.iloc[:, 0].map(self.user_map)
        X.iloc[:, 1] = X.iloc[:, 1].map(self.item_map)

        epoch_0 = self.epoch
        if n_epochs is None:
            n_epochs = self.max_epochs - epoch_0

        n_users, n_items = self.R.shape

        # Repeat until convergence
        for i_epoch in range(n_epochs):
            if self.epoch >= self.max_epochs:
                print("max_epochs = {}".format(self.max_epochs))
                break
            # Shuffle X
            X = X.sample(frac=1)
            if self.minibatch_size == 1:
                for row in X.itertuples():
                    index, user, item, rating = row[:4]
                    pred = self.predict_1_train(user, item)
                    err = pred - self.R[user, item]
                    self.b_u[user] -= self.learning_rate * (err + self.user_bias_reg * self.b_u[user])
                    self.b_i[item] -= self.learning_rate * (err + self.item_bias_reg * self.b_i[item])
                    self.U[user, :] -= self.learning_rate * (
                        err * self.I[item, :] + self.user_reg * self.U[user, :]
                    )
                    self.I[item, :] -= self.learning_rate * (
                        err * self.U[user, :] + self.item_reg * self.I[item, :]
                    )
            else:
                raise ValueError("Minibatch size greater than 1 not supported yet.")
            error = self._calc_train_error(self.U, self.I, self.mu, self.b_u, self.b_i, self.R)
            self.train_errors.append(error)
            if self.verbose:
                print("[Epoch {}/{}] train error: {}".format(self.epoch, self.max_epochs, error))
            self.epoch += 1
        return self

    def predict_1_train(self, user, item):
        pred = self.mu + self.b_u[user] + self.b_i[item]
        pred += np.dot(self.U[user, :], self.I[item, :])
        return pred

    def predict(self, X):
        """Generate predictions for user/item pairs
        
        Parameters
        ----------
        X : pandas dataframe, shape = (n_pairs, 2)
            User, item dataframe
            
        Returns
        -------
        rating_pred : 1d numpy array, shape = (n_pairs,)
            Array of rating predictions for each user/item pair
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame")
        X = X.copy()
        user_col, item_col = X.columns[:2]
        known_user_and_item_mask = (
            X[user_col].isin(self.user_map.index) & X[item_col].isin(self.item_map.index)
        )
        X_known, X_unknown = X[known_user_and_item_mask], X[~known_user_and_item_mask]
        user_inds = X_known[user_col].map(self.user_map)
        item_inds = X_known[item_col].map(self.item_map)
        rating_pred = np.array([
            self.predict_1_train(u_ind, i_ind)
            for u_ind, i_ind in zip(user_inds, item_inds)
        ])
        X.loc[known_user_and_item_mask, 'rating'] = rating_pred
        X.loc[~known_user_and_item_mask, 'rating'] = self.baseline_algo.predict(X_unknown)
        return X['rating'].values
