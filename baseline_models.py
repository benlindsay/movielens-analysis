# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Ben Lindsay <benjlindsay@gmail.com>

import numpy as np
import pandas as pd


class SimpleAverageModel():
    """A very simple model that just uses the average of the ratings in the
    training set as the prediction for the test set.

    Attributes
    ----------
    mean_ : float
        Average of the training set ratings
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        self.mean_ = y.mean()

    def predict(self, X):
        return np.ones(len(X)) * self.mean_


class AverageByIdModel():
    """Simple model that predicts based on average ratings for a given Id
    (movieId or userId) from training data

    Attributes
    ----------
    averages_by_id_ : pandas Series, shape = [n_ids]
        Pandas series of rating averages by id
    overall_average_ : float
        Average rating over all training samples
    """
    def __init__(self, column):
        self.column = column

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples]
            Array of n_samples movieIds or userIds
        y : array-like, shape = [n_samples]
            Target values (movie ratings)

        Returns
        -------
        self : object
        """
        X_y_df = pd.DataFrame({'id': X[self.column], 'rating': y})
        self.averages_by_id_ = (
            X_y_df
            .groupby('id')['rating']
            .mean()
            .rename('average_rating')
        )
        self.overall_average_ = np.mean(y)

    def predict(self, X):
        """Return rating predictions

        Parameters
        ----------
        X : array-like, shape = [n_samples]
            Array of n_samples movieIds or userIds

        Returns
        -------
        y_pred : array-like, shape = [n_samples]
            Array of n_samples rating predictions
        """
        if isinstance(X, pd.DataFrame):
            # if X.shape[1] > 1:
            #     raise ValueError(
            #         "X should be a 1D array-like object"
            #     )
            X = X[self.column]
        X_df = pd.DataFrame({'id': X})
        X_df = X_df.join(self.averages_by_id_, on='id')
        X_df['average_rating'].fillna(self.overall_average_, inplace=True)
        return X_df['average_rating'].values


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
    mu_ : float
        Average rating over all training samples
    b_u_ : pandas Series, shape = [n_users]
        User residuals
    b_i_ : pandas Series, shape = [n_movies]
        Movie residuals
    damping_factor_ : float, default=0
        Factor to bring residuals closer to 0. Must be positive.
    """
    def __init__(self, damping_factor=0):
        self.damping_factor_ = damping_factor

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : DataFrame, shape = [n_samples, 2]
            DataFrame with columns 'userId', and 'movieId'
        y : array-like, shape = [n_samples]
            Target values (movie ratings)

        Returns
        -------
        self : object
        """
        X = X.copy()
        X['rating'] = y
        self.mu_ = np.mean(y)
        user_counts = X['userId'].value_counts()
        movie_counts = X['movieId'].value_counts()
        b_u = (
            X[['userId', 'rating']]
            .groupby('userId')['rating']
            .sum()
            .subtract(user_counts * self.mu_)
            .divide(user_counts + self.damping_factor_)
            .rename('b_u')
        )
        X = X.join(b_u, on='userId')
        X['movie_residual'] = X['rating'] - X['b_u'] - self.mu_
        b_i = (
            X[['movieId', 'movie_residual']]
            .groupby('movieId')['movie_residual']
            .sum()
            .divide(movie_counts + self.damping_factor_)
            .rename('b_i')
        )
        self.b_u_ = b_u
        self.b_i_ = b_i
        return self

    def predict(self, X):
        """Return rating predictions

        Parameters
        ----------
        X : DataFrame, shape = [n_samples, 2]
            DataFrame with columns 'userId', and 'movieId'

        Returns
        -------
        y_pred : array-like, shape = [n_samples]
            Array of n_samples rating predictions
        """
        X = X.copy()
        X = X.join(self.b_u_, on='userId').fillna(0)
        X = X.join(self.b_i_, on='movieId').fillna(0)
        return self.mu_ + X['b_u'] + X['b_i']
