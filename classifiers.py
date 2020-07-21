from abc import ABC, abstractmethod

import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier_


class Classifier(ABC):
    @abstractmethod
    def model_class(self):
        pass


class LogisticClassifier(Classifier):

    def model_class(self):
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 200, 1000, key='max_iter')
        penalty = st.sidebar.radio("Penalty", ('l1', 'l2'))

        model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty)
        data = pd.DataFrame([C, max_iter, penalty], ['C', 'max_iter', 'penalty'])
        return model, data


class DecisionTreeClassifier(Classifier):

    def model_class(self):
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        max_features = st.sidebar.radio("Max features", ("auto", "sqrt", "log2"), key='max_features')
        min_samples_leaf = st.sidebar.number_input(
                "The minimum number of samples to be at a leaf node",
                0.0, 0.5, step=0.01, key='min_samples_leaf'
            )
        min_samples_split = st.sidebar.number_input(
                "The minimum number of samples to split an internal node",
                1, 20, step=1, key='min_samples_split'
            )
        model = DecisionTreeClassifier_(max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        data = pd.DataFrame([max_features, max_depth, min_samples_leaf, min_samples_split], ['max_features', 'max_depth', 'min_samples_leaf', 'min_samples_split'])
        return model, data


class RandomForestClassifier(Classifier):
    def model_class(self):
        n_estimators = st.sidebar.number_input(
                "The number of trees in the forest",
                100, 5000, step=10, key='new'
            )
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1)
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'))
        n_jobs = -1
        model = RandomForestClassifier_(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=n_jobs)
        data = pd.DataFrame([n_estimators, max_depth, bootstrap, n_jobs], ['n_estimators', 'max_depth', 'bootstrap', 'n_jobs'])
        return model, data


class SVMClassifier(Classifier):

    def model_class(self):
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        data = pd.DataFrame([C, kernel, gamma], ['C', 'gamma', 'kernel'])
        return model, data

