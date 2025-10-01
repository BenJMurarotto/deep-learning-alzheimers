import os
import time
import warnings

#os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from tensorflow import keras
from sklearn.model_selection import KFold, cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.svm import SVC

from kanwrapper import KAN
from mlpwrapper import MLP
from vit import ViTWrapper
from dataloader_vis import read_data

warnings.filterwarnings("ignore")

t0 = time.time()

input_shape = (50, 50, 1)

num_inner_folds = 3
num_outer_folds = 3
n_iter = 10
n_jobs = 1

inputs, targets, addresses = read_data(input_shape, address='./oct_data')
targets = targets.reshape(-1, 1)
print('Input shape: {}'.format(inputs.shape))
print('Target shape: {}'.format(targets.shape))

vit_params = dict()
vit_params['projection_dim'] = np.arange(32, 64, 2)
vit_params['patch_size'] = np.arange(4, 9)
vit_params['transformer_layers'] = np.arange(1, 5)
vit_params['num_heads'] = np.arange(1, 5)
vit_params['mlp_head_unit_1'] = [32, 64]
vit_params['mlp_head_unit_2'] = [16, 32]

kan_params = dict()
kan_params['layers'] = np.arange(1, 5)
kan_params['units'] = np.arange(10, 100)
kan_params['grid_size'] = np.arange(1, 100)
kan_params['spline_order'] = np.arange(1, 5)
kan_params['activation'] = ['relu', 'gelu', 'sigmoid']

mlp_params = dict()
mlp_params['layers'] = np.arange(1, 5)
mlp_params['units'] = np.arange(10, 100)
mlp_params['activation'] = ['relu', 'gelu', 'sigmoid']

kfold_inner = KFold(n_splits=num_inner_folds, shuffle=True, random_state=1)
kfold_outer = KFold(n_splits=num_outer_folds, shuffle=True, random_state=1)

#vit_search = BayesSearchCV(estimator=ViTWrapper(), search_spaces=vit_params, n_jobs=n_jobs, n_iter=n_iter, cv=kfold_inner, verbose=3)
kan_search = BayesSearchCV(estimator=KAN(), search_spaces=kan_params, n_jobs=n_jobs, n_iter=n_iter, cv=kfold_inner, verbose=3)
#mlp_search = BayesSearchCV(estimator=MLP(), search_spaces=mlp_params, n_jobs=n_jobs, n_iter=n_iter, cv=kfold_inner, verbose=3)

#vit_scores = cross_val_score(vit_search, inputs, targets, n_jobs=1, cv=kfold_outer, verbose=3)
#print('ViT Accuracy: ', np.mean(vit_scores), np.std(vit_scores))

kan_scores = cross_val_score(kan_search, inputs, targets, n_jobs=1, cv=kfold_outer, verbose=3)
print('KAN Accuracy: ', np.mean(kan_scores), np.std(kan_scores))

#mlp_scores = cross_val_score(mlp_search, inputs, targets, n_jobs=1, cv=kfold_outer, verbose=3)
#print('MLP Accuracy: ', np.mean(mlp_scores), np.std(mlp_scores))

print('Runtime: %d seconds'%(time.time() - t0))
'''
mlp_search.fit(inputs, targets)
print(mlp_search.best_score_)
print(mlp_search.best_params_)
'''
'''
vit_search.fit(inputs, targets)
print(vit_search.best_score_)
print(vit_search.best_params_)
'''
'''
kan_search.fit(inputs, targets)
print(kan_search.best_score_)
print(kan_search.best_params_)
'''
