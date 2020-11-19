# -*- coding: utf-8 -*-
"""
# Created on Wed May 27 22:13:54 2020

@author: Kofi Agyeman
# UCR CS235 - Data Mining Techbiques: Spring 2020
 
# ************************************************************************
# Phishing detection algorithm: Support Vector Machine implementation

# Data set:
# https://www.kaggle.com/akashkr/phishing-website-dataset

# Sources:
#    Parts of implementation inspired by SVM tutorials from scikit learn and 
#    Towards datascience svn from scratch online
@ https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
@ https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2

"""
#%% Load requisite libraries
#=============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score

#=============================================================================


#%% Load, Partition and visualize data:
#=============================================================================    
"""
 Data courtesy of Kaggle data repository 
 Selected a complete data set with no missing entries with labeled and defined
 features
 * No need to handle missing data
 * feature values of dataset already normalized to [-1, 0, 1]
 * Therefore no need to normalize the data set
"""
st_tm = datetime.now()
print("start time: ", st_tm)
print("Loading Data set.... ")

# Load data =======
phis_data = pd.read_csv('Data/dataset.csv')    
print("Data columns: ", phis_data.columns)
phis_data.head()

# Get number of legitimate and phishing sites in dataset
num_ph = len(phis_data[phis_data.Result == -1])
num_lg = len(phis_data[phis_data.Result == 1])

print("Total # of legitimate data tuples: ", num_lg)
print("Total # of phishing data tuples", num_ph)

# Extract predictor data frame and output data series 
y_full_dt = phis_data.loc[:, 'Result']    # Pull out 'Result' column from data frame 
x_full_dt = phis_data.iloc[:, 1:31]       # Drop 'Index' and 'Result' columns

out_y = y_full_dt.to_numpy()              # convert output series to vector
prd_x = x_full_dt.to_numpy()              # convert feature columns to array
prd_x_b = np.append(np.ones([len(prd_x), 1]), prd_x, 1) # add a column of ones for b intercept 


# ======== Split data into Training and Testing (20 %) partitions: ===========
x_trn, x_tst, y_trn, y_tst = train_test_split(prd_x_b, out_y, test_size = 0.2,
                                              random_state = 42)

# Count number of phishing and legit data tuples in test and training data
num_tst_ph = np.count_nonzero(y_tst == -1)
num_tst_lg = np.count_nonzero(y_tst == 1)
num_trn_ph = np.count_nonzero(y_trn == -1)
num_trn_lg = np.count_nonzero(y_trn == 1)

# Creat a bar chart of training and test legit and phishing data tuples
data_bar = {'Train': len(x_trn), 'Test': len(x_tst), 'Legit': num_lg, 'Phis': num_ph,
            'Train_leg': num_trn_lg, 'Train_phis': num_trn_ph, 'Test_leg': num_tst_lg,
            'Test_phis': num_tst_ph}

# Print phishing dataset information and feature list ==============
phis_data.info()                        # Dataset information
phis_data.describe()

# Create bar chart display of total dataset and training and testing splits 
names = list(data_bar.keys())
vals = list(data_bar.values())

fig = plt.figure()
plt.bar(names, vals)
plt.title("Dataset Breakdown")
plt.xlabel("data")
plt.ylabel("count")
plt.show() 

#%% SVM with linear kernel 
#=============================================================================

# Function fk_linear_svm: performs linear svm and outputs weights
def fk_linear_svm(ln_rate, reg, mx_eps, ct_thr, x_Trn, y_Trn):
    mx_epochs = mx_eps                            # max trial epochs
    wghts = np.zeros(x_Trn.shape[1])              # init weights vector
    n_stps = 0
    prv_cost = float("inf")
    cost_thrshd = ct_thr                          # cost threshold
    
    for epch in range(1, mx_epochs):
        x_tp, y_tp = shuffle(x_Trn, y_Trn)
        for i, x in enumerate(x_tp):
            ys = np.array([y_tp[i]])
            xs = np.array([x])
            dist = 1 - (ys*np.dot(xs, wghts))
            dfw = np.zeros(len(wghts))
            
            for i, d in enumerate(dist):
                if max(0, d) == 0:
                    di = wghts
                else:
                    di = wghts - (reg*ys[i]*xs[i])
                dfw += di
        
            dfw = dfw/len(ys)
            asct = dfw
            wghts = wghts - (ln_rate*asct)
            
        if epch == 2**n_stps or epch == mx_epochs - 1:
            N = x_Trn.shape[0]
            dist = 1 - y_Trn*(np.dot(x_Trn, wghts))
            dist[dist < 0] = 0                               # max distance
            hg_loss = reg*(np.sum(dist) / N)
            cost = 1 / 2*np.dot(wghts, wghts) + hg_loss      # find cost
            
            print ("Itra is: {} and Cost is: {}".format(epch, cost))
            if abs(prv_cost - cost) < cost_thrshd*prv_cost:
                wght_final = wghts
                print("the minimum cost", cost)
            prv_cost = cost
            n_stps = n_stps + 1
            
    wght_final = wghts
    return wght_final

# Function prints out SVM kernel evaluation report
def fk_performance_printout(y_t, y_p, kern):
    cf_mat = confusion_matrix(y_t, y_p)
    class_rept = classification_report(y_t, y_p, digits = 4)
    acrcy = accuracy_score(y_t, y_p)
    r_call = recall_score(y_t, y_p)
    precsn = precision_score(y_t, y_p)
    
    # Print performance measures on test data set
    print(kern, ": Model performance on test dataset:")
    print("Confusion matrix: ")
    print(cf_mat)
    print("Classification Report:")
    print(class_rept)
    print("Acccuracy on test dataset: {}".format(acrcy))
    print("Recall on test dataset: {}".format(r_call))
    print("Precision on test dataset: {}".format(precsn))
    
#%% Train linear SVM kernel model
# ===========================================================================
print("start svm model training....")

# Parameters
reg = 10000
lrn_rt = 0.000001
mx_eps = 5000
cst_t = 0.001

# Train linear svm kernel and find weights
svm_Wts = fk_linear_svm(lrn_rt, reg, mx_eps, cst_t, x_trn, y_trn)

print("Training completed")
#print("Svm weights: {}".format(svm_wghts))

print("Training finished.")
print("weights are: {}".format(svm_Wts))

# Linear SVM clssification evaluation and testing
#=========================================================================
print("Testing and evaluation of the model...")
# For all x test data predict class with trained model
y_test_predicted = np.array([])
for i in range(x_tst.shape[0]):
    yp = np.sign(np.dot(x_tst[i], svm_Wts))
    y_test_predicted = np.append(y_test_predicted, yp)  # Class predictions
    

# Print performance measures on test data set
fk_performance_printout(y_tst, y_test_predicted, "SVM Linear kernel")

end_tm = datetime.now()
duration = end_tm - st_tm
print("Linear svm duration: ", duration)

#%% Non-linear kernel and parameter exploration and optimization (poly and rbf)
#   kernels and gamma/C parameters
#=============================================================================
rbf_st_tm = datetime.now()
print("Rbf start time: ", rbf_st_tm)            # print cell start time

gamma_vals = np.logspace(-2, 2, 5)              # gamma values to test

c_vals = np.logspace(-2, 2, 5)                  # C values to test
param_grid = dict(gamma = gamma_vals, C = c_vals)

# split training data for trainng and cross validation purposes 
cv_splits = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, 
                                   random_state = 50)

#%% Gaussian (rbf) Kernel using sckit learn functionality
# ============================================================================

# Perform grid search with  
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid = param_grid, 
                        cv = cv_splits)

grid_rbf.fit(x_trn, y_trn)                    # find model weights

phis_mdl_final = grid_rbf.best_estimator_     # pick best model performance parameters 
y_pred_rbf = phis_mdl_final.predict(x_tst)    # For x test data predict classes with trained model

print("Rbf Results:")              # print results
print(grid_rbf.best_params_, grid_rbf.cv_results_, grid_rbf.best_estimator_)


# Evaluation Gaussian SVM performance 
# ===========================================================================

# Print Gaussian SVM kernel performance measures on test data set
fk_performance_printout(y_tst, y_pred_rbf, "SVM Gaussian kernel")

rbf_end_tm = datetime.now()
duration = rbf_end_tm - rbf_st_tm
print("Rbf duration: ", duration)

cx_val_sc = cross_val_score(phis_mdl_final, x_trn, y_trn, cv = 10)
print("Cros validation mean: ", cx_val_sc.mean())


#%% Polynomial Kernel using sckit learn functionality with various degress: 
    # investigated degress: [2, 10]
# ============================================================================

poly_st_tm = datetime.now()
print("Poly start time: ", poly_st_tm)            # print cell start time

# Perform grid search with  
dgr = 8
grid_ply = SVC(kernel = 'poly', degree = dgr)

grid_ply.fit(x_trn, y_trn)                       # find model fit and best params

# Testing with polynomial kernel
y_pred_poly = grid_ply.predict(x_tst)            # For x test data predict classes with trained model

print("\Poly Results:")                          # print results

# Evaluation Polynomial SVM performance 
# ===========================================================================

# Print Polynomial SVM kernel performance measures on test data set
fk_performance_printout(y_tst, y_pred_poly, "SVM Polynomial kernel")

poly_end_tm = datetime.now()
duration = poly_end_tm - poly_st_tm
print("Poly duration: ", duration)



#%% Feature/attribute optimization with best performing kernel
"""
In this section we attempt to remove less significant contributing features
by removing highly correlated attributes
The next 2 functions are modified versions from an online tutorial 
"""
# Auxiliary functions for feature selection returns array after attribute prunning
def remove_corr_attributes(x_tr, ct):
    c_trsh = ct
    p_corr = x_tr.corr()
    
    drop_feature = np.full(p_corr.shape[0], False, dtype=bool)
    for i in range(p_corr.shape[0]):
        for j in range(i + 1, p_corr.shape[0]):
            if abs(p_corr.iloc[i, j]) >= c_trsh:
                drop_feature[j] = True
    attributes_dropped = x_tr.columns[drop_feature]
    x_tr.drop(attributes_dropped, axis=1, inplace=True)
    return attributes_dropped
    
# Function returns array after attribute prunning 
def remove_less_significant_attributes(x_tr, y_tr):
    sl = 0.05
    reg_ols = None
    features_dropped = np.array([])
    for itr in range(0, len(x_tr.columns)):
        reg_ols = sm.OLS(y_tr, x_tr).fit()
        max_col = reg_ols.pvalues.idxmax()
        max_val = reg_ols.pvalues.max()
        if max_val > sl:
            x_tr.drop(max_col, axis='columns', inplace=True)
            features_dropped = np.append(features_dropped, [max_col])
        else:
            break
    reg_ols.summary()
    return features_dropped

# Reload and initialize functions
opt_st_tm = datetime.now()
print("Optimum start time: ", opt_st_tm)            # print cell start time

phis_data_ftr = pd.read_csv('Data/dataset.csv') 
y_full_ftr = phis_data_ftr.loc[:, 'Result']    # Pull out 'Result' column from data frame 
x_full_ftr = phis_data_ftr.iloc[:, 1:31]       # Drop 'Index' and 'Result' columns

#%% Model training with Feature selection, classification and Evaluation

# Prune attributes
cor_tresh = 0.75
remove_corr_attributes(x_full_ftr, cor_tresh)
remove_less_significant_attributes(x_full_ftr, y_full_ftr)

prd_x_ftr = x_full_ftr.to_numpy()
prd_x_ftr_b = np.append(np.ones([len(prd_x_ftr), 1]), prd_x_ftr, 1) # add a column of ones for b intercept 
out_y_ftr = y_full_ftr.to_numpy()

# Split attribute prunned data into Training and Testing sets
x_trn, x_tst, y_trn, y_tst = train_test_split(prd_x_ftr_b, out_y_ftr, test_size = 0.2,
                                              random_state = 42)

gamma_vals = np.logspace(-2, 2, 5)              # gamma values to test

c_vals = np.logspace(-2, 2, 5)                  # C values to test
param_grid = dict(gamma = gamma_vals, C = c_vals)

# split training data for trainng and cross validation purposes 
cv_splits = StratifiedShuffleSplit(n_splits = 10, test_size = 0.2, 
                                   random_state = 42)

#%% Gaussian rbf kernel using prunned features
# ============================================================================

opt_st_tm = datetime.now()
print("Rbf start time: ", opt_st_tm)            # print cell start time

# Perform grid search with  
grid_opt = GridSearchCV(SVC(kernel='rbf'), param_grid = param_grid, 
                        cv = cv_splits)

grid_opt.fit(x_trn, y_trn)                  # find classifier model weights

phis_mdl_opt = grid_opt.best_estimator_     # pick best model performance parameters 
y_pred_opt = phis_mdl_opt.predict(x_tst)    # For x test data predict classes with trained model

print("Optimum Results:")                   # print results
print(grid_opt.best_params_, grid_opt.cv_results_, grid_opt.best_estimator_)


# Evaluation Optimum Feature set and SVM kernel performance 
# ===========================================================================

# Print Optimum SVM performance measures on test data set
fk_performance_printout(y_tst, y_pred_opt, "SVM Gaussian kernel plus FSelect")

opt_end_tm = datetime.now()
duration = opt_end_tm - opt_st_tm
print("Optimum duration: ", duration)

#%% Linear kernel with pruned features 
#=============================================================================
# Parameters
reg = 10000
lrn_rt = 0.000001
mx_eps = 5000
cst_t = 0.001

ln_fs_st_tm = datetime.now()
print("Optimum start time: ", ln_fs_st_tm)            # print cell start time

# Train linear svm kernel and find weights
svm_Wts_fs = fk_linear_svm(lrn_rt, reg, mx_eps, cst_t, x_trn, y_trn)

# For all x test data predict class with trained model
y_test_predicted_fs = np.array([])
for i in range(x_tst.shape[0]):
    yp = np.sign(np.dot(x_tst[i], svm_Wts_fs))
    y_test_predicted_fs = np.append(y_test_predicted_fs, yp)  # Class predictions
    
# Lin SVM with pruned features performance assessment measures =============
# Print performance measures on test data set
fk_performance_printout(y_tst, y_test_predicted_fs, "SVM linear kernel plus FSelect")

ln_fs_end_tm = datetime.now()
duration = ln_fs_end_tm - ln_fs_st_tm
print("Linear FS duration: ", duration)

#%% Polynomial Kernel with feature selection: Degree = 8 
# ============================================================================

poly_st_tm_fs = datetime.now()
print("Poly plus feature selection start time: ", poly_st_tm_fs)            # print cell start time

# Perform grid search with  
dgr = 8
grid_ply_fs = SVC(kernel = 'poly', degree = dgr)

grid_ply_fs.fit(x_trn, y_trn)                   # find model fit and best params

# Testing with polynomial kernel
y_pred_poly_fs = grid_ply_fs.predict(x_tst)     # For x test data predict classes with trained model

# Evaluation Polynomial SVM plus feature performance 
# ===========================================================================
# Print Gaussian SVM kernel performance measures on test data set
fk_performance_printout(y_tst, y_pred_poly_fs, "SVM polynomial kernel plus FSelect")

poly_end_tm_fs = datetime.now()
duration = poly_end_tm_fs - poly_st_tm_fs
print("Poly duration: ", duration)

# ============================================================================
# END