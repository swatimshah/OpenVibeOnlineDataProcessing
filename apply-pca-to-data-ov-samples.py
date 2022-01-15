import numpy
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from matplotlib import pyplot
from pandas import DataFrame
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from numpy import savetxt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical 
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
import dill as pickle
from sklearn.pipeline import Pipeline
from numpy import asarray
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import learning_curve
from sklearn.ensemble import StackingClassifier
from keras.layers import Input
from keras.models import Model
from keras.losses import binary_crossentropy
from tensorflow.keras import regularizers
from numpy.random import seed
from tensorflow.random import set_seed
import tensorflow
from sklearn.decomposition import PCA


# setting the seed
seed(1)
set_seed(1)

# load data from spreadsheet
#X_train_whole = loadtxt('d:\\Atharva_3g-2o-flashes_1_to_12_152_small_mat_experiment_1.csv', delimiter=',')
#X_train_whole = loadtxt('d:\\flashes_1_to_12_152_small_mat_atharva_exp_2_1g_4o.csv', delimiter=',')
#X_train_whole = loadtxt('D:\\flashes_1_to_12_152_small_mat_neilay_experiment_2.csv', delimiter=',')
#X_train_whole = loadtxt('D:\\flashes_1_to_12_152_small_mat_hiren_1g_7o.csv', delimiter=',')
#X_train_whole = loadtxt('D:\\flashes_1_to_12_152_small_mat_1_to_45_swati_1g_7o.csv', delimiter=',')
#X_train_whole = loadtxt('D:\\flashes_1_to_12_152_small_mat_aditya_1_to_45_3g_4o.csv', delimiter=',')
#X_train_whole = loadtxt('D:\\flashes_1_to_12_152_small_mat_mugdha_1g_4o_1_to_45.csv', delimiter=',')
#X_train_whole = loadtxt('D:\\flashes_1_to_12_152_small_mat_ritu_6o_1_to_45.csv', delimiter=',')
#X_train_whole = loadtxt('D:\\flashes_1_to_12_152_small_mat_atharva_1_to_45_3g_2o.csv', delimiter=',')
X_train_whole = loadtxt('D:\\table_of_flashes_1_to_12_228_ov_training.csv', delimiter=',')



too_high_train = X_train_whole > 60.
X_train_whole[too_high_train] = 60.

too_low_train = X_train_whole < -60.
X_train_whole[too_low_train] = -60.


#Group all target samples
choiceTarget = X_train_whole[:, -1] == 1.
targetData = X_train_whole[choiceTarget, 0:228]
print(targetData.shape)

#Group all non-target samples
choiceNonTarget = X_train_whole[:, -1] == 0.
nonTargetData = X_train_whole[choiceNonTarget, 0:228]
print(nonTargetData.shape)


#Calculate average target vector
avg_data_vector_target = numpy.mean(targetData[:, 0:228], axis=0)
print(avg_data_vector_target.shape)


#Calculate average non-target vector
avg_data_vector_nonTargetData = numpy.mean(nonTargetData[:, 0:228], axis=0)
print(avg_data_vector_nonTargetData.shape)

avg_t_nt = numpy.mean(numpy.append(targetData, nonTargetData, axis=0), axis=0)

pca_target = PCA(n_components=228, svd_solver='auto')
#pca_target = PCA(0.95)
target_new = pca_target.fit(targetData)
print(pca_target.components_.shape)
print(pca_target.explained_variance_ratio_)
eigen_vector_target = pca_target.components_[0, 0:228]
eigen_vector_target_1 = pca_target.components_[1, 0:228]
eigen_vector_target_2 = pca_target.components_[2, 0:228]

pca_non_target = PCA(n_components=228, svd_solver='auto')
#pca_non_target = PCA(0.95)
non_target_new = pca_non_target.fit(nonTargetData)
print(pca_non_target.components_.shape)
print(pca_non_target.explained_variance_ratio_)
eigen_vector_nonTargetData = pca_non_target.components_[0, 0:228]
eigen_vector_nonTargetData_1 = pca_non_target.components_[1, 0:228]
eigen_vector_nonTargetData_2 = pca_non_target.components_[2, 0:228]

data_characteristics = numpy.empty([10, 228])

data_characteristics = numpy.append(numpy.asarray(avg_data_vector_target).reshape(-1, 228), numpy.asarray(avg_data_vector_nonTargetData).reshape(-1, 228), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(avg_t_nt), (-1, 228)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_target), (-1, 228)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_target_1), (-1, 228)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_target_2), (-1, 228)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_nonTargetData), (-1, 228)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_nonTargetData_1), (-1, 228)), axis=0)
data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(eigen_vector_nonTargetData_2), (-1, 228)), axis=0)

#data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(pca_target.explained_variance_ratio_), (-1, 10)), axis=0)
#data_characteristics = numpy.append(data_characteristics, numpy.reshape(numpy.asarray(pca_non_target.explained_variance_ratio_), (-1, 10)), axis=0)

savetxt('d:\\data_characteristics_ov_training.csv', data_characteristics, delimiter=',')

