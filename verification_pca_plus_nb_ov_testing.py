from sklearn import preprocessing
from numpy import loadtxt
import numpy
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from numpy import savetxt
import keras.backend as K
from tensorflow.python.keras.backend import eager_learning_phase_scope
from numpy import mean
import pickle


MY_CONST = 60.
MY_CONST_NEG = -60.

def NormalizeData(data):
    return (data + (MY_CONST)) / (MY_CONST - (MY_CONST_NEG))

model_file = open ("LDA_model_ov_testing.sav", "rb")
model = pickle.load(model_file)

X = loadtxt('d:\\table_of_flashes_1_to_12_228_ov_testing.csv', delimiter=',')

mean_of_test = mean(X[:, 0:228])
print(mean_of_test)
input = X[:, 0:228] - mean_of_test
too_high_input = input > MY_CONST
input[too_high_input] = MY_CONST
too_low_input = input < MY_CONST_NEG
input[too_low_input] = MY_CONST_NEG
input = NormalizeData(input)
savetxt('d:\\input-swati-online.csv', input, delimiter=',')

y_real = X[:, -1]

y_pred = model.predict(input) 
matrix = confusion_matrix(y_real, y_pred)
print(matrix)
