from keras.models import Sequential
from sklearn import datasets
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pickle
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from keras.losses import sparse_categorical_crossentropy

def classifier(X, y):
    """
    Description of classifier
    """
    NOF_ROW, NOF_COL =  X.shape

    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(12, kernel_initializer='random_normal', input_dim=NOF_COL, activation='relu'))
        model.add(Dense(6, kernel_initializer='random_normal', activation='relu'))
        model.add(Dense(3, kernel_initializer='random_normal', activation='softmax'))
        # Compile model
        model.compile(loss=sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
        return model

    # evaluate using 10-fold cross validation
    seed = 7
    np.random.seed(seed)
    model = KerasClassifier(build_fn=create_model, epochs=300, batch_size=32, verbose=1)
    return model


def main():
    """
    Description of main
    """
    print("loading iris..")
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X = preprocessing.scale(X)
    print(X.shape)	

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    model_tt = classifier(X_train, y_train)
    print("Got the model")	
    model_tt.fit(X_train,y_train)
    print("Fitted the model")	

    #kfold = KFold(n_splits=10, shuffle=True) 
    #results = cross_val_score(model_tt, X_test[0:25,:], y_test[0:25], cv=kfold)
    #print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    #--------------------------------------------------
    # This works OK 
    #-------------------------------------------------- 
    iris_score = model_tt.score(X_test, y_test)	
    print (iris_score)	
    iris_prob = model_tt.predict_proba(X_test)
    print(iris_prob)		
    predictions = model_tt.predict(X_test[0:25])
    print(predictions)	

    matrix = confusion_matrix(y_test[0:25], np.argmax(iris_prob[0:25], axis=1))
    print(matrix)

    unseen_predictions = model_tt.predict_proba(X_test[25:40])
    	
    matrix = confusion_matrix(y_test[25:40], np.argmax(unseen_predictions, axis=1))
    print(matrix)
    


main()