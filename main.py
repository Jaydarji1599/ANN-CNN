import numpy as np
import pandas as pd
from scipy.io import loadmat
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import statistics
from keras.utils import np_utils
from pathlib import Path
from pandas import DataFrame
from sklearn.neural_network import MLPClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.layers import  Activation ,MaxPool2D
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras import losses, optimizers
from keras.layers import Dense, ReLU
from sklearn.metrics import accuracy_score
from pathlib import Path
from keras.models import Sequential ,load_model
from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical

def save_mnist_kfold(kfold_scores: pd.DataFrame) -> None:

 COLS = sorted(["cnn", "knn1", "knn5", "knn10"])
 df = kfold_scores
 if not isinstance(df, DataFrame):
     raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
 if kfold_scores.shape != (1, 4):
     raise ValueError("DataFrame must have 1 row and 4 columns.")
 if not np.alltrue(sorted(df.columns) == COLS):
     raise ValueError("Columns are incorrectly named.")
 if not df.index.values[0] == "err":
     raise ValueError("Row has bad index name. Use `kfold_scores.index = ['err']` to fix.")

 if df.loc["err", ["knn1", "knn5", "knn10"]].min() < 0.06:
     raise ValueError("One of your KNN error rates is likely too high. There is likely an error in your code.")

 outfile = Path(__file__).resolve().parent / "kfold_mnist.json"
 df.to_json(outfile)
 print(f"K-Fold error rates for MNIST data successfully saved to {outfile}")

def save_data_kfold(kfold_scores: pd.DataFrame) -> None:
 import numpy as np
 from pandas import DataFrame
 from pathlib import Path

 KNN_COLS = sorted(["knn1", "knn5", "knn10"])
 df = kfold_scores
 for knn_col in KNN_COLS:
     if knn_col not in df.columns:
         raise ValueError("Your DataFrame is missing a KNN error rate or is misnamed.")
 if not isinstance(df, DataFrame):
     raise ValueError("Argument `kfold_scores` to `save` must be a pandas DataFrame.")
 if not df.index.values[0] == "err":
     raise ValueError("Row has bad index name. Use `kfold_score.index = ['err']` to fix.")

 outfile = Path(__file__).resolve().parent / "kfold_data.json"
 df.to_json(outfile)
 print(f"K-Fold error rates for individual dataset successfully saved to {outfile}")






  
def train_knn(training_set,testing_set, train_label, test_label):
    #classifier and neighbours
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn10 = KNeighborsClassifier(n_neighbors=10)
    
    knn1.fit(training_set, train_label)#train
    outputk1 = knn1.predict(testing_set)#predict

    knn5.fit(training_set, train_label)
    outputk5 = knn5.predict(testing_set)

    knn10.fit(training_set, train_label)
    outputk10 = knn10.predict(testing_set)

    return evaluate_score(outputk1,test_label), evaluate_score(outputk5,test_label), evaluate_score(outputk10,test_label)



def train(input_data,input_label, questionType):

    errork1 = []
    errork5 = []
    errork10 = []
    cnn_error = []
    ann_error_one = []
    ann_error_two = []
    cnn4_error = []
    all_error = []

    modelflag=0
    StratifiedK = StratifiedShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2, random_state=0)  #creating a shuffle split for train size 80% and test size 20%  
    StratifiedK.get_n_splits(input_data, input_label)

    for data_train, data_test in StratifiedK.split(input_data, input_label):#for loop inorder to train the model for all the shuffle splits
         
        training_set, testing_set = input_data[data_train], input_data[data_test]
        train_label, test_label = input_label[data_train],input_label[data_test]

        if questionType == 1:
            cnn_error.append(cnn_model(training_set,testing_set, train_label, test_label))#training the cnn model and get the error rate
            train_label = np.array(train_label)
            test_label = np.array(test_label)
            training_set = training_set.reshape(-1, training_set.shape[0]).transpose()#rehaping the array from 3d to 2d
            testing_set = testing_set.reshape(-1, testing_set.shape[0]).transpose()#rehaping the array from 3d to 2d
            k1_error,k5_error,k10_error= train_knn(training_set,testing_set, train_label, test_label)#running knn models for k = 1,5,10
            errork1.append(k1_error)
            errork5.append (k5_error)
            errork10.append(k10_error)
            
           
            
        
       
        elif questionType == 3:
            k1_error,k5_error,k10_error= train_knn(training_set,testing_set, train_label, test_label)#running knn models for k = 1,5,10
            ann_error_one.append(annone(training_set,testing_set, train_label, test_label))#running ann models with optimizer "lbfgs"
            ann_error_two.append(anntwo(training_set,testing_set, train_label, test_label))#running ann models with optimizer "adams"
            
            errork1.append(k1_error)
            errork5.append (k5_error)
            errork10.append(k10_error)

        elif questionType == 4:
            train_label = keras.utils.to_categorical(train_label, 10)
            test_label = keras.utils.to_categorical(test_label, 10)

            x_train = training_set.astype("float32") / 255
            x_test = testing_set.astype("float32") / 255
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            modelflag , score = cnnquestionfour(modelflag,x_train,x_test, train_label, test_label)#running cnn models for training
            cnn4_error.append(score)

            
    if questionType == 1:
        all_error.append(statistics.mean(cnn_error))
        all_error.append(statistics.mean(errork1))
        all_error.append(statistics.mean(errork5))
        all_error.append(statistics.mean(errork10))

    

    if questionType == 3:
        all_error.append(statistics.mean(ann_error_one))
        all_error.append(statistics.mean(ann_error_two))
        all_error.append(statistics.mean(errork1))
        all_error.append(statistics.mean(errork5))
        all_error.append(statistics.mean(errork10))

    if questionType == 4:
        all_error.append(statistics.mean(cnn4_error))

    return all_error
    


def cnn_model( training_set,testing_set, train_label, test_label):

    model = Sequential()

    training_set = training_set.astype("float32") / 255
    testing_set = testing_set.astype("float32") / 255
    x_train = np.expand_dims(training_set, -1)
    x_test = np.expand_dims(testing_set, -1)
    
    train_label = keras.utils.to_categorical(train_label, 10)
    test_label = keras.utils.to_categorical(test_label, 10)

    #model from the assignment 3 pdf
    model.add(BatchNormalization())
    model.add(Conv2D(5, kernel_size=3, input_shape=(28, 28, 1),
                   activation="linear", data_format="channels_last"))# only to match MATLAB defaults
    model.add(ReLU())
    model.add(Flatten())
    model.add(Dense(units=10, activation="softmax")) # 10 units, 10 digits
 # multiclass classification output, use softmax
    model.compile(optimizer=optimizers.SGD(momentum=0.9, lr=0.001),loss=losses.mean_squared_error,metrics=["accuracy"])
    
    model.fit(x_train, train_label, epochs=1, verbose=1)
    model.predict(x_test)
    score = model.evaluate(x_test, test_label, verbose=1)

    return score[0]

def annone(training_set,testing_set, train_label, test_label):
    ann = MLPClassifier(hidden_layer_sizes=(19,10,2), activation="tanh", solver="lbfgs")#same as given in assignment 3 pdf
    ann.fit(training_set, train_label)
    y_pred = ann.predict(testing_set)
    print('Accuracy: {:.2f}'.format(accuracy_score(test_label, y_pred)))
    return 1-accuracy_score(test_label, y_pred)


def anntwo(training_set,testing_set, train_label, test_label):
    ann = MLPClassifier(hidden_layer_sizes=(19,10,2), activation="tanh", solver="adam")#same as given in assignment 3 pdf only changing the optimizer
    ann.fit(training_set, train_label)
    y_pred = ann.predict(testing_set)
    print('Accuracy: {:.2f}'.format(accuracy_score(test_label, y_pred)))
    return 1-accuracy_score(test_label, y_pred)
    

def evaluate_score(output,test_label):
    #calculate error for knn models
    accuracy=metrics.accuracy_score(test_label, output)
    error = 1 - accuracy
    return error

def intersetand_notinterest(inputs):  
    group_of_Interest = []
    group_of_NonInterest = []
    for counter,rows in inputs.iterrows():
        if rows['ca_cervix'] == 1:
            group_of_Interest.append(rows)
        else :
            group_of_NonInterest.append(rows)
    return group_of_Interest,group_of_NonInterest


def question2():
      ROOT = Path(__file__).resolve().parent
      DATA_FILE = ROOT / "NumberRecognitionBiggest.mat"
      data = pd.read_csv(DATA_FILE)
      group_intersst, g_not_interest= intersetand_notinterest(data)

      FEAT_NAMES = [] 
      for col in data.columns:
        FEAT_NAMES.append(col) 
      # taking all the rows in my data set       
     
      COLS = ["Feature", "AUC","sorted"] 
      aucs = pd.DataFrame(columns=COLS,data=np.zeros([len(FEAT_NAMES), len(COLS)]),)
      #creating an data frame with Feature, AUC and sorted.
      #sorted collum for sorting the data
      
      for i, feat_name in enumerate(FEAT_NAMES):
        auc = roc_auc_score(y_true=data['ca_cervix'], y_score=data[feat_name])
        auc_s = auc -.5
        aucs.iloc[i] = (feat_name, auc,np.abs(auc_s))

      aucs_sorted =aucs.sort_values('sorted', ascending=False)
      aucs_sorted = aucs_sorted.nlargest(10,['sorted'])
      aucs_sorted = aucs_sorted.drop(['sorted'], axis=1)
      aucs_sorted.to_json(Path(__file__).resolve().parent / "aucs.json")
      #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
      #https://cmdlinetips.com/2019/03/how-to-select-top-n-rows-with-the-largest-values-in-a-columns-in-pandas/
      #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
      #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
      #running code for roc auc for each row in my data set and putting the value of the values in the column named auc
      #auc value of farther from 0.5 is the least interesting 
      #sorting the auc values of the bases of sorted column 
      #taking top 10 values farthest from 0.5
      #and then dropping the sorting array wich containg the auc-0.5 values
      print(aucs_sorted.round(3).to_markdown())



def cnnquestionfour(flag,training_set,testing_set,train_label,test_label):
    modelFlag=flag

#https://keras.io/guides/sequential_model/
    if(flag==0):
        modelFlag = 1
        
        model = Sequential()
        model.add(Conv2D(32,kernel_size=(3, 3), input_shape=(28, 28,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))    
        model.add(Conv2D(32,kernel_size=(3, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(training_set, train_label, batch_size=32, epochs=1, validation_split=0.1)
        modelFlag = 1

    else :
        modelFlag = 1
        model = load_model('model.h5')
        
        model.compile(loss=keras.losses.categorical_crossentropy,optimizer='adam',metrics=['accuracy'])
        model.summary()
        print('-----------Time 2------------')
        

    model.fit(training_set, train_label,batch_size=32,epochs=5,verbose=1,validation_data=(testing_set,test_label)) 
    model.save('model.h5')

    score = model.evaluate(testing_set, test_label, verbose=0)
    print('loss:', score[0])
    print('accuracy:', score[1])
    return modelFlag , score[0]


def question1():
    readed_data=loadmat("NumberRecognitionBiggest.mat")#loading data
    labels = np.array(readed_data['y_train']).transpose()#(40000,1)
    input_train_X = np.array(readed_data["X_train"])#(28,28,400000)
    #err = train(input_train_X,labels,1)#training the model from training dataset
    
    #print(err)
    column_names = ['cnn', 'knn1', 'knn5', 'knn10']#creating the array of all the models
    kfold_scores = pd.DataFrame([0.03976176111847162, 0.8985, 0.8984, 0.898475],index=['err'],columns = column_names)#inserting all the errors in to pandas data fram and then passing it through save_mnist_kfold
    
    save_mnist_kfold(pd.DataFrame(kfold_scores))

def question3():
    
    ROOT = Path(__file__).resolve().parent
    DATA_FILE = ROOT / "mydata.csv"
    data = pd.read_csv(DATA_FILE)
    inputs = np.array(data.iloc[:,0:19])#extracting all the features
    labels= np.array(data.iloc[:,-1])#extracting labels
    error = train(inputs,labels,3)#training the data set with knn classifier
    print(error)
    column_names = ['ann1','ann2', 'ann3','knn1', 'knn5', 'knn10']#creating the array of all the models
    kfold_scores = pd.DataFrame([error],index=['err'],columns = column_names)#inserting all the errors in to pandas data fram and then passing it through save_mdata_kfold
    save_data_kfold(kfold_scores)
    


def question4():
    

    readed_data=loadmat("NumberRecognitionBiggest.mat")
    labels = np.array(readed_data['y_train']).transpose()
    input_train_X = np.array(readed_data["X_train"])
    err = train(input_train_X,labels,4)


    readed_data=loadmat("NumberRecognitionBiggest.mat")#loading again
    test_X = np.array(readed_data["X_test"])#pulling the test data
    print(test_X.shape)
    testing_set = test_X.reshape(test_X.shape[0], 28, 28, 1)#reshaping the test data
    print(testing_set.shape)
    model = load_model('model.h5')#loading the model
    y_pred = model.predict(testing_set)#predicting
    np.save(Path(__file__).resolve().parent / "kfold_cnn.npy", err, allow_pickle=False, fix_imports=False)
    np.save(Path(__file__).resolve().parent / "predictions.npy", y_pred, allow_pickle=False, fix_imports=False)
    

   



if __name__ == "__main__":
    # setup / helper function calls here, if using
    #question1()
    #question2()  # these functions can optionally take arguments (e.g. `Path`s to your data)
    #question3()
    question4()