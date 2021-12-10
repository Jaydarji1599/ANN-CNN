# Question 1


## a    According to the error rates KNN10 out performs all the other classifiers.

## b    According to my thinking The CNN model isnt performing well because the model isnt designed well enough. The parameters that we are taking in this model are basic so model is struggling to extract enoguh information from the traning data. Also CNN is used for images and in this question we are using CNN for mnist data set.The CNN designed with better and appropriate parameters and including many layers would perform better. KNN1 and KNN5 isnt performing well because of the the overfitting and under fitting of the data. Although KNN1 and KNN5 is able to form more precise boundries around the 10 classes, the KNN1 and KNN5 is overfitting the data. Knn1 the error rates are very low and its not reasonable because the machine is overfitting that is it is learned too much of just one class. k=10 is the best and most reliable model because the chances of overfitting is the least when compared to k=1 and k=5.

# Question 2
 
## Data Discription- The data consist of behaviour of people and corresponding risk to develop cancer. The data has 19 attributes, meaning 19 behaviour and 76 instances meaning 76 samples. The data is used for classification and which behaviour has the highest risk of having cancer. The count of group of interests are 21 and the count of group not of interests are 51.

## AUC Values
|    | Feature                    |   AUC |
|---:|:---------------------------|------:|
| 17 | empowerment_abilities      | 0.17  |
| 16 | empowerment_knowledge      | 0.195 |
| 10 | perception_severity        | 0.213 |
| 18 | empowerment_desires        | 0.226 |
|  9 | perception_vulnerability   | 0.248 |
|  8 | norm_fulfillment           | 0.25  |
| 13 | socialSupport_emotionality | 0.254 |
| 12 | motivation_willingness     | 0.267 |
|  2 | behavior_personalHygine    | 0.282 |
|  3 | intention_aggregation      | 0.298 |

# Question 3 

## ANN with ADAM optimiser is the best performing. As ADAM optimiser can provide optimise algorithm that can handle noisy problems it is best performing. As for my dataset, the dataset oitself is too small and it only has 2 classes, ADAM is more reliable as it does a good job in seperating two classes.

## Question 4

#   loss: 0.03211081117466915
accuracy: 0.9897500276565552
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 32)        9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 32)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 800)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               102528    
_________________________________________________________________
activation_1 (Activation)    (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1290      
_________________________________________________________________
activation_2 (Activation)    (None, 10)                0         
=================================================================
Total params: 113,386
Trainable params: 113,386
Non-trainable params: 0


## my model was taking too much time for running so i wasnt able to genereate some of the files. i am really sorry if i get a chance i would like to submit the files 