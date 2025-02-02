# 🚀 SpaceX-AppliedDS
## Machine Learning Predictions
Create a machine learning pipeline to predict if the first stage will land given the data from the preceding labs.

## ❗Objectives
- Perform Exploratory Data Analysis and determine Training Labels
- Find best Hyperparameter for SVM, Classification Trees, and Logistic Regression.
- Find the method that performs best with test data.

## Import Libraries
```python
import piplite
await piplite.install(['numpy'])
await piplite.install(['pandas'])
await piplite.install(['seaborn'])
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
```
The code below is given through the IBM lab and will be the function used to plot the confusion matrix.
```python
def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed']) 
    plt.show()
```
## Load the Dataframe
```python
from js import fetch
import io

URL1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
resp1 = await fetch(URL1)
text1 = io.BytesIO((await resp1.arrayBuffer()).to_py())
data = pd.read_csv(text1)
```
```python
data.head()
```

<img width="905" alt="Screenshot 2025-02-02 at 4 35 05 PM" src="https://github.com/user-attachments/assets/81cac091-a528-4df2-bb54-1f85c9f539b2" />

```python
URL2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv'
resp2 = await fetch(URL2)
text2 = io.BytesIO((await resp2.arrayBuffer()).to_py())
X = pd.read_csv(text2)
```
```python
X.head(100)
```

<img width="903" alt="Screenshot 2025-02-02 at 4 35 51 PM" src="https://github.com/user-attachments/assets/e07d80a4-ddeb-413b-ad54-d2e7e00133ef" />

## Exploratory Data Analysis

### Task 1
Create a NumPy array from the column `Class` in `data`, by applying the method `to_numpy()` then assign it to the variable `Y`,make sure the output is a Pandas series (only one bracket df['name of column']).
```python
Y = pd.Series(data['Class'].to_numpy())
Y.head(10)
```

<img width="142" alt="Screenshot 2025-02-02 at 4 37 33 PM" src="https://github.com/user-attachments/assets/5e16b80a-9d0a-4e8d-8a33-c1b37a74fdcd" />

### Task 2

Standardize the data in `X` then reassign it to the variable  `X` using the transform provided below.
```python
transform = preprocessing.StandardScaler()
X = transform.fit(X).transform(X)
```
We split the data into training and testing data using the function  `train_test_split`. The training data is divided into validation data, a second set used for training data; then the models are trained and hyperparameters are selected using the function `GridSearchCV`.

### Task 3
Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to 0.2 and random_state to 2. The training data and test data should be assigned to the following labels.

`X_train`, `X_test`, `Y_train`, `Y_test`
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
```
```python
Y_test.shape
```

<img width="67" alt="Screenshot 2025-02-02 at 4 41 24 PM" src="https://github.com/user-attachments/assets/cae7a878-6bdb-48fa-a9a6-5218587d65a3" />

There are only 18 samples.

### Task 4
Create a logistic regression object then create a GridSearchCV object  logreg_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
```python
parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}
```
```python
parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()

logreg_cv=GridSearchCV(lr, parameters, cv=10)
logreg_cv.fit(X_train, Y_train)
```
```python
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
```

<img width="709" alt="Screenshot 2025-02-02 at 4 43 22 PM" src="https://github.com/user-attachments/assets/7e903132-e5dd-4fd9-869f-c88357d4b376" />

### Task 5
Calculate the accuracy on the test data using the method `score`:
```python
accuracy = logreg_cv.score(X_test, Y_test)
accuracy
```

<img width="163" alt="Screenshot 2025-02-02 at 4 44 12 PM" src="https://github.com/user-attachments/assets/dc33d204-d0a7-4e25-9d2f-3c0ac3c2c18f" />

Lets look at the confusion matrix:
```python
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

<img width="562" alt="Screenshot 2025-02-02 at 4 44 56 PM" src="https://github.com/user-attachments/assets/894da88e-ac33-49bb-bf91-0747db10b576" />

Logistic regression can distinguish between the different classes. 

True positive - 12

False positive - 3

### Task 6
Create a support vector machine object then create a  `GridSearchCV` object  `svm_cv` with cv = 10. Fit the object to find the best parameters from the dictionary `parameters`.
```python
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
```
```python
svm_cv = GridSearchCV(svm, parameters, cv=10)
svm_cv.fit(X_train, Y_train)
```
```python
print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)
```

<img width="825" alt="Screenshot 2025-02-02 at 4 48 04 PM" src="https://github.com/user-attachments/assets/184688e1-7bb9-47a2-a737-cf5cf4579ecb" />

### Task 7
Calculate the accuracy on the test data using the method `score`:
```python
svm_accuracy = svm_cv.score(X_test, Y_test)
svm_accuracy
```

<img width="228" alt="Screenshot 2025-02-02 at 4 49 01 PM" src="https://github.com/user-attachments/assets/ddbfb629-62b8-4d9e-939d-41dbc54b8176" />

Lets look at the confusion matrix:
```python
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

<img width="545" alt="Screenshot 2025-02-02 at 4 49 38 PM" src="https://github.com/user-attachments/assets/83c48293-6e07-4979-aa08-fe7426895e91" />

### Task 8
Create a decision tree classifier object then create a  `GridSearchCV` object  `tree_cv` with cv = 10. Fit the object to find the best parameters from the dictionary `parameters`.
```python
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()
```
```python
tree_cv = GridSearchCV(tree, parameters, cv=10)
tree_cv.fit(X_train, Y_train)
```
```python
print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)
```

<img width="925" alt="Screenshot 2025-02-02 at 4 51 19 PM" src="https://github.com/user-attachments/assets/e7a06dd4-d9fe-4067-8df5-87cfb2d1e7fb" />

### Task 9
Calculate the accuracy on the test data using the method `score`:
```python
tree_accuracy = tree_cv.score(X_test, Y_test)
tree_accuracy
```

<img width="184" alt="Screenshot 2025-02-02 at 4 52 08 PM" src="https://github.com/user-attachments/assets/a640530c-8b6a-4dbd-8a47-849bb681e9bc" />

We can plot the confusion matrix

```python
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

<img width="560" alt="Screenshot 2025-02-02 at 4 52 56 PM" src="https://github.com/user-attachments/assets/cdd4b524-120d-4d49-afa2-c02631d61075" />

### Task 10
Create a k nearest neighbors object then create a  `GridSearchCV` object  `knn_cv` with cv = 10. Fit the object to find the best parameters from the dictionary `parameters`.
```python
parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()
```
```python
knn_cv = GridSearchCV(KNN, parameters, cv=10)
knn_cv.fit(X_train, Y_train)
```
```python
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)
```

<img width="749" alt="Screenshot 2025-02-02 at 4 54 21 PM" src="https://github.com/user-attachments/assets/604d0a08-047e-4e55-934a-c845968c4020" />

### Task 11
Calculate the accuracy on the test data using the method `score`:
```python
knn_accuracy = knn_cv.score(X_test, Y_test)
knn_accuracy
```

<img width="200" alt="Screenshot 2025-02-02 at 4 54 57 PM" src="https://github.com/user-attachments/assets/5c81dfdb-e367-49dd-9f67-34df3544c576" />

We can plot the confusion matrix
```python
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
```

<img width="539" alt="Screenshot 2025-02-02 at 4 55 27 PM" src="https://github.com/user-attachments/assets/68943299-cbed-4dd3-833d-ce32f77cf166" />

### Task 12
Find the method that performs best.
- There is no one particular method that performs best, based on the test scores.
- The small sample size can have an impact on same test scores.
- Decision tree model best model based on the scores on the whole dataset due to having the highest accuracy.

## Further Conclusions
- SpaceX Falcon 9 landings have improved. As more launches are made the better the performance and greater the success.
- Machine learning models can be used to predict future SpaceX Falcon 9 landings. 
***




