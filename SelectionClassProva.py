import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier


def rf_function(X_train, X_val, y_train, y_val):
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }      
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=10)

    # Train the classifier
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Migliori parametri:", best_params)

    # Compute predictions and confusion matrix for validation
    predictions_val = best_model.predict(X_val)
    conf_matrix_val = confusion_matrix(y_val, predictions_val)
    class_names=y_val.unique()
    # Compute predictions and confusion matrix for train
    predictions_train = best_model.predict(X_train)
    conf_matrix_train = confusion_matrix(y_train, predictions_train)
    class_names=y_val.unique()

    return predictions_val, predictions_train, conf_matrix_val, conf_matrix_train, class_names

def svm_function(X_train, X_val, y_train, y_val):
    param_grid = {
    'kernel': ['poly', 'rbf'],   
    'C': [0.1, 1, 10],                                 
    }     
    svm = SVC()
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10)

    # Train the classifier
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Migliori parametri:", best_params)

    # Compute predictions and confusion matrix for validation
    predictions_val = best_model.predict(X_val)
    conf_matrix_val = confusion_matrix(y_val, predictions_val)
    class_names=y_val.unique()
    # Compute predictions and confusion matrix for train
    predictions_train = best_model.predict(X_train)
    conf_matrix_train = confusion_matrix(y_train, predictions_train)
    class_names=y_val.unique()

    return predictions_val, predictions_train, conf_matrix_val, conf_matrix_train, class_names



def plot_confMat(conf_matrix, class_names, name, n_features, classifier):
    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=25)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize = 20)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=60)
    ax.yaxis.set_ticklabels(class_names, fontsize = 25)
    plt.yticks(rotation=0)

    #plt.title(f'{name} Confusion Matrix', fontsize=60)

    plt.savefig(f'ConMat_{name}_{classifier}_{n_features}.png')
    plt.show()

    return ' '


def compute_accuracy(y, predictions):
    class_accuracies = {}
    for class_label in set(y):
        # Extract indices corresponding to the current class
        class_indices = (y == class_label)

        # Calculate accuracy for the current class
        class_accuracy = accuracy_score(y[class_indices], predictions[class_indices])

        # Store the accuracy
        class_accuracies[class_label] = class_accuracy

    # Print partial accuracies for each class
    print("Partial Accuracies for each class:")
    acc = 0
    for class_label, accuracy in class_accuracies.items():
        acc = acc + accuracy
        print(f"Class {class_label}: {accuracy:.4f}")
    acc = acc/5
    print("Total Accuracy:", acc)
    return acc






# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('dataset1.csv')


## Assuming the target variable is in a column named 'label'


data['label'] = data['label'].replace({'Sadness': 'Sad', 'Surprised': 'Surprise', 'Anger':'Angry', 'Happy': 'Happyness'})
keep = [ 'Anger', 'Happyness', 'Sad', 'Disgust', 'Neutral', 'Surprise']
X = data[data['label'].isin(keep)]
print("\n\n\ndimensione dataset:", X.shape)


#
y = X['label']

X = X.drop(['name', 'label'], axis=1)



n_features = 25
knn = KNeighborsClassifier(n_neighbors=10)
sfs = SequentialFeatureSelector(knn, n_features_to_select=n_features)
sfs.fit(X, y)
sfs.get_support()
X = sfs.transform(X)


#Get Class Labels
class_names=y.unique()
print(class_names)
counts = pd.Series(y).value_counts()
percentages = counts / counts.sum() * 100
print(percentages)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\n\n\nprova dataset features extraction:", X)





#svm
classifier = 'svm'
# Initialize SVM classifier
predictions_val_svm, predictions_train_svm, conf_matrix_val_svm, conf_matrix_train_svm, class_names = svm_function(X_train, X_val, y_train, y_val)
plot_confMat(conf_matrix_train_svm, class_names, 'train', n_features, classifier)
plot_confMat(conf_matrix_val_svm, class_names, 'validation', n_features, classifier)
acc_train = compute_accuracy(y_train, predictions_train_svm)
acc_val = compute_accuracy(y_val, predictions_val_svm)





#random forest 
classifier = 'rf'
predictions_val_rf, predictions_train_rf, conf_matrix_val_rf, conf_matrix_train_rf, class_names = rf_function(X_train, X_val, y_train, y_val)
plot_confMat(conf_matrix_train_rf, class_names, 'train', n_features, classifier)
plot_confMat(conf_matrix_val_rf, class_names, 'validation', n_features, classifier)
acc_train = compute_accuracy(y_train, predictions_train_rf)
acc_val = compute_accuracy(y_val, predictions_val_rf)






















