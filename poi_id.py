#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','exercised_stock_options','other','expenses', 
                 'ratio email to poi','shared_receipt_with_poi','total_stock_value',
             'total_payments','bonus','restricted_stock'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# To load the dataset into dataframe format for easier processing.    
df_data = pd.DataFrame(data_dict)
df_data = df_data.transpose().reset_index()    
    
### Task 2: Remove outliers
df_data = df_data[df_data['index'] != 'TOTAL']  # remove the Total row
df_data = df_data.set_index('index') # make name to index

### Task 3: Create new feature(s)

df_data.replace('NaN',np.nan,inplace=True)
df_data.fillna(0,inplace=True)
df_data = df_data.replace([np.inf,-np.inf],0)

df_data['ratio email from poi'] = df_data['from_poi_to_this_person'] / df_data['to_messages']
df_data['ratio email to poi'] = df_data['from_this_person_to_poi'] / df_data['from_messages']
df_data['ratio of total payment vs bonus'] = df_data['bonus'] / df_data['total_payments']
df_data['ratio of total stock vs exercised stock'] = df_data['exercised_stock_options'] / df_data['total_stock_value']

df_data.fillna(0,inplace=True)
df_data = df_data.replace([np.inf,-np.inf],0)

### Store to my_dataset for easy export below.
my_dataset = df_data.to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, auc, make_scorer, precision_score, recall_score

clf = DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = DecisionTreeClassifier(random_state=42, criterion='entropy', min_samples_split=17,splitter='best',max_depth=4, min_samples_leaf=4)


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

clf = clf.fit(features_train, labels_train)    

pred= clf.predict(features_test)

acc=accuracy_score(labels_test, pred)
print('accuracy score: ', acc)

precision=precision_score(labels_test, pred)
print('Precision score: ', precision)

recall = recall_score(labels_test, pred)
print('Recall score: ', recall)

f1 = f1_score(labels_test, pred)
print('F1 score: ', f1)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)