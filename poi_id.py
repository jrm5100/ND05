#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


##########
###Most of the analysis was performed in the "Data Exploration" notebook.
###Only necessary final code was used here.
##########

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

#Remove "THE TRAVEL AGENCY IN THE PARK" since it isn't a person
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
#Remove "Total" row
data_dict.pop('TOTAL')

### Task 3: Create new feature(s)

financial_features = ['salary', 'deferral_payments', 'total_payments',
                      'loan_advances', 'bonus', 'restricted_stock_deferred',
                      'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive',
                      'restricted_stock', 'director_fees']

def fillzero(value):
    if value == 'NaN':
        return 0
    else:
        return value

for k,v in data_dict.items():
    v['from_poi_norm'] = float(v['from_poi_to_this_person'])/float(v['from_messages'])
    v['to_poi_norm'] = float(v['from_this_person_to_poi'])/float(v['to_messages'])
    v['shared_poi_norm'] = float(v['shared_receipt_with_poi'])/float(v['to_messages'])
    v['email_info_available'] = 1 if v['to_messages']!='NaN' else 0
    for ff in financial_features: #Replace NaN with 0 for financial features
        v[ff+"_fill0"] = fillzero(v[ff])

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#Use only created features, select some subset in individual classification pipelines
features_list = ['poi'] + [ff+'_fill0' for ff in financial_features] +\
                ['from_poi_norm', 'to_poi_norm', 'shared_poi_norm', 'email_info_available']


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Other classifiers run in the notebook file
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB

clf = make_pipeline(Imputer(),
                    StandardScaler(),
                    SelectKBest(k=7),
                    GaussianNB())


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
sss_test = StratifiedShuffleSplit(labels, 1000, test_size=0.3, random_state=42)
precision = cross_val_score(clf, features, labels, 'precision', sss_test)
recall = cross_val_score(clf, features, labels, 'recall', sss_test)
mean = lambda a: sum(a)/len(a)
print("Mean Precision = {:.4}".format(mean(precision)))
print("Mean Recall = {:.4}".format(mean(recall)))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)