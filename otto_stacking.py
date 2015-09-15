import pandas as pd
import numpy as np
from sklearn import feature_extraction
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy.optimize import minimize

# load the data
train = pd.read_csv("train.csv", index_col = 'id')
test = pd.read_csv("test.csv", index_col = 'id')
sample = pd.read_csv("sampleSubmission.csv")

# prepare the train and target value
target = train.target.values
train = train.drop('target', axis = 1)

# we need a independent test set to find the best weights to
# combine classifiers
sss = StratifiedShuffleSplit(target, test_size = 0.05, random_state = 1234)
for train_index, test_index in sss:
  break
train_x, train_y = train.values[train_index], target[train_index]
test_x, test_y = train.values[test_index], target[test_index]

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train_x = tfidf.fit_transform(train_x).toarray()
test_x = tfidf.fit_transform(test_x).toarray()

# transform the target into classes
label_enc = preprocessing.LabelEncoder()
train_y = label_enc.fit_transform(train_y)
test_y = label_enc.fit_transform(test_y)

# set up the random forest classifier and make prediction
rfc = ensemble.RandomForestClassifier(n_estimators = 20, random_state = 4141)
rfc.fit(train_x, train_y)
score = log_loss(test_y, rfc.predict_proba(test_x))
print('RFC LogLoss score: ', score)

# set up the logistic regression classifier and make prediction
logreg = LogisticRegression()
logreg.fit(train_x, train_y)
score = log_loss(test_y, logreg.predict_proba(test_x))
print('LogisticRegression LogLoss score: ', score)


# looking for the optimum weights to combine two classifier
clfs = []
clfs.append(rfc)
clfs.append(logreg)
predictions = []
for clf in clfs:
  predictions.append(clf.predict_proba(test_x))
#  print 'the predictions size: ', len(predictions)
  
def log_loss_func(weights):
  final_prediction = 0
  for weight, prediction in zip(weights, predictions):
#    print weight, prediction.shape
    final_prediction += weight*prediction
  return log_loss(test_y, final_prediction)
  
starting_values = [0.5]*len(predictions)
bounds = [(0, 1)]*len(predictions)
res = minimize(log_loss_func, starting_values, method = 'SLSQP', bounds = bounds)

print 'Ensamble score: ', res['fun']
print 'Best weights: ', res['x']


# create the results file
final_prediction = 0
predictions = []
for clf in clfs:
  predictions.append(clf.predict_proba(test))
  
for weight, prediction in zip(res['x'], predictions):
  final_prediction += weight*prediction
predict = pd.DataFrame(final_prediction, index = sample.id.values, columns = sample.columns[1:])
predict.to_csv('rdf_benchmark.csv', index_label = 'id')





