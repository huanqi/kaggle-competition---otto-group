import pandas as pd
import numpy as np
from sklearn import feature_extraction, ensemble, preprocessing
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.cross_validation import StratifiedShuffleSplit


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


# set up the xgboost classifier
xg_train = xgb.DMatrix(train_x, label = train_y)
xg_fulltrain = xgb.DMatrix(train, label = target)
xg_test = xgb.DMatrix(test_x, label = test_y)
test_temp_label = np.ndarray(shape =(test.shape[0], ), dtype = float)
xg_result = xgb.DMatrix(test, label = test_temp_label)

trainlist = [(xg_train, 'train')]
evallist = [(xg_text, 'eval')]

# setup the parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softprob'
# scale weight of positive examples
param['eta'] = 0.02
param['max_depth'] = 15
param['silent'] = 1
param['eval_metric'] = 'mlogloss'
param['nthread'] = 4
param['min_child_weight'] = 2
param['num_class'] = 9
num_round = 20

bst = xgb.train(param, xg_train, num_round, trainlist)
yprob = bst.predict( xg_test)
log_loss_score = log_loss(test_y, yprob)
print ('xgboost LogLoss score: ', log_loss_score)


# create the results file
#predict = bst.predict(xg_result)
#predict = pd.DataFrame(predict, index = sample.id.values, columns = sample.columns[1:])
#predict.to_csv('xgboost_benchmark.csv', index_label = 'id')





