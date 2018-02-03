import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# check missing values per column
# train.isnull().sum(axis=0)/train.shape[0]

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)

# set datatime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute

cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')

cat_cols = cols + ['countrycode','browserid','devid']

for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values) + list(test[col].values))
    train[col] = lbl.transform(list(train[col].values))
    test[col] = lbl.transform(list(test[col].values))

cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

X_train, X_test, y_train, y_test = train_test_split(train[cols_to_use], train['click'], test_size = 0.5)



gbm = xgb.XGBClassifier(max_depth=10, n_estimators=300, learning_rate=0.03).fit(X_train, y_train)
# clf = lgb.train(params, dtrain,num_boost_round=500,valid_sets=dval,verbose_eval=20)

y_pred = gbm.predict(X_test)
auc = accuracy_score(y_test, y_pred)
print(auc)

preds = gbm.predict(test[cols_to_use])

sub = pd.DataFrame({'ID':test['ID'], 'click':preds})
sub.to_csv('xgb_pyst.csv', index=False)