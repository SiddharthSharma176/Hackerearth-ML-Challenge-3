import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#test1 = pd.read_csv("test.csv")
# train = train[0:10]
# test = test[0:10]

# print(type(train))
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

train.loc[:, ("click2")] = train["click"] == 0
train.loc[:, ("click2")] = train["click2"].astype(int)


inputX = train.loc[:, cols_to_use].as_matrix()

inputY = train.loc[:, ['click', 'click2']].as_matrix()

inputX_test = test.loc[:, cols_to_use].as_matrix()
# print(inputX)
# print(inputY)

learning_rate = 0.00000001
training_epochs = 10
display_step = 2
n_samples = inputY.size

x = tf.placeholder(tf.float32, [None,10])

W = tf.Variable(tf.zeros([10,2]))

b = tf.Variable(tf.zeros([2]))

y_values = tf.add(tf.matmul(x, W), b)

# y = tf.nn.softmax(y_values)
y = tf.reciprocal(1 + tf.exp(-y_values))

y_ = tf.placeholder(tf.float32, [None,2])

cost = tf.reduce_sum(tf.pow(y_ - y,2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(training_epochs):  
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) # Take a gradient descent step using our inputs and labels

    # That's all! The rest of the cell just outputs debug messages. 
    # Display logs per epoch step
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print ("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc)) #, \"W=", sess.run(W), "b=", sess.run(b)

print ("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print ("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

test_preds = sess.run(y, feed_dict={x: inputX_test})

submit = pd.DataFrame({'ID':test['ID'], 'click':test_preds[:,0]})
submit.to_csv('tf_3.csv', index=False)