"""In NearestNeighbors, while working on algorithm using CIFAR10 dataset I couldn't understand how the math worked
and later I figured it out. 

Let me explain"""

# cifar10 has 50,000 training images and each maps out 32*32 pixels with each of 3 channels containing uint8 values.
# test set has similar but 10,000 values

import numpy as np
import pandas as pd

# assume a training set
train = pd.DataFrame({'a':[1,10,300], 'b':[1,50,600], 'c':[0,80,900], 'target':['cat', 'dog', 'car']})
print(train)

# assume a test set with 2 images to classify
test = pd.DataFrame({'a':[15,2], 'b':[45,1], 'c':[50,4]}) # by looking at this, 1st image is dog and next cat

pred = []
# so everything is fine till now

train_data = train[['a','b','c']]
print(train_data)

for i in range(test.shape[0]):
    x = np.reshape(test.iloc[i], 3)
    
    ab_diff = np.abs(train_data - x)

    net = np.sum(ab_diff, axis=1)

    min_index = np.argmin(net)

    pred.append(train['target'][min_index])

print(pred)
# see output is dog and cat