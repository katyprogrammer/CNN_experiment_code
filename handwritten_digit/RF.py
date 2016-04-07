from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

dataset = pd.read_csv('input/train.csv')
label = dataset[[0]].values.ravel()
train = dataset.iloc[:,1:].values
test = pd.read_csv('input/test.csv').values

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, label)
pred = rf.predict(test)
np.savetxt('submission_rf.csv', np.c_[range(1,len(test)+1), pred], delimiter=',', header='ImageId,Label', comments='', fmt='%d')
