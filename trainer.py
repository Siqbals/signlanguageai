import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#load data 
landict = pickle.load(open('./landmark.pickle', 'rb'))

#convert to numpy array
data = np.asarray(landict['data'])
labels = np.asarray(landict['labels'])

'''
take data, and split it into two sets, x train and x test 
take labels and split it into two sets, y train and y test

training set - used to train the model
test set - used to test the model itself (here we have specified 0.2, 20% of data is kept as a test)

shuffle - shuffling the data
stratify - split data set, but keep same proportion in x and y sets 
'''
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#set the model type
model = RandomForestClassifier()

#get the model fit 
model.fit(x_train,y_train)

#training the model itself!
y_predict = model.predict(x_test)

f = open('model.p', 'wb')
pickle.dump({'model':model}, f)
f.close()