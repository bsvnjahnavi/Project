#import the required libraries
import pandas as pd
import numpy as np


#1 reading the datsets from the agaricus-lepiota.data
X = pd.read_csv('agaricus-lepiota.data', names = ['classes', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'], na_values = "?")

#2 To remove all the null values from the dataset
X.dropna(axis = 0, how = 'any', inplace = True)
#print("After dropping all rows with any NaNs, shape of X is:", X.shape)

y = X.classes
X.drop('classes', axis = 1, inplace = True)
y = y.map({'e': 0, 'p': 1})

#3 using the one hot binary code to convert it into a binary
X = pd.get_dummies(X, columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
print(X)

#4 spliting the Data into train, validate, test
train, validate, test = \
              np.split(X.sample(frac=1, random_state=42), 
                       [int(.6*len(X)), int(.75*len(X))])
					   
#5 saving into text files for training, values and testing 
train.to_csv('training.txt', sep=',', header = False, index=False)
validate.to_csv('val.txt', sep=',', header = False, index=False)
test.to_csv('testing.txt', sep=',', header = False, index=False)
