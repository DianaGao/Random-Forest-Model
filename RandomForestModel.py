# -*- coding: utf-8 -*-

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the dataset
df = pd.read_csv("Social_Network_Ads.csv")
x = df.iloc[:, [2,3]].values
Y = df.iloc[:,4].values

# split the datasets into training & testing datasets
from sklearn.cross_validation import train_test_split
#tts = train_test_split()
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size = 0.2, random_state = 0 )

# feature the scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Train the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 20, criterion = "entropy", random_state = 0)
model.fit(X_train, y_train)


#predict the test dataset
y_pred = model.predict(X_test)

#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#visulizing the training result
from matplotlib.colors import ListedColormap
x_set, y_set = X_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:,0].max()+1, step = 0.01),
                     np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.01))
plt.contourf(x1,x2, model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(("red", "green")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                 c = ListedColormap(("red", "green"))(i), label = j)

plt.title("Training set result")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

#visulizing the training result
x_set, y_set = X_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min()-1, stop = x_set[:,0].max()+1, step = 0.01),
                     np.arange(start = x_set[:, 1].min()-1, stop = x_set[:, 1].max()+1, step = 0.01))
plt.contourf(x1,x2, model.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(("yellow", "blue")))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                 c = ListedColormap(("yellow", "blue"))(i), label = j)
plt.title("Testing set result")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()


