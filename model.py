




import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV,cross_val_score,cross_validate,train_test_split

from sklearn.metrics import mean_squared_error

import pickle





df = pd.read_csv('datasets\wine.csv')





df.head()





df.shape





df.dtypes





df.describe()





X = df.copy()
y = X.pop('quality')





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)





model = RandomForestRegressor(max_depth=1,random_state=42)




model.fit(X_train,y_train)





prediction = model.predict(X_test)





mean_squared_error = np.sqrt(mean_squared_error(y_test,prediction))





pickle.dump(model, open('model.pkl', 'wb'))


model_2 = pickle.load(open('model.pkl','rb'))


with open('metrics.txt','w') as outfile:
    outfile.write("Mean squared arror is {}".format(round(mean_squared_error,3)))





