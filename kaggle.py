# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
#
# # Read the data
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')
# test = test[np.isfinite(test['Fare'])]
# # pull data into target (y) and predictors (X)
# train_y = train.Survived
# cols = [ 'Fare']
#
#
# train = train[np.isfinite(train['Fare'])]
# #train = train[np.isfinite(train['Sex'])]
# #train = train[np.isfinite(train['Age'])]
#
# # Create training predictors data
# train_X = train[cols]
#
# model = RandomForestRegressor()
# model.fit(train_X, train_y)
#
# ans = model.predict(test[cols])
#
# f = open("demofile.csv", "w")
# f.write('PassengerId,Survived\n')
# for i in range(892,1310) :
#     f.write(str(i)+"," + "0" + '\n')
#
#
#-------------------------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('train.csv')
print(train)
