import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(24,input_dim=4, activation='relu'))
model.add(Dense(24,activation='relu'))
model.add(Dense(4,activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))


bandits = [0.7, 0.78, 0.55, 0.4]

state = [0, 0, 0, 0]
K = [0, 0, 0, 0]
for i in range(10000) :
    action = 0
    if np.random.rand(1) < 0.1 : action = np.random.randint(4)
    else : action = np.argmax(model.predict(next_state))

    g = bandits[action]
    #oyna
    ran = np.random.random()
    reward = 1 if (ran < g) else 0


    Q[action] = Q[action] + (1./(K[action]+1))*(reward - Q[action])
    K[action] = K[action] + 1
    print(Q)
