import numpy as np
import gym
import random

from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from random import randint



memory = deque(maxlen=2000)

env = gym.make('MountainCar-v0')

model = Sequential()
model.add(Dense(24, input_dim=2, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(3, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))


model2 = Sequential()
model2.add(Dense(24, input_dim=2, activation='relu'))
model2.add(Dense(24, activation='relu'))
model2.add(Dense(3, activation='linear'))
model2.compile(loss='mse', optimizer=Adam(lr=0.001))

for i in range(100000) :
    state = env.reset()
    done = False
    score = 0
    while not done :
        #calc action
        if i < 1005 and 1000<i : env.render()
        if i < 2005 and 2000<i : env.render()
        if i < 3005 and 3000<i : env.render()
        if i < 4005 and 4000<i : env.render()
        if i < 5005 and 5000<i : env.render()
        if i < 6005 and 6000<i : env.render()
        if i < 7005 and 7000<i : env.render()
        if i < 8005 and 8000<i : env.render()
        if i < 9005 and 9000<i : env.render()
        if i < 10005 and 10000<i : env.render()
        if i < 11005 and 11000<i : env.render()
        if i < 12005 and 12000<i : env.render()
        if i < 13005 and 13000<i : env.render()
        if i < 14005 and 14000<i : env.render()
        if i < 15005 and 15000<i : env.render()
        #if i > 500 : env.render()
        action = 1
        AB = randint(0, 9)
        if np.random.rand() <= 0.01 : action = random.randrange(3)
        else : action = np.argmax(model.predict(np.array(state).reshape(1,2))[0])
        next_state, reward, done, _ = env.step(action)
        suret = next_state[1]
        pos = next_state[0]
        suret = (suret - (-0.07)) / (0.07- (-0.07)) - 0.5+0.1
        # pos = (pos - (-1.2)) / (0.6- (-1.2))

        reward = abs(suret)

        memory.append((state, action, next_state, reward, done))

        state = next_state
        score = score + reward
    print(i, ' Score', score)

    #Relay
    if len(memory) < 32:
        continue
    #beyinden 32 dene misal goturduk
    sample = random.sample(memory, 32)
    for state, action, next_state, reward, done in sample :
        next_state = np.array(next_state).reshape(1,2)
        state = np.array(state).reshape(1,2)
        ra = randint(0,1)
        if ra == 0 :
            #123
            if not done : target = reward + 0.95 * np.amax(model.predict(next_state)[0])
            else : target = -1
            #print(model.predict(next_state)[0]) ``
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1,verbose=0)
        else :
            if not done : target = reward + 0.95 * np.amax(model.predict(next_state)[0])
            else : target = -1
            #print(model.predict(next_state)[0]) ``
            target_f = model.predict(state)
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1,verbose=0)
            #asd
        #geleceye pragnoz ver
        #target = reward + 0.95 * np.amax(model.predict(next_state)[0])
        if not done : target = reward + 0.95 * np.amax(model.predict(next_state)[0])
        else : target = -1
        #print(model.predict(next_state)[0]) ``
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1,verbose=0)
