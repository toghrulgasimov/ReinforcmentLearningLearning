import numpy as np
import gym
import random

from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


memory = deque(maxlen=2000)

env = gym.make('CartPole-v1')

model = Sequential()
model.add(Dense(24, input_dim=4, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=0.001))


for i in range(100000) :
    state = env.reset()
    done = False
    score = 0
    while not done :
        #calc action
        if i > 10000 : env.render()
        if i < 5300 and 5000<i : env.render()
        #env.render()
        action = 1
        if np.random.rand() <= 0.01 : action = random.randrange(2)
        else : action = np.argmax(model.predict(np.array(state).reshape(1,4))[0])
        next_state, reward, done, _ = env.step(action)
        print(next_state)
        memory.append((state, action, next_state, reward, done))
        state = next_state
        score = score + 1
    print(i, ' Score', score)

    #Relay
    if len(memory) < 32:
        continue
    #beyinden 32 dene misal goturduk
    sample = random.sample(memory, 32)
    for state, action, next_state, reward, done in sample :
        next_state = np.array(next_state).reshape(1,4)
        state = np.array(state).reshape(1,4)
        #geleceye pragnoz ver
        if not done : target = reward + 0.95 * np.amax(model.predict(next_state)[0])
        else : target = reward

        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1,verbose=0)
