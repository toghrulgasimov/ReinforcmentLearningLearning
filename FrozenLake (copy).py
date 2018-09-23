import gym
import numpy as np
import random


env = gym.make('FrozenLake-v0')

env.reset()

#Initialize table with all zeros
Q = np.ones([env.observation_space.n,env.action_space.n]) * -1
Q[15,3] = 1
print(Q)
# Set learning parameters
lr = .8
y = .95
num_episodes = 70000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        #env.render()
        if i >= 10000 and i <= 10020 : env.render()
        if i >= 20000 and i <= 20020 : env.render()
        if i >= 30000 and i <= 30020 : env.render()
        if i >= 40000 and i <= 40020 : env.render()
        if i >= 50000 and i <= 50020 : env.render()
        if i >= 60000 and i <= 60020 : env.render()
        if i >= 70000 and i <= 70020 : env.render()
        if i >= 80000 and i <= 80020 : env.render()
        if i >= 90000 and i <= 90020 : env.render()
        #print(np.random.randn(1,env.action_space.n))
        #Choose an action by greedily (with noise) picking from Q table
        a = 0
        if random.uniform(0,1) <= 0.1 : a = random.randint(0, 3)
        else : a = np.argmax(Q[s,:])
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        #print(r)
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)
    #print('----------------------')


print "Score over time: " +  str(sum(rList)/num_episodes)
print "Final Q-Table Values"
print Q
