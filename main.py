"""
Created by ChenJiongde/c00500953
"""
from env_test import TmsEnv
from rl_1 import RL_DDPG
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE


MAX_EPISODES = 500
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 500
reward_list = []
ON_TRAIN = True

# set environment
env = TmsEnv()

s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method
rl = RL_DDPG(a_dim, s_dim, a_bound)

# start training
def train():
    for i in range(MAX_EPISODES):
        s = env.reset()
        for j in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            # print('action in main: ', a)

            s_, r, done = env.step(a)


            # pro = Popen('cmd', shell=False, stdout=PIPE, stderr=PIPE)
            # out, err = pro.communicate()
            # print(out)
            # print(err)


            rl.store_transition(s, a, r, s_)

            if rl.pointer > MEMORY_CAPACITY:
                print('learning start')
                rl.learn()


            s = s_
            print('step:', j)
            if done or j == MAX_EP_STEPS - 1:
                reward_list.append(r)
                print('Ep: %i | %s  | step: %i' % (i, '---' if not done else 'done', j))

                break
            env.dymola.dsfinal2dsin()

    rl.save()
    x = range(MAX_EPISODES)
    plt.plot(x, reward_list)
    plt.show()



def eval():
    rl.restore()
    while True:
        s = env.reset()
        for _ in range(MAX_EP_STEPS):
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()

# summary

"""
env should have at least:
env.reset()
env.render()
env.step()

while RL should have at least:
rl.choose_action()
rl.store_transition()
rl.learn()
rl.memory_full
"""