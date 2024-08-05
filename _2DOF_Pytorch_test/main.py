"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
# from final.env import ArmEnv
# from final.rl import DDPG

from _2DOF_Pytorch_test.env import ArmEnv
# from rl import DDPG
from _2DOF_Pytorch_test.rl_torch import DDPG

MAX_EPISODES = 900
MAX_EP_STEPS = 300
ON_TRAIN = 0 #True

# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    reward_all = []
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done, _ = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                reward_all.append(ep_r)
                break
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.arange(len(reward_all)), reward_all)
    plt.ylabel('reward_all')
    plt.xlabel('training steps')
    plt.show()
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    env.set_goal(300,300)
    timer = 0
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done, angle_all = env.step(a)
        print(f"angle_all = {angle_all}")

        timer +=1
        if timer % 800 == 200:
            env.set_goal(100, 300)
        if timer % 800 == 400:
            env.set_goal(100, 100)
        if timer % 800 == 600:
            env.set_goal(300, 100)
        if timer % 800 == 0:
            env.set_goal(300, 300)




if ON_TRAIN:
    train()
else:
    eval()
    # cde = 0



