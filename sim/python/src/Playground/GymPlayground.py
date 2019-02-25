import gym
#env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
env = gym.make('SpaceInvaders-v0')


for _ in range(10): # episode loop
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action