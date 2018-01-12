from gym.envs.registration import register

register(
    id='FrozenLake8x8rob-v0',
    entry_point='envs.frozen_lake:FrozenLakeEnv',
    kwargs={'map_name' : '8x8nohole', 'is_slippery' : False},
#    kwargs={'map_name' : '8x8', 'is_slippery' : False},
    max_episode_steps=200,
    reward_threshold=0.99, # optimum = 1
)

register(
    id='FrozenLake16x16rob-v0',
    entry_point='envs.frozen_lake:FrozenLakeEnv',
#    kwargs={'map_name' : '8x8nohole', 'is_slippery' : False},
    kwargs={'map_name' : '16x16', 'is_slippery' : False},
    max_episode_steps=400,
    reward_threshold=0.99, # optimum = 1
)

register(
    id='MountainCarRob-v0',
    entry_point='envs.mountain_car:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='CartPoleRob-v0',
    entry_point='envs.cartpole:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='TestRob-v0',
    entry_point='envs.testrob:TestRobEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='TestRob2-v0',
    entry_point='envs.testrob2:TestRob2Env',
    max_episode_steps=80,
    reward_threshold=195.0,
)

register(
    id='TestRob3-v0',
    entry_point='envs.testrob3:TestRob3Env',
    max_episode_steps=80,
    reward_threshold=195.0,
)

