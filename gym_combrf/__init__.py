from gym.envs.registration import register

register(
    id='combrf-v0',
    entry_point='gym_combrf.envs:CombRF_Env',
)

register(
    id='combrf-v2',
    entry_point='gym_combrf.envs:CombRF_Env_v2',
)
