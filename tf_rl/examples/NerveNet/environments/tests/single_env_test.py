import gym
import environments.register as register

# env_name = "CentipedeThree-v1"
env_name = "CentipedeFour-v1"
# env_name = "AntS-v1"
# env_name = "AntWithGoal-v1"
# env_name = "Ant-v2"
# env_name = "Humanoid-v2"
# env_name = "CentipedeFive-v1"

env = gym.make(env_name)
print("=====", env.action_space.shape)

for ep in range(10):
    env.reset()
    for _ in range(200):
        env.render()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)

"""
ALL_ENV_LIST = [
    WalkersHopperone-v1
    WalkersHalfhumanoidone-v1
    WalkersHalfcheetahone-v1
    WalkersFullcheetahone-v1
    WalkersOstrichone-v1
    WalkersHoppertwo-v1
    WalkersHalfhumanoidtwo-v1
    WalkersHalfcheetahtwo-v1
    WalkersFullcheetahtwo-v1
    WalkersOstrichtwo-v1
    WalkersHopperthree-v1
    WalkersHalfhumanoidthree-v1
    WalkersHalfcheetahthree-v1
    WalkersFullcheetahthree-v1
    WalkersOstrichthree-v1
    WalkersHopperfour-v1
    WalkersHalfhumanoidfour-v1
    WalkersHalfcheetahfour-v1
    WalkersFullcheetahfour-v1
    WalkersOstrichfour-v1
    WalkersHopperfive-v1
    WalkersHalfhumanoidfive-v1
    WalkersHalfcheetahfive-v1
    WalkersFullcheetahfive-v1
    WalkersOstrichfive-v1
    CentipedeThree-v1
    CentipedeFour-v1
    CentipedeFive-v1
    CentipedeSix-v1
    CentipedeSeven-v1
    CentipedeEight-v1
    CentipedeTen-v1
    CentipedeTwelve-v1
    CentipedeFourteen-v1
    CentipedeTwenty-v1
    CentipedeThirty-v1
    CentipedeForty-v1
    CentipedeFifty-v1
    ReacherZero-v1
    ReacherOne-v1
    ReacherTwo-v1
    ReacherThree-v1
    ReacherFour-v1
    ReacherFive-v1
    ReacherSix-v1
    ReacherSeven-v1
    SnakeThree-v1
    SnakeFour-v1
    SnakeFive-v1
    SnakeSix-v1
    SnakeSeven-v1
    SnakeEight-v1
    SnakeNine-v1
    SnakeTen-v1
    SnakeTwenty-v1
    SnakeForty-v1
]
"""