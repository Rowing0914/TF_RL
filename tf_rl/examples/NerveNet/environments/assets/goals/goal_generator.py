import numpy as np

def generate_goals(num_goals):
    radius = 2.0
    angle = np.random.uniform(0, np.pi, size=(num_goals,))
    xpos = radius*np.cos(angle)
    ypos = radius*np.sin(angle)
    return np.concatenate([xpos[:, None], ypos[:, None]], axis=1)

if __name__ == "__main__":
    import pickle

    goals = generate_goals(num_goals=100)
    pickle.dump(goals, open("ant_train.pkl", "wb"))
    goals = generate_goals(num_goals=100)
    pickle.dump(goals, open("ant_eval.pkl", "wb"))
