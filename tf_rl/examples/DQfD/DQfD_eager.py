import gym
import argparse
import os
import time
import itertools
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tf_rl.common.wrappers import MyWrapper, wrap_deepmind, make_atari
from tf_rl.examples.DQN import Model as Model_CartPole_DQN
from tf_rl.common.memory import PrioritizedReplayBuffer
from tf_rl.common.utils import AnnealingSchedule, soft_target_model_update_eager, eager_setup
from tf_rl.common.policy import EpsilonGreedyPolicy_eager, BoltzmannQPolicy_eager, TestPolicy
from tf_rl.common.networks import Duelling_atari as Model
from tf_rl.agents.DQN import DQN
from tf_rl.agents.DQfD import DQfD

eager_setup()


def pretrain_without_prioritisation(agent, expert, policy, expert_policy, env, num_demo, num_train):
    """

    Populating the memory with demonstrations

    """
    batch_experience = list()
    batches = list()

    print("Pupulating a memory with demonstrations")
    for _ in range(num_demo):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action_e = expert_policy.select_action(expert, state)
            action_l = policy.select_action(agent, state)

            next_state, reward, done, _ = env.step(action_e)
            batch_experience.append([state, action_e, action_l, reward, next_state, done])
            state = next_state
            episode_reward += reward

            if len(batch_experience) == 10:
                batches.append(batch_experience)
                batch_experience = list()

        print("Game Over with score: {0}".format(episode_reward))

    print(len(batches))

    """

    Pre-train the agent with collected demonstrations

    """

    for i in range(num_train):
        sample = random.sample(batches, 1)

        states, actions_e, actions_l, rewards, next_states, dones = [], [], [], [], [], []

        for row in sample[0]:
            states.append(row[0])
            actions_e.append(row[1])
            actions_l.append(row[2])
            rewards.append(row[3])
            next_states.append(row[4])
            dones.append(row[5])
        states, actions_e, actions_l, rewards, next_states, dones = np.array(states), np.array(actions_e), np.array(
            actions_l), np.array(rewards), np.array(next_states), np.array(dones)
        agent.update(states, actions_e, actions_l, rewards, next_states, dones)

        if np.random.rand() > 0.3:
            if params.update_hard_or_soft == "hard":
                agent.target_model.set_weights(agent.main_model.get_weights())
            elif params.update_hard_or_soft == "soft":
                soft_target_model_update_eager(agent.target_model, agent.main_model, tau=params.soft_update_tau)

    """

    Test the pre-trained agent

    """

    for _ in range(10):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # action_e = expert_policy.select_action(expert, state)
            action_l = expert_policy.select_action(agent, state)

            # next_state, reward, done, _ = env.step(action_e)
            next_state, reward, done, _ = env.step(action_l)
            state = next_state
            episode_reward += reward

        print("Game Over with score: {0}".format(episode_reward))

    return agent, replay_buffer


def pretrain_with_prioritisation(agent, expert, policy, expert_policy, env, replay_buffer, num_demo, num_train):
    """

        Populating the memory with demonstrations

    """
    states, actions_e, actions_l, rewards, next_states, dones = [], [], [], [], [], []

    print("Pupulating a memory with demonstrations")
    for _ in range(num_demo):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action_e = expert_policy.select_action(expert, state)
            action_l = policy.select_action(agent, state)
            next_state, reward, done, _ = env.step(action_e)

            states.append(state)
            actions_e.append(action_e)
            actions_l.append(action_l)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done or len(states) == 10:
                # since we rely on the expert at this populating phase, this hardly happens though.
                # at the terminal state, if a memory is not full, then we would fill the gap till n-step with 0
                if len(states) < 10:
                    for _ in range(10 - len(states)):
                        states.append(np.zeros(state.shape))
                        actions_e.append(0)
                        actions_l.append(0)
                        rewards.append(0)
                        next_states.append(np.zeros(state.shape))
                        dones.append(0)
                else:
                    assert len(states) == 10
                    replay_buffer.add(states, (actions_e, actions_l), rewards, next_states, dones)
                states, actions_e, actions_l, rewards, next_states, dones = [], [], [], [], [], []

        print("EXPERT: Game Over with score: {0}".format(episode_reward))

    del states, actions_e, actions_l, rewards, next_states, dones, state, action_e, action_l, reward, next_state, done

    """

    Pre-train the agent with collected demonstrations

    """
    for i in range(num_train):
        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(1, Beta.get_value(i))

        # manually unpack the actions into expert's ones and learner's ones
        actions_e = actions[:, 0, :].reshape(1, 10)
        actions_l = actions[:, 1, :].reshape(1, 10)

        loss = agent.update(states[0, :, :], actions_e, actions_l, rewards, next_states[0, :, :], dones)

        # add noise to the priorities
        loss = np.abs(np.mean(loss).reshape(1, 1)) + params.prioritized_replay_noise

        # Update a prioritised replay buffer using a batch of losses associated with each timestep
        replay_buffer.update_priorities(indices, loss)

        if np.random.rand() > 0.3:
            if params.update_hard_or_soft == "hard":
                agent.target_model.set_weights(agent.main_model.get_weights())
            elif params.update_hard_or_soft == "soft":
                soft_target_model_update_eager(agent.target_model, agent.main_model, tau=params.soft_update_tau)

    """

    Test the pre-trained agent

    """

    for _ in range(10):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # action_e = expert_policy.select_action(expert, state)
            action_l = expert_policy.select_action(agent, state)

            # next_state, reward, done, _ = env.step(action_e)
            next_state, reward, done, _ = env.step(action_l)
            state = next_state
            episode_reward += reward

        print("LEARNER: Game Over with score: {0}".format(episode_reward))

    return agent, replay_buffer


if __name__ == '__main__':

    logdirs = logdirs()

    try:
        os.system("rm -rf {}".format(logdirs.log_DQfD))
    except:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="CartPole", help="game env type")
    parser.add_argument("--num_episodes", default=100, type=int, help="game env type")
    args = parser.parse_args()

    if args.mode == "CartPole":
        env = MyWrapper(gym.make("CartPole-v0"))
    elif args.mode == "Atari":
        env = wrap_deepmind(make_atari("PongNoFrameskip-v4"))

    params = Parameters(algo="DQfD", mode=args.mode)
    params.num_episodes = args.num_episodes
    replay_buffer = PrioritizedReplayBuffer(params.memory_size, alpha=params.prioritized_replay_alpha)
    Beta = AnnealingSchedule(start=params.prioritized_replay_beta_start, end=params.prioritized_replay_beta_end,
                             decay_steps=params.decay_steps)
    agent = DQfD(args.mode, Model, Model, env.action_space.n, params, logdirs.model_DQN)
    if params.policy_fn == "Eps":
        Epsilon = AnnealingSchedule(start=params.epsilon_start, end=params.epsilon_end,
                                    decay_steps=params.decay_steps)
        policy = EpsilonGreedyPolicy_eager(Epsilon_fn=Epsilon)
    elif params.policy_fn == "Boltzmann":
        policy = BoltzmannQPolicy_eager()

    reward_buffer = deque(maxlen=params.reward_buffer_ep)
    summary_writer = tf.contrib.summary.create_file_writer(logdirs.log_DQfD)

    expert = DQN(args.mode, Model_CartPole_DQN, Model_CartPole_DQN, env.action_space.n, params, logdirs.model_DQN)
    expert_policy = TestPolicy()
    expert.check_point.restore(expert.manager.latest_checkpoint)
    print("Restore the model from disk")

    # agent, _ = pretrain_without_prioritisation(agent, expert, policy, expert_policy, env, 100, 100)
    agent, replay_buffer = pretrain_with_prioritisation(agent, expert, policy, expert_policy, env, replay_buffer, 100,
                                                        1000)

    # to indicate the pre-training has been done
    agent.pretrain_flag = 0

    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            global_timestep = 0
            states_buf, actions_e_buf, actions_l_buf, rewards_buf, next_states_buf, dones_buf = [], [], [], [], [], []

            for i in range(4000):
                state = env.reset()
                total_reward = 0
                start = time.time()
                cnt_action = list()
                policy.index_episode = i
                agent.index_episode = i
                for t in itertools.count():
                    # env.render()
                    action = policy.select_action(agent, state)
                    next_state, reward, done, info = env.step(action)

                    states_buf.append(state)
                    actions_e_buf.append(action)
                    actions_l_buf.append(0)
                    rewards_buf.append(reward)
                    next_states_buf.append(next_state)
                    dones_buf.append(done)
                    state = next_state
                    total_reward += reward
                    state = next_state
                    cnt_action.append(action)

                    if done or len(states_buf) == 10:
                        tf.contrib.summary.scalar("reward", total_reward, step=global_timestep)

                        # since we rely on the expert at this populating phase, this hardly happens though.
                        # at the terminal state, if a memory is not full, then we would fill the gap till n-step with 0
                        if len(states_buf) < 10:
                            for _ in range(10 - len(states_buf)):
                                states_buf.append(np.zeros(state.shape))
                                actions_e_buf.append(0)
                                actions_l_buf.append(0)
                                rewards_buf.append(0)
                                next_states_buf.append(np.zeros(state.shape))
                                dones_buf.append(0)
                        else:
                            assert len(states_buf) == 10
                            replay_buffer.add(states_buf, (actions_e_buf, actions_l_buf), rewards_buf, next_states_buf,
                                              dones_buf)
                        states_buf, actions_e_buf, actions_l_buf, rewards_buf, next_states_buf, dones_buf = [], [], [], [], [], []

                        if global_timestep > params.learning_start:
                            # PER returns: state, action, reward, next_state, done, weights(a weight for an episode), indices(indices for a batch of episode)
                            states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(1,
                                                                                                                  Beta.get_value(
                                                                                                                      i))

                            # manually unpack the actions into expert's ones and learner's ones
                            actions_e = actions[:, 0, :].reshape(1, 10)
                            actions_l = actions[:, 1, :].reshape(1, 10)

                            loss = agent.update(states[0, :, :], actions_e, actions_l, rewards, next_states[0, :, :],
                                                dones)
                            logging(global_timestep, params.num_frames, i, time.time() - start, total_reward,
                                    np.mean(loss),
                                    policy.current_epsilon(), cnt_action)

                            # add noise to the priorities
                            loss = np.abs(np.mean(loss).reshape(1, 1)) + params.prioritized_replay_noise

                            # Update a prioritised replay buffer using a batch of losses associated with each timestep
                            replay_buffer.update_priorities(indices, loss)

                            if np.random.rand() > 0.5:
                                if params.update_hard_or_soft == "hard":
                                    agent.target_model.set_weights(agent.main_model.get_weights())
                                elif params.update_hard_or_soft == "soft":
                                    soft_target_model_update_eager(agent.target_model, agent.main_model,
                                                                   tau=params.soft_update_tau)
                    if done:
                        break

                    global_timestep += 1

                # store the episode reward
                reward_buffer.append(total_reward)
                # check the stopping condition
                if np.mean(reward_buffer) > params.goal:
                    print("GAME OVER!!")
                    break

    env.close()
