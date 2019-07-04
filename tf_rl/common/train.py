import time
from collections import deque
from tf_rl.common.utils import *

"""
===== Value Based Algorithm =====

TODO: think about incorporating PER's memory updating procedure into the model
so that, we can unify train_DQN and train_DQN_PER 
"""


def train_DQN(agent, env, policy, replay_buffer, reward_buffer, summary_writer):
    """
    Training script for DQN and other advanced models without PER

    :param agent:
    :param env:
    :param policy:
    :param replay_buffer:
    :param reward_buffer:
    :param params:
    :param summary_writer:
    :return:
    """
    get_ready(agent.params)
    time_buffer = list()
    global_timestep = tf.compat.v1.train.get_global_step()
    log = logger(agent.params)
    # normaliser = RunningMeanStd(env.reset().shape[0])
    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                start = time.time()
                cnt_action = list()
                done = False
                while not done:
                    # normaliser.update(state)
                    # normaliser.normalise(state)
                    action = policy.select_action(agent, state)
                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)

                    global_timestep.assign_add(1)
                    total_reward += reward
                    state = next_state
                    cnt_action.append(action)

                    # for evaluation purpose
                    if global_timestep.numpy() % agent.params.eval_interval == 0:
                        agent.eval_flg = True

                    if (global_timestep.numpy() > agent.params.learning_start) and (
                            global_timestep.numpy() % agent.params.train_interval == 0):
                        states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)

                        loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

                    # synchronise the target and main models by hard
                    if (global_timestep.numpy() > agent.params.learning_start) and (
                            global_timestep.numpy() % agent.params.sync_freq == 0):
                        agent.manager.save()
                        agent.target_model.set_weights(agent.main_model.get_weights())

                """
                ===== After 1 Episode is Done =====
                """

                tf.contrib.summary.scalar("reward", total_reward, step=i)
                tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
                if i >= agent.params.reward_buffer_ep:
                    tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
                tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

                # store the episode reward
                reward_buffer.append(total_reward)
                time_buffer.append(time.time() - start)

                if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                    log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss),
                                policy.current_epsilon(), cnt_action)
                    time_buffer = list()

                if agent.eval_flg:
                    test_Agent(agent, env)
                    agent.eval_flg = False

                # check the stopping condition
                if global_timestep.numpy() > agent.params.num_frames:
                    print("=== Training is Done ===")
                    test_Agent(agent, env, n_trial=agent.params.test_episodes)
                    env.close()
                    break


def train_DQN_PER(agent, env, policy, replay_buffer, reward_buffer, Beta, summary_writer):
    """
    Training script for DQN with PER

    :param agent:
    :param env:
    :param policy:
    :param replay_buffer:
    :param reward_buffer:
    :param params:
    :param summary_writer:
    :return:
    """
    get_ready(agent.params)
    global_timestep = tf.compat.v1.train.get_global_step()
    time_buffer = list()
    log = logger(agent.params)
    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                start = time.time()
                cnt_action = list()
                done = False
                while not done:
                    action = policy.select_action(agent, state)
                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)

                    global_timestep.assign_add(1)
                    total_reward += reward
                    state = next_state
                    cnt_action.append(action)

                    # for evaluation purpose
                    if global_timestep.numpy() % agent.params.eval_interval == 0:
                        agent.eval_flg = True

                    if (global_timestep.numpy() > agent.params.learning_start) and (
                            global_timestep.numpy() % agent.params.train_interval == 0):
                        # PER returns: state, action, reward, next_state, done, weights(a weight for an episode), indices(indices for a batch of episode)
                        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(
                            agent.params.batch_size, Beta().numpy())

                        loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

                        # add noise to the priorities
                        batch_loss = np.abs(batch_loss) + agent.params.prioritized_replay_noise

                        # Update a prioritised replay buffer using a batch of losses associated with each timestep
                        replay_buffer.update_priorities(indices, batch_loss)

                    # synchronise the target and main models by hard or soft update
                    if (global_timestep.numpy() > agent.params.learning_start) and (
                            global_timestep.numpy() % agent.params.sync_freq == 0):
                        agent.manager.save()
                        agent.target_model.set_weights(agent.main_model.get_weights())

                """
                ===== After 1 Episode is Done =====
                """

                tf.contrib.summary.scalar("reward", total_reward, step=i)
                tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
                if i >= agent.params.reward_buffer_ep:
                    tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
                tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

                # store the episode reward
                reward_buffer.append(total_reward)
                time_buffer.append(time.time() - start)

                if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                    log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss),
                                policy.current_epsilon(), cnt_action)
                    time_buffer = list()

                if agent.eval_flg:
                    test_Agent(agent, env)
                    agent.eval_flg = False

                # check the stopping condition
                if global_timestep.numpy() > agent.params.num_frames:
                    print("=== Training is Done ===")
                    test_Agent(agent, env, n_trial=agent.params.test_episodes)
                    env.close()
                    break


def pretrain_DQfD(expert, agent, env, policy, replay_buffer, reward_buffer, summary_writer, Beta):
    """
    Pre-training API for DQfD: https://arxiv.org/pdf/1704.03732.pdf

    :param agent:
    :param env:
    :param policy:
    :param replay_buffer:
    :param reward_buffer:
    :param params:
    :param summary_writer:
    :return:
    """
    get_ready(agent.params)
    global_timestep = tf.compat.v1.train.get_global_step()
    time_buffer = list()
    log = logger(agent.params)
    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                start = time.time()
                cnt_action = list()
                done = False
                while not done:
                    action_e = np.argmax(expert.predict(state))
                    action_l = policy.select_action(agent, state)
                    next_state, reward, done, info = env.step(action_e)
                    replay_buffer.add(state, [action_l, action_e], reward, next_state, done)

                    global_timestep.assign_add(1)
                    total_reward += reward
                    state = next_state
                    cnt_action.append(action_e)

                    # for evaluation purpose
                    if global_timestep.numpy() % agent.params.eval_interval == 0:
                        agent.eval_flg = True

                    if (global_timestep.numpy() > agent.params.learning_start) and (
                            global_timestep.numpy() % agent.params.train_interval == 0):
                        states, actions, rewards, next_states, dones, weights, indices = replay_buffer.sample(
                            agent.params.batch_size, Beta.get_value())

                        loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

                        # add noise to the priorities
                        batch_loss = np.abs(batch_loss) + agent.params.prioritized_replay_noise

                        # Update a prioritised replay buffer using a batch of losses associated with each timestep
                        replay_buffer.update_priorities(indices, batch_loss)

                    # synchronise the target and main models by hard or soft update
                    if (global_timestep.numpy() > agent.params.learning_start) and (
                            global_timestep.numpy() % agent.params.sync_freq == 0):
                        agent.manager.save()
                        agent.target_model.set_weights(agent.main_model.get_weights())

                """
                ===== After 1 Episode is Done =====
                """

                tf.contrib.summary.scalar("reward", total_reward, step=i)
                tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
                if i >= agent.params.reward_buffer_ep:
                    tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)
                tf.contrib.summary.histogram("taken actions", cnt_action, step=i)

                # store the episode reward
                reward_buffer.append(total_reward)
                time_buffer.append(time.time() - start)

                if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                    log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss),
                                policy.current_epsilon(), cnt_action)
                    time_buffer = list()

                if agent.eval_flg:
                    test_Agent(agent, env)
                    agent.eval_flg = False

                # check the stopping condition
                if global_timestep.numpy() > agent.params.num_frames:
                    print("=== Training is Done ===")
                    test_Agent(agent, env, n_trial=agent.params.test_episodes)
                    env.close()
                    break


def train_DQN_afp(agent, expert, env, agent_policy, expert_policy, replay_buffer, reward_buffer, params,
                  summary_writer):
    """
    Training script for DQN and other advanced models without PER

    :param agent:
    :param env:
    :param policy:
    :param replay_buffer:
    :param reward_buffer:
    :param params:
    :param summary_writer:
    :return:
    """
    get_ready(params)
    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            global_timestep = 0
            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                start = time.time()
                cnt_action = list()
                agent_policy.index_episode = i
                agent.index_episode = i
                for t in itertools.count():
                    # env.render()
                    action = agent_policy.select_action(agent, state)

                    # where the AFP comes in
                    # if learning agent is not sure about his decision, then he asks for expert's help
                    if action <= 0.5:
                        action = expert_policy.select_action(expert, state)

                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)

                    total_reward += reward
                    state = next_state
                    cnt_action.append(action)
                    global_timestep += 1

                    if (global_timestep > params.learning_start) and (global_timestep % params.train_interval == 0):
                        states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)

                        loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

                    # synchronise the target and main models by hard or soft update
                    if (global_timestep > params.learning_start) and (global_timestep % params.sync_freq == 0):
                        agent.manager.save()
                        if params.update_hard_or_soft == "hard":
                            agent.target_model.set_weights(agent.main_model.get_weights())
                        elif params.update_hard_or_soft == "soft":
                            soft_target_model_update_eager(agent.target_model, agent.main_model,
                                                           tau=params.soft_update_tau)

                    if done:
                        tf.contrib.summary.scalar("reward", total_reward, step=i)
                        # store the episode reward
                        reward_buffer.append(total_reward)

                        if global_timestep > params.learning_start:
                            try:
                                logging(global_timestep, params.num_frames, i, time.time() - start, total_reward,
                                        np.mean(loss), 0, cnt_action)
                            except:
                                pass

                        break

                # check the stopping condition
                # if np.mean(reward_buffer) > params.goal or global_timestep.numpy() > params.num_frames:
                if global_timestep.numpy() > params.num_frames:
                    print("GAME OVER!!")
                    env.close()
                    break


def train_DRQN(agent, env, policy, replay_buffer, reward_buffer, params, summary_writer):
    """
    Training script for DQN and other advanced models without PER

    :param agent:
    :param env:
    :param policy:
    :param replay_buffer:
    :param reward_buffer:
    :param params:
    :param summary_writer:
    :return:
    """
    get_ready(params)
    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            global_timestep = 0
            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                start = time.time()
                cnt_action = list()
                policy.index_episode = i
                agent.index_episode = i
                episode_memory = list()
                for t in itertools.count():
                    # env.render()
                    action = policy.select_action(agent, state.reshape(1, 4))
                    next_state, reward, done, info = env.step(action)
                    episode_memory.append((state, action, reward, next_state, done))

                    total_reward += reward
                    state = next_state
                    cnt_action.append(action)
                    global_timestep += 1

                    if global_timestep > params.learning_start:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(params.batch_size)
                        _states, _actions, _rewards, _next_states, _dones = [], [], [], [], []
                        for index, data in enumerate(zip(states, actions, rewards, next_states, dones)):
                            s1, a, r, s2, d = data
                            ep_start = np.random.randint(0, len(s1) + 1 - 4)
                            # states[i] = s1[ep_start:ep_start+4, :]
                            # actions[i] = a[ep_start:ep_start+4]
                            # rewards[i] = r[ep_start:ep_start+4]
                            # next_states[i] = s2[ep_start:ep_start+4, :]
                            # dones[i] = d[ep_start:ep_start+4]
                            _states.append(s1[ep_start:ep_start + 4, :])
                            _actions.append(a[ep_start:ep_start + 4])
                            _rewards.append(r[ep_start:ep_start + 4])
                            _next_states.append(s2[ep_start:ep_start + 4, :])
                            _dones.append(d[ep_start:ep_start + 4])

                        _states, _actions, _rewards, _next_states, _dones = np.array(_states), np.array(
                            _actions), np.array(_rewards), np.array(_next_states), np.array(_dones)

                        # loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)
                        loss, batch_loss = agent.update(_states, _actions, _rewards, _next_states, _dones)
                        logging(global_timestep, params.num_frames, i, time.time() - start, total_reward, np.mean(loss),
                                policy.current_epsilon(), cnt_action)

                        if np.random.rand() > 0.5:
                            agent.manager.save()
                            if params.update_hard_or_soft == "hard":
                                agent.target_model.set_weights(agent.main_model.get_weights())
                            elif params.update_hard_or_soft == "soft":
                                soft_target_model_update_eager(agent.target_model, agent.main_model,
                                                               tau=params.soft_update_tau)

                    if done:
                        tf.contrib.summary.scalar("reward", total_reward, step=global_timestep)
                        reward_buffer.append(total_reward)

                        s1, a, r, s2, d = [], [], [], [], []
                        for data in episode_memory:
                            s1.append(data[0])
                            a.append(data[1])
                            r.append(data[2])
                            s2.append(data[3])
                            d.append(data[4])

                        replay_buffer.add(s1, a, r, s2, d)
                        break

                # check the stopping condition
                if np.mean(reward_buffer) > params.goal:
                    print("GAME OVER!!")
                    env.close()
                    break


"""

===== Policy Based Algorithm =====

"""


# I have referred to https://github.com/laket/DDPG_Eager in terms of design of algorithm
# I know this is not precisely accurate to the original algo, but this works better than that... lol
def train_DDPG(agent, env, replay_buffer, reward_buffer, summary_writer):
    get_ready(agent.params)

    global_timestep = tf.compat.v1.train.get_global_step()
    time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
    log = logger(agent.params)

    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                start = time.time()
                agent.random_process.reset_states()
                done = False
                episode_len = 0
                while not done:
                    # env.render()
                    if global_timestep.numpy() < agent.params.learning_start:
                        action = env.action_space.sample()
                    else:
                        action = agent.predict(state)
                    # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
                    next_state, reward, done, info = env.step(action * env.action_space.high)
                    replay_buffer.add(state, action, reward, next_state, done)

                    global_timestep.assign_add(1)
                    episode_len += 1
                    total_reward += reward
                    state = next_state

                    # for evaluation purpose
                    if global_timestep.numpy() % agent.params.eval_interval == 0:
                        agent.eval_flg = True

                """
                ===== After 1 Episode is Done =====
                """

                # train the model at this point
                for t_train in range(episode_len):  # in mujoco, this will be 1,000 iterations!
                    states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
                    loss = agent.update(states, actions, rewards, next_states, dones)
                    soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.soft_update_tau)
                    soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.soft_update_tau)

                tf.contrib.summary.scalar("reward", total_reward, step=i)
                tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
                if i >= agent.params.reward_buffer_ep:
                    tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)

                # store the episode reward
                reward_buffer.append(total_reward)
                time_buffer.append(time.time() - start)

                if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                    log.logging(global_timestep.numpy(), i, np.sum(time_buffer), reward_buffer, np.mean(loss), 0, [0])

                if agent.eval_flg:
                    test_Agent_DDPG(agent, env)
                    agent.eval_flg = False

                # check the stopping condition
                if global_timestep.numpy() > agent.params.num_frames:
                    print("=== Training is Done ===")
                    test_Agent_DDPG(agent, env, n_trial=agent.params.test_episodes)
                    env.close()
                    break


def train_SAC(agent, env, replay_buffer, reward_buffer, summary_writer):
    get_ready(agent.params)

    global_timestep = tf.compat.v1.train.get_global_step()
    log = logger(agent.params)

    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for i in itertools.count():
                state = env.reset()
                total_reward = 0
                start = time.time()
                done = False
                episode_len = 0
                while not done:
                    # env.render()
                    if global_timestep.numpy() < agent.params.learning_start:
                        action = env.action_space.sample()
                    else:
                        action = agent.predict(state)

                    next_state, reward, done, info = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)

                    global_timestep.assign_add(1)
                    episode_len += 1
                    total_reward += reward
                    state = next_state

                    # evaluate the model once in 10 episodes
                    if i + 1 % 10 == 0:
                        agent.eval_flg = True

                    # train the model at this point
                    if global_timestep.numpy() > agent.params.learning_start:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(agent.params.batch_size)
                        loss = agent.update(states, actions, rewards, next_states, dones)
                        soft_target_model_update_eager(agent.target_critic, agent.critic,
                                                       tau=agent.params.soft_update_tau)

                """
                ===== After 1 Episode is Done =====
                """

                tf.contrib.summary.scalar("reward", total_reward, step=i)
                tf.contrib.summary.scalar("exec time", time.time() - start, step=i)
                if i >= agent.params.reward_buffer_ep:
                    tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=i)

                # store the episode reward
                reward_buffer.append(total_reward)

                # we log the training progress once in a `reward_buffer_ep` time
                if global_timestep.numpy() > agent.params.learning_start and i % agent.params.reward_buffer_ep == 0:
                    log.logging(global_timestep.numpy(), i, time.time() - start, reward_buffer, np.mean(loss), 0, [0])

                if agent.eval_flg:
                    test_Agent_TRPO(agent, env)
                    agent.eval_flg = False

                # check the stopping condition
                if global_timestep.numpy() > agent.params.num_frames:
                    print("=== Training is Done ===")
                    test_Agent_TRPO(agent, env, n_trial=agent.params.test_episodes)
                    env.close()
                    break


# design pattern follows this repo: https://github.com/TianhongDai/hindsight-experience-replay
def train_HER(agent, env, replay_buffer, summary_writer):
    get_ready(agent.params)
    global_timestep = tf.compat.v1.train.get_global_step()
    total_ep = 0

    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for epoch in range(agent.params.num_epochs):
                successes = list()
                for cycle in range(agent.params.num_cycles):
                    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                    for ep in range(agent.params.num_episodes):
                        state = env.reset()
                        # obs, achieved_goal, desired_goal in `numpy.ndarray`
                        obs, ag, dg, rg = state_unpacker(state)
                        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                        success = list()
                        for ts in range(agent.params.num_steps):
                            # env.render()
                            action = agent.predict(obs, dg)
                            action = action_postprocessing(action, agent.params)

                            next_state, _, _, info = env.step(action)

                            # obs, achieved_goal, desired_goal in `numpy.ndarray`
                            next_obs, next_ag, next_dg, next_rg = state_unpacker(next_state)

                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())
                            ep_g.append(dg.copy())
                            ep_actions.append(action.copy())

                            global_timestep.assign_add(1)
                            success.append(info.get('is_success'))
                            obs = next_obs
                            # rg = next_rg
                            ag = next_ag

                        """
                        === After 1 ep ===
                        """
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        mb_obs.append(ep_obs)
                        mb_ag.append(ep_ag)
                        mb_g.append(ep_g)
                        mb_actions.append(ep_actions)
                        successes.append(success)

                        total_ep += ep
                        tf.contrib.summary.scalar("Train Success Rate", np.mean(success), step=total_ep)

                    """
                    === After num_episodes ===
                    """
                    # convert them into arrays
                    mb_obs = np.array(mb_obs)
                    mb_ag = np.array(mb_ag)
                    mb_g = np.array(mb_g)
                    mb_actions = np.array(mb_actions)
                    replay_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])

                    # ==== update normaliser ====
                    mb_obs_next = mb_obs[:, 1:, :]
                    mb_ag_next = mb_ag[:, 1:, :]
                    # get the number of normalization transitions
                    num_transitions = mb_actions.shape[1]
                    # create the new buffer to store them
                    buffer_temp = {'obs': mb_obs,
                                   'ag': mb_ag,
                                   'g': mb_g,
                                   'actions': mb_actions,
                                   'obs_next': mb_obs_next,
                                   'ag_next': mb_ag_next,
                                   }
                    transitions = replay_buffer.sample_func(buffer_temp, num_transitions)
                    # update
                    agent.o_norm.update(transitions['obs'])
                    agent.g_norm.update(transitions['g'])
                    # ==== finish update normaliser ====

                    # Update Loop
                    for _ in range(agent.params.num_updates):
                        transitions = replay_buffer.sample(agent.params.batch_size)
                        agent.update(transitions)

                    # sync networks
                    soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.tau)
                    soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.tau)

                """
                === After 1 epoch ===
                """
                # each epoch, we test the agent
                success_rate = test_Agent_HER(agent, env, n_trial=agent.params.test_episodes)
                tf.contrib.summary.scalar("Test Success Rate", success_rate, step=epoch)

                print("Epoch: {:03d}/{} | Train Success Rate: {:.3f} | Test Success Rate: {:.3f}".format(
                    epoch, agent.params.num_epochs, np.mean(np.array(successes)), success_rate
                ))

            print("=== Training is Done ===")
            test_Agent_HER(agent, env, n_trial=agent.params.test_episodes)
            env.close()


# in this algo, since the order of occurrence is important so that
# we don't use Experience Replay to randomly sample trajectory
def train_TRPO(agent, env, reward_buffer, summary_writer):
    get_ready(agent.params)

    global_timestep = tf.compat.v1.train.get_global_step()
    time_buffer = deque(maxlen=agent.params.reward_buffer_ep)
    log = logger(agent.params)
    init_state = env.reset()
    normaliser = RunningMeanStd(init_state.shape[0])
    total_ep = 0

    # init_normaliser(env, normaliser) # init normaliser's moments by going through some episodes before training

    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            while global_timestep < agent.params.num_frames:
                states, actions, rewards, = [], [], []
                for _ in range(agent.params.num_rollout):
                    state = env.reset()
                    normaliser.normalise(state)
                    total_reward = 0
                    start = time.time()
                    done = False
                    while not done:
                        # env.render()
                        action = agent.predict(state)
                        next_state, reward, done, info = env.step(action)
                        next_state = normaliser.normalise(next_state)

                        states.append(state)
                        actions.append(action)
                        # rewards.append(reward*0.0025) # reward scaling
                        rewards.append(reward)  # reward scaling

                        global_timestep.assign_add(1)
                        total_reward += reward
                        state = next_state

                    """
                    ===== After 1 Episode =====
                    """

                    total_ep += 1
                    reward_buffer.append(total_reward)
                    time_buffer.append(time.time() - start)

                    normaliser.update(np.array(states))
                    tf.contrib.summary.scalar("reward", total_reward, step=total_ep)
                    tf.contrib.summary.scalar("exec time", time.time() - start, step=total_ep)
                    tf.contrib.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=total_ep)

                """
                ===== After Rolling out of episodes is Done =====
                """
                # update the weights: inside it's got a for-loop and a stopping condition
                # so that if the value of KL-divergence exceeds some threshold, then we stop updating.
                loss = agent.update(states, actions, rewards)
                log.logging(global_timestep.numpy(), total_ep, np.sum(time_buffer), reward_buffer, np.mean(loss), 0,
                            [0])

                test_Agent_TRPO(agent, env)

                # check the stopping condition
                if global_timestep.numpy() > agent.params.num_frames:
                    print("=== Training is Done ===")
                    test_Agent_TRPO(agent, env, n_trial=agent.params.test_episodes)
                    env.close()
                    break


"""

Distributed Version of Training APIs

"""

import ray


def train_HER_ray(agent, env, replay_buffer, summary_writer):
    ray.init()
    get_ready(agent.params)
    global_timestep = tf.compat.v1.train.get_global_step()
    total_ep = 0

    with summary_writer.as_default():
        # for summary purpose, we put all codes in this context
        with tf.contrib.summary.always_record_summaries():

            for epoch in range(agent.params.num_epochs):
                successes = list()
                for cycle in range(agent.params.num_cycles):
                    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                    # for ep in range(agent.params.num_episodes):
                    agent_id = ray.put(agent)
                    env_id = ray.put(env)
                    tasks = [_inner_train_HER.remote(agent_id, env_id) for _ in range(agent.params.num_episodes)]

                    res = ray.get(tasks)
                    print(res)
                    # asdf

                    """
                    === After num_episodes ===
                    """
                    # convert them into arrays
                    mb_obs = np.array(mb_obs)
                    mb_ag = np.array(mb_ag)
                    mb_g = np.array(mb_g)
                    mb_actions = np.array(mb_actions)
                    replay_buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])

                    # ==== update normaliser ====
                    mb_obs_next = mb_obs[:, 1:, :]
                    mb_ag_next = mb_ag[:, 1:, :]
                    # get the number of normalization transitions
                    num_transitions = mb_actions.shape[1]
                    # create the new buffer to store them
                    buffer_temp = {'obs': mb_obs,
                                   'ag': mb_ag,
                                   'g': mb_g,
                                   'actions': mb_actions,
                                   'obs_next': mb_obs_next,
                                   'ag_next': mb_ag_next,
                                   }
                    transitions = replay_buffer.sample_func(buffer_temp, num_transitions)
                    # update
                    agent.o_norm.update(transitions['obs'])
                    agent.g_norm.update(transitions['g'])
                    # ==== finish update normaliser ====

                    # Update Loop
                    for _ in range(agent.params.num_updates):
                        transitions = replay_buffer.sample(agent.params.batch_size)
                        agent.update(transitions)

                    # sync networks
                    soft_target_model_update_eager(agent.target_actor, agent.actor, tau=agent.params.tau)
                    soft_target_model_update_eager(agent.target_critic, agent.critic, tau=agent.params.tau)

                """
                === After 1 epoch ===
                """
                # each epoch, we test the agent
                success_rate = test_Agent_HER(agent, env, n_trial=agent.params.test_episodes)
                tf.contrib.summary.scalar("Test Success Rate", success_rate, step=epoch)

                print("Epoch: {:03d}/{} | Train Success Rate: {:.3f} | Test Success Rate: {:.3f}".format(
                    epoch, agent.params.num_epochs, np.mean(np.array(successes)), success_rate
                ))

            print("=== Training is Done ===")
            test_Agent_HER(agent, env, n_trial=agent.params.test_episodes)
            env.close()


@ray.remote
def _inner_train_HER(agent, env):
    successes, mb_obs, mb_ag, mb_g, mb_actions = [], [], [], [], []
    state = env.reset()
    # obs, achieved_goal, desired_goal in `numpy.ndarray`
    obs, ag, dg, rg = state_unpacker(state)
    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
    success = list()
    for ts in range(agent.params.num_steps):
        # env.render()
        action = agent.predict(obs, dg)
        action = action_postprocessing(action, agent.params)

        next_state, _, _, info = env.step(action)

        # obs, achieved_goal, desired_goal in `numpy.ndarray`
        next_obs, next_ag, next_dg, next_rg = state_unpacker(next_state)

        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_g.append(dg.copy())
        ep_actions.append(action.copy())

        success.append(info.get('is_success'))
        obs = next_obs
        # rg = next_rg
        ag = next_ag

    """
    === After 1 ep ===
    """
    ep_obs.append(obs.copy())
    ep_ag.append(ag.copy())
    mb_obs.append(ep_obs)
    mb_ag.append(ep_ag)
    mb_g.append(ep_g)
    mb_actions.append(ep_actions)
    successes.append(success)
    return successes, mb_obs, mb_ag, mb_g, mb_actions
