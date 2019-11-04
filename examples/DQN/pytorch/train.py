import time
import itertools
import numpy as np
from tf_rl.common.utils import logger
from examples.DQN.utils.eval_agent import eval_Agent


def train(global_timestep,
          agent,
          env,
          replay_buffer,
          reward_buffer,
          summary_writer,
          num_eval_episodes,
          num_frames,
          eval_interval,
          hot_start,
          train_freq,
          batch_size,
          sync_freq,
          interval_MAR):
    time_buffer = list()
    log = logger(num_frames=num_frames, interval_MAR=interval_MAR)
    for epoch in itertools.count():
        state = np.array(env.reset())
        total_reward = 0
        start = time.time()
        cnt_action = list()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)
            replay_buffer.add(state, action, reward, next_state, done)

            global_timestep += 1
            agent.timestep = global_timestep
            total_reward += reward
            state = next_state
            cnt_action.append(action)

            # for evaluation purpose
            if global_timestep % eval_interval == 0:
                agent.eval_flg = True

            if (global_timestep > hot_start) and (global_timestep % train_freq == 0):
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                agent.update(states, actions, rewards, next_states, dones)

            # synchronise the target and main models by hard
            if (global_timestep > hot_start) and (global_timestep % sync_freq == 0):
                agent.save()
                agent.sync_network()

        """
        ===== After 1 Episode is Done =====
        """
        summary_writer.add_scalar("train/reward", total_reward, global_timestep)
        summary_writer.add_scalar("train/exec_time", time.time() - start, global_timestep)
        if global_timestep > hot_start:
            summary_writer.add_scalar("train/MAR", np.mean(reward_buffer), global_timestep)
        summary_writer.add_histogram("train/taken actions", np.array(cnt_action), global_timestep)

        # store the episode reward
        reward_buffer.append(total_reward)
        time_buffer.append(time.time() - start)

        if global_timestep > hot_start and epoch % interval_MAR == 0:
            log.logging(time_step=global_timestep,
                        exec_time=np.sum(time_buffer),
                        reward_buffer=reward_buffer,
                        epsilon=agent.policy.current_epsilon(global_timestep))
            time_buffer = list()

        if agent.eval_flg:
            # replay_buffer.save()
            score = eval_Agent(agent, env)
            summary_writer.add_scalar("eval/Score", score, global_timestep)
            agent.eval_flg = False

        # check the stopping condition
        if global_timestep >= num_frames:
            print("=== Training is Done ===")
            score = eval_Agent(agent, env, n_trial=num_eval_episodes)
            summary_writer.add_scalar("eval/Score", score, global_timestep)
            env.close()
            break