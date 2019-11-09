import gin
import time
import itertools
import tensorflow as tf
import numpy as np
from tf_rl.common.to_markdown import params_to_markdown
from tf_rl.common.utils import logger
from tf_rl.examples.DQN.utils.eval_agent import eval_Agent


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
          interval_MAR,
          google_colab):
    time_buffer = list()
    log = logger(num_frames=num_frames, interval_MAR=interval_MAR)
    with summary_writer.as_default():
        tf.compat.v2.summary.text(name="Hyper-params",
                                  data=params_to_markdown(gin.operative_config_str()),
                                  step=0)
        for epoch in itertools.count():
            state = env.reset()
            total_reward = 0
            start = time.time()
            cnt_action = list()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)

                global_timestep.assign_add(1)
                total_reward += reward
                state = next_state
                cnt_action.append(action)

                # for evaluation purpose
                if global_timestep.numpy() % eval_interval == 0:
                    agent.eval_flg = True

                if (global_timestep.numpy() > hot_start) and (global_timestep.numpy() % train_freq == 0):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    agent.update(states, actions, rewards, next_states, dones)

                # synchronise the target and main models by hard
                if (global_timestep.numpy() > hot_start) and (global_timestep.numpy() % sync_freq == 0):
                    agent.manager.save()
                    agent.target_model.set_weights(agent.main_model.get_weights())

            """
            ===== After 1 Episode is Done =====
            """
            ts = global_timestep.numpy()
            tf.compat.v2.summary.scalar("train/reward", total_reward, step=ts)
            tf.compat.v2.summary.scalar("train/exec_time", time.time() - start, step=ts)
            if ts > hot_start:
                tf.compat.v2.summary.scalar("train/MAR", np.mean(reward_buffer), step=ts)
            tf.compat.v2.summary.histogram("train/taken actions", cnt_action, step=ts)

            # store the episode reward
            reward_buffer.append(total_reward)
            time_buffer.append(time.time() - start)

            if ts > hot_start and epoch % interval_MAR == 0:
                log.logging(time_step=ts,
                            exec_time=np.sum(time_buffer),
                            reward_buffer=reward_buffer,
                            epsilon=agent.policy.current_epsilon())
                time_buffer = list()

            if agent.eval_flg:
                # replay_buffer.save()
                score = eval_Agent(agent, env, google_colab=google_colab)
                tf.compat.v2.summary.scalar("eval/Score", score, step=ts)
                agent.eval_flg = False

            # check the stopping condition
            if ts >= num_frames:
                print("=== Training is Done ===")
                score = eval_Agent(agent, env, n_trial=num_eval_episodes, google_colab=google_colab)
                tf.compat.v2.summary.scalar("eval/Score", score, step=ts)
                env.close()
                break
