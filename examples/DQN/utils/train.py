import gin
import time
import itertools
import tensorflow as tf
import numpy as np
from tf_rl.common.to_markdown import params_to_markdown
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
          interval_move_ave):
    time_buffer = list()
    log = logger(num_frames=num_frames, interval_move_ave=interval_move_ave)
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
                    loss, batch_loss = agent.update(states, actions, rewards, next_states, dones)

                # synchronise the target and main models by hard
                if (global_timestep.numpy() > hot_start) and (global_timestep.numpy() % sync_freq == 0):
                    agent.manager.save()
                    agent.target_model.set_weights(agent.main_model.get_weights())

            """
            ===== After 1 Episode is Done =====
            """
            with tf.name_scope("Train"):
                tf.compat.v2.summary.scalar("reward", total_reward, step=global_timestep.numpy())
                tf.compat.v2.summary.scalar("exec time", time.time() - start, step=global_timestep.numpy())
                if epoch >= interval_move_ave:
                    tf.compat.v2.summary.scalar("Moving Ave Reward", np.mean(reward_buffer), step=global_timestep.numpy())
                tf.compat.v2.summary.histogram("taken actions", cnt_action, step=global_timestep.numpy())

            # store the episode reward
            reward_buffer.append(total_reward)
            time_buffer.append(time.time() - start)

            if global_timestep.numpy() > hot_start and epoch % interval_move_ave == 0:
                log.logging(global_timestep.numpy(), np.sum(time_buffer), reward_buffer, np.mean(loss),
                            agent.policy.current_epsilon(), cnt_action)
                time_buffer = list()

            if agent.eval_flg:
                replay_buffer.save()
                eval_Agent(agent, env)
                agent.eval_flg = False

            # check the stopping condition
            if global_timestep.numpy() >= num_frames:
                print("=== Training is Done ===")
                eval_Agent(agent, env, n_trial=num_eval_episodes)
                env.close()
                break
