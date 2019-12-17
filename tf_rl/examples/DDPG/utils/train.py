import gin
import time
import itertools
import tensorflow as tf
import numpy as np
from tf_rl.common.to_markdown import params_to_markdown
from tf_rl.common.utils import logger
from tf_rl.examples.DDPG.utils.eval_agent import eval_Agent


@tf.function
def soft_target_model_update_eager(target, source, tau=1e-2):
    """
    Soft update model parameters.
    target = tau * source + (1 - tau) * target

    :param main:
    :param target:
    :param tau:
    :return:
    """

    for param, target_param in zip(source.weights, target.weights):
        target_param.assign(tau * param + (1 - tau) * target_param)


def train(agent,
          env,
          replay_buffer,
          reward_buffer,
          summary_writer,
          num_eval_episodes,
          num_frames,
          tau,
          eval_interval,
          hot_start,
          batch_size,
          interval_MAR,
          log_dir,
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
            agent.random_process.reset_states()
            done = False
            episode_len = 0
            while not done:
                if agent.global_ts.numpy() < hot_start:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                # scale for execution in env (in DDPG, every action is clipped between [-1, 1] in agent.predict)
                next_state, reward, done, info = env.step(action * env.action_space.high)
                replay_buffer.add(state, action, reward, next_state, done)

                """
                === Update the models
                """
                if agent.global_ts.numpy() > hot_start:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    loss = agent.update(states, actions, rewards, next_states, dones)
                    soft_target_model_update_eager(agent.target_actor, agent.actor,
                                                   tau=tau)
                    soft_target_model_update_eager(agent.target_critic, agent.critic,
                                                   tau=tau)

                agent.global_ts.assign_add(1)
                episode_len += 1
                total_reward += reward
                state = next_state

                # for evaluation purpose
                if agent.global_ts.numpy() % eval_interval == 0:
                    agent.eval_flg = True

            """
            ===== After 1 Episode is Done =====
            """
            # save the updated models
            agent.actor_manager.save()
            agent.critic_manager.save()

            # store the episode related variables
            reward_buffer.append(total_reward)
            time_buffer.append(time.time() - start)

            # logging on Tensorboard
            ts = agent.global_ts.numpy()
            tf.compat.v2.summary.scalar("train/reward", total_reward, step=ts)
            tf.compat.v2.summary.scalar("train/exec_time", time.time() - start, step=ts)
            if ts > hot_start:
                tf.compat.v2.summary.scalar("train/MAR", np.mean(reward_buffer), step=ts)

            # logging
            if ts > hot_start and epoch % interval_MAR == 0:
                log.logging(time_step=ts,
                            exec_time=np.sum(time_buffer),
                            reward_buffer=reward_buffer,
                            epsilon=0)
                time_buffer = list()

            if agent.eval_flg:
                score = eval_Agent(agent, env, log_dir=log_dir, google_colab=google_colab)
                tf.compat.v2.summary.scalar("eval/Score", score, step=ts)
                agent.eval_flg = False

            # check the stopping condition
            if ts >= num_frames:
                print("=== Training is Done ===")
                score = eval_Agent(agent, env, n_trial=num_eval_episodes, log_dir=log_dir, google_colab=google_colab)
                tf.compat.v2.summary.scalar("eval/Score", score, step=ts)
                env.close()
                break
