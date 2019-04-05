import tensorflow as tf
import numpy as np

class AnnealingSchedule:
    def __init__(self, start=1.0, end=0.1, decay_steps=500):
        self.start       = start
        self.decay_steps = 500
        self.annealed_value = np.linspace(start, end, decay_steps)

    def get_value(self, timestep):
        return self.annealed_value[min(timestep, self.decay_steps - 1)]

def sync_main_target(sess, main, target):
    """
    Synchronise the models

    :param main:
    :param target:
    :return:
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(main.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)
        
    sess.run(update_ops)


def soft_target_model_update(sess, main, target, tau=1e-2):
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    :param main:
    :param target:
    :param tau:
    :return:
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(main.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(target.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):

        # θ_target = τ*θ_local + (1 - τ)*θ_target
        op = e2_v.assign(tau*e1_v + (1 - tau)*e2_v)
        update_ops.append(op)

    sess.run(update_ops)


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def ClipIfNotNone(grad, _min, _max):
    """
    reference => https://stackoverflow.com/a/39295309
    :param grad:
    :return:
    """
    if grad is None:
        return grad
    return tf.clip_by_value(grad, _min, _max)
