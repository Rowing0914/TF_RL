## Introduction


## Hyper-params
- `dopamine.gin`: Dopamine's hyper-params, [[Ref]](https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/configs/dqn.gin)
- `nature_dqn.gin`: Mnih et al. (2015)
- Based on the common setting of training as in `domapine.gin`, I have examined some possibilities below
    - `experimental/adam_huber.gin`
    - `experimental/adam_mse.gin`
    - `experimental/rmsprop_mse.gin`
    - Since `dopamine.gin` uses `RMSProp` with Huber Loss func, I didn't add it here!