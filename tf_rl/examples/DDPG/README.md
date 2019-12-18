## Introduction

DDPG implementation




## Usage

- TensorFlow
    ```shell script
    python tf_rl/examples/DDPG/main.py --gin_file=./tf_rl/examples/DDPG/config/main.gin \ 
                                       --gin_params="train_eval.env_name='Ant-v2'"
    ```

- To override the existing `gin_file` by command line args, pls check [here](https://github.com/google/gin-config/blob/master/docs/index.md#experiments-with-multiple-gin-files-and-extra-command-line-bindings)
