## Intro
this is the tool to extract log data from tensorboard-formated log data and convert them into `Numpy Ndarray` then plot them by `matplotlib` in two different ways, with or without target(goal) line.

## Usage
1. Put the log files in `logs` directory
```shell
# Example directory architecture
.
├── logs
│   ├── Ant-v2
│   │   └── events.out.tfevents.1559306782.noio0925.v2
│   ├── HalfCheetah-v2
│   │   └── events.out.tfevents.1559306871.noio0925.v2
│   └── Hopper-v2
│       └── events.out.tfevents.1559306885.noio0925.v2
├── README.md
├── result_graphs
│   ├── without_target_line
│   └── with_target_line
└── visualise.py

```
2. Confirm the existence of two directories under `result_graphs`, which is `with_target_line` and `without_target_line`
3. Run `$ python3.6 visualise.py`
