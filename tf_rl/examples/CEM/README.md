## Introduction

CEM implementation of Numpy




## Usage

- TensorFlow
    ```shell script
    python tf_rl/examples/CEM/main.py
    ```



## Result

<img src="./images/CartPole-v0.png" width="45%" height="45%"> <img src="./images/CartPole-v1.png" width="45%" height="45%">



## Overview of algorithm

1. Generation of a sample of random data (trajectories, vectors, etc.) according to a specified random mechanism.
2. Updating the parameters of the random mechanism, typically parameters of pdfs, on the basis of the data, in order to produce a “better” sample in the next iteration.



## Reference
- [The Cross Entropy method for Fast Policy Search, S.Mannor et al., 2003](https://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf)
- [Base code](https://gist.github.com/andrewliao11/d52125b52f76a4af73433e1cf8405a8f)