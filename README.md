# BiolHessRNN

This repository (in TensorFlow 1.12) computes the leading Hessian eigenvalue for recurrent neural networks (RNNs). It works for RNNs trained using gradient descent via backpropagation through time or using approximate gradient descent via a three-factor rule that is biologically-plausible. 

The code is developed on top of the LSNN repository (https://github.com/IGITUGraz/LSNN-official) [1]. Major additions to that repository are as follows:
1. Computing the leading eigenvalue of loss' Hessian matrix for RNNs in TensorFlow. 
2. Ensuring that gradient approximations (implemented with automatic differentiation) are applied only during training and not during the Hessian computation.   

## Usage

The main code is in the `bin/` folder. You can use the following command to run:
``sh run_saveHessRNN.sh``

The command above runs ``saveHess_seqMNIST.py``, which contains the code to setup and train a RNN as well as to compute and save leading loss' Hessian eigenvalue. Each run should take about 2 hours to complete. Because run_saveHessRNN.sh runs both BPTT and three-factor learning five times each in series, all 10 runs should be done within 20 hours. One can also perform these runs in parallel to save time. 

The folder 'lsnn/' contains the source code retained from the lsnn package [1]. 

## Installation

The installation instruction is copied from (https://github.com/IGITUGraz/LSNN-official), and please refer to that repository for troubleshooting steps. The code is compatible with python 3.4 to 3.7 and tensorflow 1.7 to 1.12 (CPU and GPU versions).

> You can run the training scripts **without installation** by temporarily including the repo directory
> in your python path like so: `` PYTHONPATH=. python3 bin/tutorial_sequential_mnist_with_LSNN.py`` 

From the main folder run:  
`` pip3 install --user .``  
To use GPUs one should also install it:
 ``pip3 install --user tensorflow-gpu``.

## License

Please refer to LICENSE for copyright details


## References

[1] Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, and Wolfgang Maass. “Long short-term memory and learning-to-learn in networks of spiking neurons”. In: 32nd Conference on Neural Information Processing Systems. 2018, pp. 787–797.
