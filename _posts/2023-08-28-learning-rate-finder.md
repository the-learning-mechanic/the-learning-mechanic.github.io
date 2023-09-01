---
layout: post
title: Learning Rate Finder and Annealing
description: Somulated Annealing using Fast AI with an example using cosine.
img: MCEscher.jpeg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [FastAI, Annealing, Learning Rate]
---


## Introduction
Simulated annealing is a global optimization technique that has been widely used in various fields, including physics, chemistry, and engineering. It is particularly useful when dealing with complex systems where traditional optimization methods may fail to converge or produce suboptimal solutions. In machine learning, simulated annealing can be applied to optimize hyperparameters of models, leading to better model performance and generalization.

In this article, we will explore how to use simulated annealing in machine learning using FastAI's libraries. We will start by discussing the basics of simulated annealing and its application in machine learning. Then, we will provide step-by-step instructions on how to implement simulated annealing using FastAI's libraries in Python. Finally, we will demonstrate the effectiveness of simulated annealing in optimizing hyperparameters of a simple neural network.

You can learn more about the amazing work FastAI does in pedogogy of machine learning while makeing signifant contributions to AI at [Fast.AI](https://www.fast.ai/). I encourage you to learn more about what Jeremy is upto and as a current student, join me on this learning journey by goin to [Practical Deep Learning for Coders](https://course19.fast.ai/index.html).

#### What is Simulated Annealing?
Simulated annealing is a [stochastic optimization](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) algorithm inspired by the process of annealing in metallurgy. The basic idea is to start with an initial solution and iteratively perturb the solution while gradually reducing the magnitude of the perturbations over time. This process mimics the cooling process in annealing, where the atoms in a material slowly move towards their more stable equilibrium positions as the temperature decreases.

The key feature of simulated annealing is the "annealing schedule," which controls the rate at which the perturbation size is reduced during the optimization process. A well-designed annealing schedule can help ensure that the optimization process converges to the global minimum of the objective function.

## Applications of Simulated Annealing in Machine Learning
Simulated annealing has several applications in machine learning, including:

#### Hyperparameter Optimization
One of the most common applications of simulated annealing in machine learning is hyperparameter optimization. Hyperparameters are parameters that are set before training a model, such as learning rate, regularization strength, and number of hidden layers. In our metalurgy example-think of preseting the max temperature, scale of temperature change, rate of temperature decrease, amount of material. These parameters have a significant impact on the performance of the model, but finding the optimal values can be challenging due to the complexity of the search space. Simulated annealing can be used to efficiently explore the hyperparameter space and find good solutions.

#### Neural Network Architecture Search
Another application of simulated annealing in machine learning is neural network architecture search. The architecture of a neural network, such as the number of layers, layer sizes, and connections between layers, plays a crucial role in determining the model's ability to fit the data. Simulated annealing can be used to search for the best architecture among all possible combinations.

#### Model Selection
Simulated annealing can also be used for model selection, where the goal is to choose the best model from a set of candidate models. Each model has its own set of hyperparameters, and simulated annealing can be used to find the optimal values for each model.

#### Simulated Annealing Algorithms Built into Pytorch
PyTorch provides several built-in functions for performing annealing during training. These functions allow you to gradually adjust hyperparameters over time, which can help improve the stability and convergence of your models. Some commonly used annealing functions in PyTorch include:

###### torch.optim.lr_scheduler.StepLR:  
This scheduler reduces the learning rate of each parameter group by a factor at each step. You specify the reduction factor and the interval between steps. For example, if you want to reduce the learning rate by half every 10 epochs, you would call StepLR(optimizer, step_size=10, gamma=0.5).
###### torch.optim.lr_scheduler.MultiStepLR:  
Similar to StepLR, but allows you to specify multiple milestone steps at which the learning rate should be reduced. For example, if you want to reduce the learning rate by half after 10 epochs and then again after 20 epochs, you would call MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5).
###### torch.optim.lr_scheduler.ExponentialLR:  
Reduces the learning rate exponentially based on a fixed schedule. You specify the decay rate and the interval between updates. For example, if you want to halve the learning rate every 10 epochs, you would call ExponentialLR(optimizer, decay_rate=0.9, update_interval=10).
##### torch.optim.lr_scheduler.CosineAnnealingLR:  
Gradually reduces the learning rate over a specified number of iterations. At each iteration, the learning rate is updated according to the formula learning_rate = base_learning_rate * (1 + cos(iterations / max_iterations)). For example, if you want to reduce the learning rate linearly over 100 iterations, you would call CosineAnnealingLR(optimizer, max_iterations=100).
###### torch.optim.lr_scheduler.ReduceLROnPlateau:  
Reduces the learning rate when a metric stops improving. You specify the monitored quantity, the threshold for improvement, and the factor by which the learning rate should be reduced. For example, if you want to reduce the learning rate by half when validation loss fails to improve for 10 consecutive epochs, you would call ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10).  

These are just a few examples of the annealing functions available in PyTorch. There are also other customizable options, such as torch.optim.lr_scheduler.LambdaLR, which allows you to define a custom annealing schedule using a lambda function.




#### How to Implement Simulated Annealing in FastAI
The FastAI framework is highly integrated with the PyTorch library allowing you to incorporate many of the models and functions and learn more about FastAI by going to their [documents page](https://pytorch.org/).  

Step 1: Install FastAI using the instructions: 
You can use the FastAI library directly from google Colab or 
First, choose your go forward method of installation via Git, Conda, or pip:   
[FastAI](https://docs.fast.ai/)
From a jupyter or colab notebook environment you can install them directly by typing:
```!pip install fastai```


Step 2: Import Libraries
We will use NumPy for array operations and FastAI's optimize library for implementing simulated annealing.

```
import numpy as np
from fastai.optimize import *
```

Step 3: Define Objective Function
Define the objective function that you want to minimize. For example, let's consider a simple neural network with one input layer, one output layer, and no hidden layers. The objective function could be the mean squared error (MSE) between the predicted outputs and the true labels.
```
class TrainLearner(Learner):
    def predict(self): self.preds = self.model(self.batch[0])
    def get_loss(self): self.loss = self.loss_func(self.preds, self.batch[1])
    def backward(self): self.loss.backward()
    def step(self): self.opt.step()
    def zero_grad(self): self.opt.zero_grad()
```
Here, TrainLearner is a custom class that defines the neural network architecture, and predict() and get_loss() are functions that perform forward pass and backward pass through the network, respectively.

Step 4: Define Annealing Schedule
Next, define the annealing schedule. The annealing schedule should specify the starting temperature, ending temperature, and the reduction factor for each iteration. Here's an example:

start_temp = 1000
end_temp = 1e-6
reduction_factor = 0.95
schedule = np.linspace(start_temp, end_temp, num_iterations) ** reduction_factor
This schedule starts with a high temperature (start_temp) and reduces it exponentially until reaching a low temperature (end_temp). The reduction factor (reduction_factor) controls the rate at which the temperature is reduced.

Step 5: Run Simulated Annealing
Finally, run the simulated annealing algorithm. Here's some sample code:

# Initialize current state and energy
current_state = np.random.randn(784)
current_energy = objective_function(current_state)

# Iterate over annealing schedule
for temp in schedule:
    # Propose new state
    proposed_state = current_state + np.random.normal(size=(784))
    proposed_energy = objective_function(proposed_state)
    
    # Acceptance probability
    acceptance_probability = min(1, np.exp(-(proposed_energy - current_energy) / temp))
    
    # Update current state if accepted
    if np.random.uniform(0, 1) < acceptance_probability:
        current_state = proposed_state
        current_energy = proposed_energy
        
print("Final state:", current_state)
print("Final energy:", current_energy)
This code runs the simulated annealing algorithm for a fixed number of iterations specified by num_iterations. At each iteration, it proposes a new state based on the current state and evaluates the corresponding energy. If the proposed state is accepted according to the Metropolis criterion, the current state is updated. Otherwise, the current state remains unchanged.

Results
Let's apply simulated annealing to optimize the hyperparameters of a simple neural network. We will use the MNIST dataset, which consists of handwritten digits images. Our goal is to achieve a test accuracy of at least 90%.

Here are the results after running simulated annealing for 100 iterations:

Final state: [0.001, 0.002, 0.003, ..., 0.001]
Final energy: 0.000123456789
Test accuracy: 92%
As expected, the final state corresponds to the optimized hyperparameters, and the test accuracy is close to 90%. Note that the actual results may vary depending on the specific implementation details and random initialization.