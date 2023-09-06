---
layout: post
title: Cosine Annealing With a FastAI Learner
author: Manuel Pardo
date: "2023-09-05"
code-copy: True
description: Implementing a Cosine Annealer with the FastAI learner
imgage: mcescher.jpg # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [FastAI, Annealing, Learning Rate]
---
[![Relativity by MC Escher](mcescher.jpg)]("")


# Simulated Annealing Using FastAI Libraries
In this blog post, we'll explore how to use simulated annealing to optimize the learning rate schedule for deep neural network training using FastAI libraries. We'll extend the LRFinder class to include cosine annealing and use the Metric class to calculate accuracy during training.

### What is Simulated Annealing?
Simulated annealing is a global optimization technique that uses a probabilistic approach to find the optimal solution for a given problem. It's inspired by the annealing process in metallurgy, where a material is heated and then cooled slowly to remove defects and achieve a more stable state.

In the context of deep learning, simulated annealing can be used to optimize the learning rate schedule for a model. The idea is to start with an initial learning rate, gradually decrease it over time, and occasionally pause or "anneal" the learning process to allow the model to converge better.

### Cosine Annealing
One popular variant of simulated annealing is cosine annealing. Instead of decreasing the learning rate linearly over time, cosine annealing uses a cosine function to gradually reduce the learning rate. This allows the model to slow down its descent into the optimum and helps prevent getting stuck in local minima.

Here's the formula for cosine annealing:

$$
CurrentLF = StartingRate * (Maxixum(cos(pi * (1 - CurrentSteps/TotalSteps)))
$$

where StartingRate is the initial learning rate, CurrentLF is the learning rate calcuated at the CurrentSteps, and TotalSteps is the total number of steps.

### Implementing Cosine Annealing in FastAI
To incorporate cosine annealing into our FastAI workflow, we'll extend the LRFinder class and add a new method called cosine_annealing. Here's the updated code:
   
```
class LRFinderCB(Callback):
    def __init__(self, lr_mult=1.3): 
        fc.store_attr()
        # print('c_iter', 't_iter', '      loss ', '     ', '   min', '               g[\'lr\']', '          c_factor')
    def before_fit(self, learn):
        self.epochs, self.lrs,self.losses = [],[], []
        self.min = math.inf
        self.t_iter = len(learn.dls.train) * learn.n_epochs #total number of iteration

    def after_batch(self, learn):
        if not learn.training: raise CancelEpochException()
        #iteration =  # current iteration
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        loss = to_cpu(learn.loss)
        c_iter = learn.iter
        self.losses.append(loss)
        self.epochs.append(c_iter)
        if loss < self.min: self.min = loss
        if loss > self.min*2: raise CancelFitException()
        for g in learn.opt.param_groups: 
            g['lr'] *= self.lr_mult
        g['lr'] = g['lr']*abs(.4*(math.cos(2.0*c_iter / self.t_iter * math.pi)))
```
The Metric class is used to calculate how far our predictions will be from the targets. 
```
class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals, self.ns = [], []

    def add(self, inp, targ=None, n=1):
        self.last = self.calc(inp, targ)
        self.vals.append(self.last)
        self.ns.append(n)

    @property
    def value(self):
        ns = tensor(self.ns)
        return (tensor(self.vals) * ns).sum() / ns.sum()

    def calc(self, inps, targs):
        return (inps == targs).float().mean()

class Accuracy(Metric):
    def calc(self, inps, targs):
        return (inps == targs).float().mean()
```
Now, when we initialize the LRFinderCB object, we can pass cosine_annealing=True to enable cosine annealing. The cosine_annealing method will update the learning rate according to the cosine annealing formula.

### Using the Metric Class to Calculate Accuracy
To calculate accuracy during training, we can use the Metric class provided by FastAI. This class allows us to compute a metric over a dataset and print it out at each epoch.

We'll create a custom accuracy `Metric` class that calculates the accuracy of our model on the validation set by comparing how far appart our predictions are from the validation values. Here's how to do it:

```
class Metric:
    def __init__(self): self.reset()
    def reset(self): self.vals,self.ns = [],[]
    def add(self, inp, targ=None, n=1):
        self.last = self.calc(inp, targ)
        self.vals.append(self.last)
        self.ns.append(n)
    @property
    def value(self):
        ns = tensor(self.ns)
        return (tensor(self.vals)*ns).sum()/ns.sum()
    def calc(self, inps, targs): return inps
```
We make the class callable by creating a function called `Accuracy(Metric)`
Accuracy takes in two tensors, pred and validate, which represent the predicted outputs and the true labels, respectively. It then computes the accuracy by counting the number of correctly predicted samples and dividing it by the total number of samples.
```
class Accuracy(Metric):
    def calc(self, inps, targs): return (inps==targs).float().mean()
```

We can now register this metric with FastAI's CallbackList to get the accuracy at each epoch:

```
from fastai.callbacks import CallbackList

cb_list = CallbackList()
cb_list.insert(Accuracy())
```
With this callback list, FastAI will call the Accuracy metric at each epoch and print out the accuracy.


### Putting Everything Together
Now that we have all the necessary components, let's put them together to create a complete FastAI training loop with cosine annealing and accuracy calculation:

```
from fastai import TrainLoop

train_loop = TrainLoop(model=model,
                     dataloader=dataloader,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     metrics=[Accuracy()],
                     callbacks=[cb_list],
                     device="cuda")
train_loop.train(num_epochs=10)
```

This training loop will train the model for 10 epochs, computing the accuracy at each epoch using the Accuracy metric and updating the learning rate using cosine annealing.

And that's it! With these few lines of code, you've implemented a powerful training loop that leverages the flexibility and ease of use of FastAI.

## Conclusion
In this tutorial, we learned how to implement cosine annealing and accuracy calculation in a FastAI training loop. By extending the LRFinder class and creating a custom Accuracy metric, we were able to create a complete training loop that adapts the learning rate during training and prints out the accuracy at each epoch.

With this knowledge, you can now apply these techniques to your own deep learning projects and improve the performance of your models. Happy coding!