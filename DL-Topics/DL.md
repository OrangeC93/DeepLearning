# Neural Networks
## Regularization:
-  dropout: randomly turn off certain parts of our network while training
    -  decide that during this epoch or this mini-batch, a bunch of neurons will be turned off
    -  assign each neuron with a probability ps at random
    -  decide of the threshold: a value that will determine if the node is kept or not
    -  dropout is only used during training time, during test time, all the nodes of our network are always present
-  data augmentation: instead of feeding the model with the same pictures every time, we do small random transformations (a bit of rotation, zoom, translation, etc…) that doesn’t change what’s inside the image (for the human eye) but changes its pixel values. 
-  weight decay: multiply the sum of squares with another smaller number to the loss function
    -  Loss = MSE(y_hat, y) + wd * sum(w^2), wd is the weight decay
    -  w(t) = w(t-1) - lr * dLoss / dw => d(wd * w^2) / dw = 2 * wd * w (similar to dx^2/dx = 2x)
    -  generally a wd = 0.1 works pretty well
 
1. https://becominghuman.ai/this-thing-called-weight-decay-a7cd4bcfccab

## adversarial-example
1. https://openai.com/blog/adversarial-example-research/


## GAN
```
How to train Discriminator
Goal: D(r) -> 1 & D(G(f)) ->0 
How: MIN(loss(D(r), r_label) + loss(D(G(f1), f_label))

How to train Generator
Goal: D(G(f2)) -> 1 or MAX(log(D(G(f2)))
How: MIN(loss(D(G(f2)), r_label))
```