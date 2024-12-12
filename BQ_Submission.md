### Q1
### Q2
### Q3: What are the differences between batch gradient descent, stochastic gradient descent (SGD),and mini-batch gradient descent?

| **Method**                | **Update Frequency**      | **Efficiency**          | **Convergence Stability**   | **Best For**                              |
|---------------------------|---------------------------|--------------------------|-----------------------------|------------------------------------------|
| Batch Gradient Descent    | After entire dataset      | Slow (large datasets)    | High (smooth updates)       | Small datasets                           |
| Stochastic Gradient Descent | After each data point     | Fast                     | Low (oscillations)          | Large/streaming datasets                 |
| Mini-Batch Gradient Descent | After each mini-batch     | Balanced                 | Medium (more stable than SGD) | Large datasets, general-purpose training |

### Q4: Explain the difference between L1 and L2 regularization. What impact do they have on model weights? 
| **Aspect**          | **L1 Regularization**           | **L2 Regularization**          |
|----------------------|----------------------------------|---------------------------------|
| **Effect on Weights**| Encourages sparsity (zeros)     | Shrinks weights uniformly      |
| **Feature Selection**| Performs implicit feature selection by zeroing out irrelevant features | Retains all features but reduces their magnitudes |
| **Optimization**     | Leads to sparse gradients (non-differentiable at zero) | Smooth gradients (always differentiable) |
| **Use Case**         | When feature selection is needed | When all features are important 