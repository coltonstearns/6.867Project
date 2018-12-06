Added an extra convolutional layer in between a section's output and it becoming a skip-connection input, to allow for
more freedom on skip-connections in upsampling.

The 2 class converged model performs poorly --> need to train it longer still to let it fully converge.
Discovered that greatly regularizing via early stopping made a much better network on 2 class with about 83% accuracy.