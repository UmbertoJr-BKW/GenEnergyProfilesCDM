def unweighted_mse(output, epsilon):
    # Calculate the loss function
    loss = (output - epsilon).square().mean()
    return loss
