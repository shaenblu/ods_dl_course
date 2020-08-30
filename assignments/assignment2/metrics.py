def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    #raise Exception("Not implemented!")
    accuracy = 0

    t = 0
    f = 0
    
    for i in range(prediction.shape[0]):
        if (prediction[i] == ground_truth[i]):
            t += 1
        else:
            f += 1
    accuracy = t / float(t + f)
    
    return accuracy
