def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    for i in range(prediction.shape[0]):
        if (prediction[i] and ground_truth[i]):
            tp += 1
        elif (prediction[i] and not ground_truth[i]):
            fp += 1
        elif (not prediction[i] and ground_truth[i]):
            fn += 1
        elif (not prediction[i] and not ground_truth[i]):
            tn += 1

    precision = (tp / float(tp + fp))
    recall = (tp / float(tp + fn))
    f1 = (2* ((precision * recall) / (precision + recall)))
    accuracy = (tp + tn) / float(tp + fp + fn + tn)
    
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
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
