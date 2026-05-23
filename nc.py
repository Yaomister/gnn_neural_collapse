import math
import torch

def calculate_metrics(features : torch.Tensor, labels, num_classes):
    # features: (N, d)
    # labels: (d,)
    # num_classes: C

    features = features.float()
    feature_dimension = features.shape[1]

   
    class_means = []
    for c in range(num_classes):
        mask = (labels == c)
        class_means.append(features[mask].mean(dim = 0))
    class_means = torch.stack(class_means)


    global_mean = class_means.mean(dim = 0)

     # NC1
    within_class_covariance = torch.zeros(feature_dimension, feature_dimension)
    num_features = features.shape[0]
    for c in range(num_classes):
        mask = (labels == c)
        diff =  features[mask] - class_means[c]
        within_class_covariance += diff.T @ diff
    within_class_covariance /= num_features

    between_class_covaraince =torch.zeros(feature_dimension)
    for c in range(num_classes):
        diff = within_class_covariance[c] - global_mean
        between_class_covaraince += diff @ diff.T
    between_class_covaraince / num_classes

    total_covariance = within_class_covariance + between_class_covaraince

    # NC2
    maximal_spread_angle = -1/(num_classes - 1)

    normalized_features = [f/ f.sum(dim= 0) for f in features]

    spread = []

    for c1 in num_classes:
        for c2 in num_classes:
            cos_similarity = torch.dot(normalized_features[c1], normalized_features[c2])
            spread.append(cos_similarity - maximal_spread_angle)

    return within_class_covariance, between_class_covaraince, total_covariance, spread

            

    

        


    return 