import torch
import numpy as np

def calculate_metrics(features : torch.Tensor, labels, num_classes):
    # features: (N, d), N is the number of feature vectors (the batch size), d is the embedding dimensions
    # labels: (N,)
    # num_classes: C

    features = features.float()
    feature_dimension = features.shape[1]

    class_means = []
    for c in range(num_classes):
        mask = (labels == c)
        # take the average across rows, which gives the centroid for that class
        class_means.append(features[mask].mean(dim = 0))
    # concatnated the class means along dimension 0
    class_means = torch.stack(class_means)

    # take all the centroids and average them across rows, you get the global mean
    global_mean = class_means.mean(dim = 0)

     # NC1
    within_class_covariance = torch.zeros(feature_dimension, feature_dimension)
    num_features = features.shape[0] # N
    for c in range(num_classes):
        mask = (labels == c)
        diff =  features[mask] - class_means[c]
        # diff is (N, d), and we want (d, d)
        within_class_covariance += diff.T @ diff
    # average across number of features
    within_class_covariance /= num_features

    between_class_covaraince =torch.zeros(feature_dimension,feature_dimension)
    for c in range(num_classes):
        diff = class_means[c] - global_mean
        # unsqueeze so we can have (d, 1)
        diff = diff.unsqueeze(1)
        # we want (d, d) again
        between_class_covaraince += diff @ diff.T
    between_class_covaraince / num_classes

    between_class_covaraince_partial_inverse = torch.linalg.pinv(between_class_covaraince)

    nc1 = torch.trace(within_class_covariance @ between_class_covaraince_partial_inverse).item() / num_classes

    # NC2
    centered_class_means = class_means - global_mean
    # make the vectors normal
    normallized_class_means = centered_class_means / centered_class_means.norm(dim=1, keepdim=True)
    maximal_spread_angle = -1/(num_classes - 1)


    angle_spread = []

    for c1 in range(num_classes):
        for c2 in range(num_classes):
            if c1 == c2:
                continue
            cos_similarity = torch.dot(normallized_class_means[c1], normallized_class_means[c2])
            angle_spread.append(abs(cos_similarity - maximal_spread_angle))

    nc2_angle_spread = np.mean(angle_spread)

    norms = centered_class_means.norm(dim=1)
    nc2_norm_std = np.std(norms).item()


    return {"within_class_variance": nc1}, {"class_mean_angles" : nc2_angle_spread, "class_mean_norms": nc2_norm_std}

        