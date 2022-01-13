from sklearn.mixture import GaussianMixture


def create_gmm(features) -> GaussianMixture:
    """Create GMM model with given features for voice recognising task.

    :param features: Features to fit (typically MFCC)
    :type features: array-like
    :return: Configured GMM model
    :rtype: GaussianMixture
    """

    gmm = GaussianMixture(
        n_components=8,
        max_iter=200,
        covariance_type='diag',
        n_init=3)
    
    gmm.fit(features)

    return gmm