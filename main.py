import os
import numpy as np

from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture

from voice_processing import get_mfcc, get_mfcc_test
from ml import create_gmm


PATH_TRAIN = './dataset/train_data/youtube/{}'
PATH_TEST  = './dataset/test_data/AudioSet/{}'

MALE   = 'male'
FEMALE = 'female'


def get_score(filename: str, gmm: GaussianMixture) -> float:
    """Gets the log likelihood score for a given audio file name.

    :param filename: Audio file name
    :type filename: str
    :param gmm: GMM model to use for calculations
    :type gmm: GaussianMixture
    :return: Log likelihood score
    :rtype: float
    """
    
    sample_rate, audiofile = read(filename)
    features = get_mfcc_test(sample_rate, audiofile)
    
    scores = np.array(gmm.score(features))
    log_likelihood = scores.sum()

    return log_likelihood

def train_for_gender(gender: str) -> GaussianMixture:
    """Trains GMM model and returns it for a specified gender.

    :param gender: Gender to be trained for
    :type gender: str
    :return: Trained GMM model
    :rtype: GaussianMixture
    """

    assert(not (gender == MALE and gender == FEMALE))

    # Get the training files' paths
    training_filenames = [os.path.join(PATH_TRAIN.format(gender), filename)
        for filename in os.listdir(PATH_TRAIN.format(gender)) if filename.endswith('.wav')]

    # Harvest the features
    features = np.array([])
    for filename in training_filenames:
        sample_rate, audiofile = read(filename)
        vector = get_mfcc(sample_rate, audiofile)

        if features.size != 0:
            features = np.vstack((features, vector))
        else:
            features = vector

    gmm = create_gmm(features)

    return gmm

def test_for_gender(gender: str, gmms: list[GaussianMixture], labels: list[str]) -> np.ndarray:
    """Runs tests on the model and gives results back.

    :param gender: Gender to test for
    :type gender: str
    :param gmms: GMM models between which to choose
    :type gmms: list[GaussianMixture]
    :param labels: Labels associated with GMM models
    :type labels: list[str]
    :return: Array of induced labels
    :rtype: np.ndarray
    """

    testing_filenames = [os.path.join(PATH_TEST.format(gender), filename)
        for filename in os.listdir(PATH_TEST.format(gender)) if filename.endswith('.wav')]

    results = np.array([])

    for filename in testing_filenames:
        log_likelihoods = np.array([])

        for gmm in gmms:
            log_likelihoods = np.r_[log_likelihoods, get_score(filename, gmm)]

        winner = np.argmax(log_likelihoods)
        results = np.append(results, labels[winner])

    return results


def main():
    gmm_male   = train_for_gender(MALE)
    gmm_female = train_for_gender(FEMALE)

    results_female = test_for_gender(FEMALE, [gmm_male, gmm_female], [MALE, FEMALE])
    results_male   = test_for_gender(MALE,   [gmm_male, gmm_female], [MALE, FEMALE])

    print(f'Female accuracy: {len(results_female[results_female == FEMALE]) / len(results_female)}')
    print(f'Male accuracy:   {len(results_male[results_male == MALE]) / len(results_male)}')

if __name__ == '__main__':
    main()