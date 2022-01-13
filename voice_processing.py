import numpy as np

from python_speech_features import mfcc
from sklearn.preprocessing import scale


def get_mfcc(sample_rate, audio):
    """Extract MFCC from audio file.

    :param sample_rate: Audio file sample rate
    :type sample_rate: int
    :param audio: Array containing audio data
    :type audio: ndarray
    """

    features = mfcc(
        audio,
        samplerate=sample_rate,
        winlen=0.025,
        winstep=0.01,
        numcep=13,
        appendEnergy=False)
    
    features = scale(features)
    
    return features

def get_mfcc_test(sample_rate, audio):
    """Extract MFCC features from audio file discarding features with
    nan values in them.

    :param sample_rate: Audio file sample rate
    :type sample_rate: int
    :param audio: Array containing audio data
    :type audio: ndarray
    """
    
    features = get_mfcc(sample_rate, audio)
    new_features = np.array([])

    for i in range(features.shape[0]):
        temp_feature = features[i, :]
        
        if np.isnan(np.amin(temp_feature)):
            continue
        
        if new_features.size != 0:
            new_features = np.vstack((new_features, temp_feature))
        else:
            new_features = temp_feature
    
    new_features = scale(new_features)

    return new_features