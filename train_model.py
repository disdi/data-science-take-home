import os
#import _pickle as cPickle
import cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")


#path to training data
source   = "voxforge_dataset/wav/"
dest   = "voxforge_dataset/model/"

# Extracting features
features = np.asarray(())

for file in os.listdir(source):
    # read the audio
    path = source + file
    sr,audio = read(path)

    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(audio,sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
gmm.fit(features)

model_dir = os.path.join(dest, "gmm")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# dumping the trained gaussian model
picklefile = model_dir + "vox.gmm"
cPickle.dump(gmm,open(picklefile,'w'))
print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
features = np.asarray(())

