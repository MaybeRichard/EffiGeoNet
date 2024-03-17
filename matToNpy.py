import scipy.io
import numpy as np
from icecream import ic

for i in range(0,600):
    trainImgPath = './dataset/moving/train_flow'
    types = 'barrel'
    imgPath = '%s%s%s%s%s%s' % (trainImgPath, '/', types, '_', str(i).zfill(6), '.mat')
    print(imgPath)
    # Load the .mat file
    mat_file = scipy.io.loadmat(imgPath)
    ic(type(mat_file['v']))
    demo = np.array([mat_file['u'],mat_file['v']])
    npyPath = '%s%s%s%s%s%s' % ('./dataset/moving/train_flow', '/', types, '_', str(i).zfill(6), '.npy')
    print(npyPath)
    np.save(npyPath, demo)

