import numpy as np
import skimage
import skimage.io as io
from skimage.transform import rescale
import scipy.io as scio
import distortion_model
import argparse
import os
from icecream import ic
# For parsing commandline arguments

parser = argparse.ArgumentParser()
parser.add_argument("--sourcedir", type=str, default='../dataset/fixed')
parser.add_argument("--datasetdir", type=str, default=r'C:\Users\SiChengLi\Documents\Code\ZJU\GeoProj\dataset\moving')
parser.add_argument("--trainnum", type=int, default=600, help='number of the training set')
parser.add_argument("--testnum", type=int, default=200, help='number of the test set')
args = parser.parse_args()

if not os.path.exists(args.datasetdir):
    os.mkdir(args.datasetdir)

trainDisPath = args.datasetdir + '/train_distorted'
trainUvPath  = args.datasetdir + '/train_flow'
testDisPath = args.datasetdir + '/test_distorted'
testUvPath  = args.datasetdir + '/test_flow'

if not os.path.exists(trainDisPath):
    os.mkdir(trainDisPath)
    
if not os.path.exists(trainUvPath):
    os.mkdir(trainUvPath)
    
if not os.path.exists(testDisPath):
    os.mkdir(testDisPath)

if not os.path.exists(testUvPath):
    os.mkdir(testUvPath)


def generatedata(types, k, trainFlag):
    print(types, trainFlag, k)

    width = 512
    height = 512

    parameters = distortion_model.distortionParameter(types)

    OriImg = io.imread('%s%s%s%s' % (args.sourcedir, '/', str(k).zfill(6), '.png'))

    disImg = np.array(np.zeros(OriImg.shape), dtype=np.uint8)
    u = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)
    v = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)

    cropImg = np.array(np.zeros((int(height / 2), int(width / 2), 3)), dtype=np.uint8)
    crop_u = np.array(np.zeros((int(height / 2), int(width / 2))), dtype=np.float32)
    crop_v = np.array(np.zeros((int(height / 2), int(width / 2))), dtype=np.float32)

    # crop range
    xmin = int(width * 1 / 4)
    xmax = int(width * 3 / 4 - 1)
    ymin = int(height * 1 / 4)
    ymax = int(height * 3 / 4 - 1)

    for i in range(width):
        for j in range(height):

            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)

            if (0 <= xu < width - 1) and (0 <= yu < height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j

                # Bilinear interpolation
                """
                Bi-linear interpolation
                Q11  Q12
                Q21  Q22
                """
                Q11 = OriImg[int(yu), int(xu), :]
                Q12 = OriImg[int(yu), int(xu) + 1, :]
                Q21 = OriImg[int(yu) + 1, int(xu), :]
                Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]

                """
                Calculate the pixel value of the distorted image by bi-linear interpolation
                """
                disImg[j, i, :] = Q11 * (int(xu) + 1 - xu) * (int(yu) + 1 - yu) + \
                                  Q12 * (xu - int(xu)) * (int(yu) + 1 - yu) + \
                                  Q21 * (int(xu) + 1 - xu) * (yu - int(yu)) + \
                                  Q22 * (xu - int(xu)) * (yu - int(yu))

                """
                Crop the image
                """
                if (xmin <= i <= xmax) and (ymin <= j <= ymax):
                    cropImg[j - ymin, i - xmin, :] = disImg[j, i, :]
                    crop_u[j - ymin, i - xmin] = u[j, i]
                    crop_v[j - ymin, i - xmin] = v[j, i]

    if trainFlag == True:
        saveImgPath = '%s%s%s%s%s%s' % (trainDisPath, '/', types, '_', str(k).zfill(6), '.png')
        saveMatPath = '%s%s%s%s%s%s' % (trainUvPath, '/', types, '_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, disImg)
        # scio.savemat(saveMatPath, {'u': crop_u, 'v': crop_v})
        scio.savemat(saveMatPath, {'u': u, 'v': v})

    else:
        saveImgPath = '%s%s%s%s%s%s' % (testDisPath, '/', types, '_', str(k).zfill(6), '.png')
        saveMatPath = '%s%s%s%s%s%s' % (testUvPath, '/', types, '_', str(k).zfill(6), '.mat')
        io.imsave(saveImgPath, disImg)
        scio.savemat(saveMatPath, {'u': u, 'v': v})

    # def generatedata(types, k, trainFlag):
#     width = 512
#     height = 512
#
#     parameters = distortion_model.distortionParameter(types)
#
#     OriImg = io.imread('%s%s%s%s' % (args.sourcedir, '/', str(k).zfill(6), '.png'))
#
#     disImg = np.array(np.zeros(OriImg.shape), dtype=np.uint8)
#     u = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)
#     v = np.array(np.zeros((OriImg.shape[0], OriImg.shape[1])), dtype=np.float32)
#
#     for i in range(width):
#         for j in range(height):
#             xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)
#             ic(i, j)
#             ic(xu, yu)
#             if (0 <= xu < width - 1) and (0 <= yu < height - 1):
#                 u[j][i] = xu - i
#                 v[j][i] = yu - j
#                 ic(u[j][i], v[j][i])
#                 Q11 = OriImg[int(yu), int(xu), :]
#                 Q12 = OriImg[int(yu), int(xu) + 1, :]
#                 Q21 = OriImg[int(yu) + 1, int(xu), :]
#                 Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]
#
#                 disImg[j, i, :] = Q11 * (int(xu) + 1 - xu) * (int(yu) + 1 - yu) + \
#                                   Q12 * (xu - int(xu)) * (int(yu) + 1 - yu) + \
#                                   Q21 * (int(xu) + 1 - xu) * (yu - int(yu)) + \
#                                   Q22 * (xu - int(xu)) * (yu - int(yu))
#
#     if trainFlag == True:
#         saveImgPath = '%s%s%s%s%s%s' % (trainDisPath, '/', types, '_', str(k).zfill(6), '.png')
#         saveMatPath = '%s%s%s%s%s%s' % (trainUvPath, '/', types, '_', str(k).zfill(6), '.mat')
#         io.imsave(saveImgPath, disImg)
#         scio.savemat(saveMatPath, {'u': u, 'v': v})
#     else:
#         saveImgPath = '%s%s%s%s%s%s' % (testDisPath, '/', types, '_', str(k).zfill(6), '.png')
#         saveMatPath = '%s%s%s%s%s%s' % (testUvPath, '/', types, '_', str(k).zfill(6), '.mat')
#         io.imsave(saveImgPath, disImg)
#         scio.savemat(saveMatPath, {'u': u, 'v': v})


for types in ['barrel']:
    for k in range(args.trainnum):
        generatedata(types, k, trainFlag = True)

    # for k in range(args.trainnum, args.trainnum + args.testnum):
    #     generatedata(types, k, trainFlag = False)
        
# for types in ['pincushion','projective']:
#     for k in range(1,args.trainnum):
#         generatepindata(types, k, trainFlag = True)
#
#     for k in range(args.trainnum, args.trainnum + args.testnum):
#         generatepindata(types, k, trainFlag = False)
