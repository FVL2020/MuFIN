import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

from glob import glob
from ntpath import basename
from scipy.misc import imread
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from sklearn.metrics import mean_absolute_error
from skimage.color import rgb2gray
import cv2 as cv

np.set_printoptions(precision=4, suppress=True)

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    # return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)
    # return np.mean(np.abs((np.mean(img_true,2) - np.mean(img_test,2))/255))
    return np.mean(np.abs(img_true - img_test))

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

psnr = []
ssim = []
mae = []
names = []
index = 1

metrics_file = os.path.join(args.output_path, 'log_' + 'metrics' + '.dat')

files = list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))
for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)

    img_gt = (imread(str(fn)) / 255.0).astype(np.float32)
    img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)
    
    img_gt = rgb2gray(img_gt)
    img_pred = rgb2gray(img_pred)
        
    img_gt_uint = cv.imread(str(fn))
    img_pred_uint = cv.imread(path_pred + '/' + basename(str(fn)))
    

    if args.debug != 0:
        plt.subplot('121')
        plt.imshow(img_gt)
        plt.title('Groud truth')
        plt.subplot('122')
        plt.imshow(img_pred)
        plt.title('Output')
        plt.show()

    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51))
    mae.append(compare_mae(img_gt, img_pred))
    # if np.mod(index, 100) == 0:
        # print(
            # str(index) + ' images processed',
            # "PSNR: %.4f" % round(np.mean(psnr), 4),
            # "SSIM: %.4f" % round(np.mean(ssim), 4),
            # "MAE: %.4f" % round(np.mean(mae), 4),
        # )
    print(name, compare_psnr(img_gt, img_pred, data_range=1), compare_ssim(img_gt, img_pred, data_range=1, win_size=51), compare_mae(img_gt, img_pred))    
    index += 1
    logs = [('Name', name), ]
    logs.append(('PSNR', compare_psnr(img_gt_uint, img_pred_uint, data_range=255)))
    logs.append(('SSIM', compare_ssim(img_gt_uint, img_pred_uint, data_range=255, multichannel=True, win_size=51)))
    logs.append(('MAE', compare_mae(img_gt_uint, img_pred_uint)))

    with open(metrics_file, 'a') as f:
        f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
    
np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4)
)
