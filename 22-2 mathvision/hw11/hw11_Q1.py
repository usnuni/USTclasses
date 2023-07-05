
import cv2
import numpy as np
import matplotlib.pyplot as plt

def thres_finder(img, thres=20,delta_T=1.0):
    # Divide the images in two parts
    x_low, y_low = np.where(img<=thres)
    x_high, y_high = np.where(img>thres)
    # Find the mean of two parts
    mean_low = np.mean(img[x_low,y_low])
    mean_high = np.mean(img[x_high,y_high])
    # Calculate the new threshold
    new_thres = (mean_low + mean_high)/2
    # Stopping criteria, otherwise iterate
    if abs(new_thres-thres)< delta_T:
        return new_thres
    else:
        return thres_finder(img, thres=new_thres,delta_T=1.0)

# Load an image in the greyscale
img = cv2.imread('hw11_sample.png', cv2.IMREAD_GRAYSCALE)
# apply threshold finder
vv1 = thres_finder(img, thres=30,delta_T=1.0)
# threshold the image
ret, thresh = cv2.threshold(img,vv1,255,cv2.THRESH_BINARY)
# Display the image 
out = cv2.hconcat([img,thresh])
cv2.imshow('threshold',out)
cv2.waitKey(0)

# approximate the background
xs = np.arange(0, img.shape[1])
ys = np.arange(0, img.shape[0])
x, y = np.meshgrid(xs, ys)
pos = np.dstack((x, y))
pos = pos.reshape(-1, 2)
A = []
I = []
mean_x = pos[:, 0].mean()
std_x = pos[:, 0].std()
mean_y = pos[:, 1].mean()
std_y = pos[:, 1].std()
norm_x = (pos[:, 0] - mean_x) / std_x
norm_y = (pos[:, 1] - mean_y) / std_y
A = np.vstack((norm_x, norm_y, norm_x * norm_y, norm_x ** 2, norm_y ** 2, np.ones_like(norm_x))).T
I = np.array(img.reshape(-1, 1))
A_plus = np.linalg.inv((A.T @ A)) @ A.T
P = A_plus @ I
background = A @ P
background = background.reshape(img.shape)

# Display the image
cv2.imshow('background', background.astype(np.uint8))
cv2.waitKey(0)

# subtract the background
subtract = img.astype(np.float32) - background.astype(np.float32)
subtract = subtract + subtract.min()
subtract = subtract.astype(np.uint8)

# Display the image
cv2.imshow('subtract', subtract)
cv2.waitKey(0)
