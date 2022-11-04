import cv2
import numpy as np


fname = 'pjh/1029/topview_undi_done/' + 'right' + '_top_undi.png'
img = cv2.imread(fname)


ipm_matrix = np.array([[-1.13784642e-01,  7.08183787e-01,  4.83263055e+02],
 [ 5.21559908e-01,  1.16018216e+00,  6.82511975e+01],
 [-3.41797021e-04,  4.50218346e-03,  1.00000000e+00]])


size = (480, 640)
ipm = cv2.warpPerspective(img, ipm_matrix, size)

cv2.imshow('ori', img) 
cv2.imshow('top', ipm) 
cv2.waitKey(0)

