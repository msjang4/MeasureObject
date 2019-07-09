import numpy as np
import glob
import cv2 as cv
import cv2
loaded =np.load('param.npz')
mtx = loaded['mtx']
dist = loaded['dist']
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

images = glob.glob('*.jpg')
while True:
    # img = cv.imread(fname)
    
    ret, img = capture.read()
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    # cv.imwrite('undistored_{}.png'.format(fname), dst)
    cv.imshow('undistorted', dst)
    if cv.waitKey(1) > 0:
        break