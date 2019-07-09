import numpy as np
import cv2 as cv
import glob
import math

size = (7,9)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((size[0]*size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)*20

#objp[:,:2] = np.mgrid[0:7,0:9].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
isreversed = []

images = glob.glob('*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, size, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        p1 = corners[0][0]
        p2 = corners[size[0]*size[1]-1][0]
        isreversed.append(p1[0] < p2[0] and p1[1] < p2[1])
        # Draw and display the corners
        cv.drawChessboardCorners(img, size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
cv.destroyAllWindows()

print(isreversed)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# for i in range(0, len(images)):
#     _, rvec , tvec =cv.solvePnP(objpoints[i], imgpoints[i], mtx, dist)
#     print(rvec, rvecs[i])
#     print(tvec, tvecs[i])

for i in range(0,len(images)):

    k = -1 if isreversed[i] else 1

    # rvec으로부터 r mat 얻기
    r = np.zeros(shape=(3,3))
    cv.Rodrigues(rvecs[i], r)

    #r의 전치행렬se
    r_inv = np.transpose(r)
    #print('r : ',r )


    #dx, dy, dz 구하기  
    dx ,dy,dz = k*np.matmul(r_inv, np.transpose([[0,0,1]])) 
    #print('dx :',dx, 'dy :',dy,'dz :',dz)

    #pan, tilt, h 구하기
    pan = math.atan2(dy,dx) - math.pi/2
    tilt = math.atan2(dz, (dx**2 + dy**2) **(1/2))
    print('rad - pan :', pan, 'rad - tilt :', tilt)
    print('deg - pan :', np.rad2deg(pan), 'deg - tilt :', np.rad2deg(tilt))
    

    #camera_pos
    cam_pos =  -k*np.matmul(r_inv, tvecs[i])
    print('cam_pos :', cam_pos)
