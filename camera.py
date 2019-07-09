import cv2
import cv2 as cv
import numpy as np
import math
capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

n = 0
avgs = [0,0,0]
size = (7,9)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((size[0]*size[1],3), np.float32)
objp[:,:2] =np.array(list(reversed(np.mgrid[0:size[0],0:size[1]].T))).reshape(-1,2)*50

# objp[:,:2] =np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)*21

while True:
    ret, img = capture.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, size, None)

    
    # If found, add object points, image points (after refining them)
    
    if ret == True:
        
        # Check reversed
        p1 = corners[0][0]
        p2 = corners[size[0]*size[1]-1][0]
        isreversed = bool(p1[0] > p2[0] and p1[1] > p2[1])
        
        if isreversed:
            continue

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

        # Draw and display the corners
        cv.drawChessboardCorners(img, size, corners2, ret)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)
        rvec = rvecs[0]
        tvec = tvecs[0]
        

        r = np.zeros(shape=(3,3))
        cv.Rodrigues(rvec, r)

        #r의 전치행렬se
        r_inv = np.transpose(r)
        #print('r : ',r )

        #dx, dy, dz 구하기  
        dx ,dy,dz = np.matmul(r_inv, np.transpose([[0,0,1]])) 

        #pan, tilt 구하기
        pan = math.atan2(dy,dx) - math.pi/2
        tilt = math.atan2(dz, (dx**2 + dy**2) **(1/2))

        # deg로 변환
        deg_pan = np.rad2deg(pan)
        deg_tilt = np.rad2deg(tilt)
        

        #camera_pos 구하기
        cam_pos = - np.matmul(r_inv, tvec)

        #h 구하기
        h = cam_pos[2][0]

        avgs = [(avgs[0]*n +deg_pan)/(n+1), (avgs[1]*n + deg_tilt)/(n+1), (avgs[2]*n + h)/(n+1)]
        n += 1
        cv2.putText(img,'%5.1f %5.1f %5.1f ' % (avgs[0], avgs[1], avgs[2]) ,(20,20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv2.LINE_AA)
        cv.imshow('img', img)
  
            
        
    else:
        cv2.imshow("img", img)
    if cv2.waitKey(1) > 0:
        q= False 
        while True:
            ch= cv2.waitKey(1)
            if  ch== ord('q'):
                np.savez_compressed('param', mtx=mtx, dist= dist,cam = cam_pos,r = r, t = tvec ) 
                np.savez_compressed('D:/개발일지/나르샤4/opencv-master/object_detection_tensorflow/param', mtx=mtx, dist= dist,cam=cam_pos,r = r, t = tvec  ) 
                q=True 
                break
            elif ch >0:
                n = 0
                avgs = [0,0,0]
                break   
        if q:
            break
        

capture.release()
cv2.destroyAllWindows()
