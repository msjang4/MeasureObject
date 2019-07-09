import numpy as np
import math
loaded =np.load('param.npz')
mtx = loaded['mtx']
dist = loaded['dist']
r = loaded['r']
t = loaded['t']
cam = loaded['cam']

x=[[1],[0],[0]]

r_inv = np.transpose(r)
print(cam)
xw = np.matmul(r_inv, x)
# print(xw)
# print(np.transpose([[0,0,1]]))
dx ,dy,dz = np.matmul(r_inv, np.transpose([[0,0,1]])) 
        #print('dx :',dx, 'dy :',dy,'dz :',dz)

#pan, tilt, h 구하기
pan = math.atan2(dy,dx) - math.pi/2
tilt = math.atan2(dz, (dx**2 + dy**2) **(1/2))
deg_pan = np.rad2deg(pan)
deg_tilt = np.rad2deg(tilt)


xpan = [math.cos(pan),math.sin(pan),0]
roll = math.acos(xw[0] * xpan[0] + xw[1]*xpan[1] + xw[2]*xpan[2])
if xw[2] <0 :
    roll = -roll
print(roll)
#print(mtx, dist,cam,r,t)