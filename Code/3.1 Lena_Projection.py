import cv2
import numpy as np

def imageontag(frame,lena):
    
    rows,cols,channels = lena.shape
    roi = frame[0:rows, 0:cols ]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(lena,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(lena,lena,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)    
    return dst
def homography(world_x,world_y,camera_x,camera_y):
    x3w,x4w,x1w,x2w=world_x       
    y3w,y4w,y1w,y2w=world_y    
    x1c,x2c,x3c,x4c,=camera_x    
    y1c,y2c,y3c,y4c=camera_y  
    A = np.array([[x1w,y1w,1,0,0,0,-x1c*x1w,-x1c*y1w,-x1c], [0,0,0,x1w,y1w,1,-y1c*x1w,-y1c*y1w,-y1c],[x2w,y2w,1,0,0,0,-x2c*x2w,-x2c*y2w,-x2c], [0,0,0,x2w,y2w,1,-y2c*x2w,-y2c*y2w,-y2c],[x3w,y3w,1,0,0,0,-x3c*x3w,-x3c*y3w,-x3c], [0,0,0,x3w,y3w,1,-y3c*x3w,-y3c*y3w,-y3c],[x4w,y4w,1,0,0,0,-x4c*x4w,-x4c*y4w,-x4c], [0,0,0,x4w,y4w,1,-y4c*x4w,-y4c*y4w,-y4c]])
    u,s,v = np.linalg.svd(A, full_matrices=True)
    h = (1/v[8,8])*v[8,:]
    H=np.reshape(h,(-3,3))
    return H

image_ref=cv2.imread('ref_marker.png')
world_x=[0,0,300,300]
world_y=[0,300,300,0]
image_lena=cv2.imread('Lena.png')
resize_lena=cv2.resize(image_lena,(300,300))
cap=cv2.VideoCapture('Tag2.mp4')
while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,img1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
#    gray1 = cv2.GaussianBlur(img1,(9,9),0)
    gray2 = np.float32(img1)    
    _,contours,_ = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    idx=1
    cnt=contours[idx]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    corners = cv2.approxPolyDP(cnt,epsilon,True)   
    shape=np.shape(corners)
    if shape[0]>=4:        
        camera_x=[corners[0][0][0],corners[1][0][0],corners[2][0][0],corners[3][0][0]]
        camera_y=[corners[0][0][1],corners[1][0][1],corners[2][0][1],corners[3][0][1]]       
        H=homography(world_x,world_y,camera_x,camera_y)#          
        dst = cv2.warpPerspective(resize_lena,H,(frame.shape[1],frame.shape[0]))        
        dst1=imageontag(frame,dst)
    
    cv2.imshow('img1',dst1)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
