import numpy as np
import cv2


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
    x3w=world_x[0]
    x4w=world_x[1]
    x1w=world_x[2]
    x2w=world_x[3]
    
    y3w=world_y[0]
    y4w=world_y[1]
    y1w=world_y[2]
    y2w=world_y[3]
    
    x1c=camera_x[0]
    x2c=camera_x[1]
    x3c=camera_x[2]
    x4c=camera_x[3]
    
    y1c=camera_y[0]
    y2c=camera_y[1]
    y3c=camera_y[2]
    y4c=camera_y[3]
    
    
    
    A = np.array([[x1w,y1w,1,0,0,0,-x1c*x1w,-x1c*y1w,-x1c], [0,0,0,x1w,y1w,1,-y1c*x1w,-y1c*y1w,-y1c],[x2w,y2w,1,0,0,0,-x2c*x2w,-x2c*y2w,-x2c], [0,0,0,x2w,y2w,1,-y2c*x2w,-y2c*y2w,-y2c],[x3w,y3w,1,0,0,0,-x3c*x3w,-x3c*y3w,-x3c], [0,0,0,x3w,y3w,1,-y3c*x3w,-y3c*y3w,-y3c],[x4w,y4w,1,0,0,0,-x4c*x4w,-x4c*y4w,-x4c], [0,0,0,x4w,y4w,1,-y4c*x4w,-y4c*y4w,-y4c]])


    u,s,v = np.linalg.svd(A, full_matrices=True)
    h = (1/v[8,8])*v[8,:]
    H=np.reshape(h,(-3,3))

    return H

world_x=[0,0,300,300]
world_y=[0,300,300,0]

image_lena=cv2.imread('Lena.png')
resize_lena=cv2.resize(image_lena,(300,300))

cap=cv2.VideoCapture('multipleTags.mp4')


while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,img1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    
    _,contours,_ = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(frame, contours,3, (0,255,0), -1)
    
    idx=1
    cnt=contours[idx]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    corners = cv2.approxPolyDP(cnt,epsilon,True)   

    shape=np.shape(corners)
    if shape[0]>=4:
        camera_x=[corners[0][0][0],corners[1][0][0],corners[2][0][0],corners[3][0][0]]
        camera_y=[corners[0][0][1],corners[1][0][1],corners[2][0][1],corners[3][0][1]]   
    
        H=homography(world_x,world_y,camera_x,camera_y)  
         
        dst = cv2.warpPerspective(resize_lena,H,(frame.shape[1],frame.shape[0])) 
        
        dst1=imageontag(frame,dst)
        
    
        
    idx=4
    cnt=contours[idx]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    corners = cv2.approxPolyDP(cnt,epsilon,True)  
    
    shape=np.shape(corners)
    if shape[0]>=4:

        camera_x=[corners[0][0][0],corners[1][0][0],corners[2][0][0],corners[3][0][0]]
        camera_y=[corners[0][0][1],corners[1][0][1],corners[2][0][1],corners[3][0][1]]   
    
        H=homography(world_x,world_y,camera_x,camera_y)  
         
        dst = cv2.warpPerspective(resize_lena,H,(frame.shape[1],frame.shape[0])) 
        
        dst2=imageontag(dst1,dst)
    
    idx=8
    cnt=contours[idx]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    corners = cv2.approxPolyDP(cnt,epsilon,True) 
    
    shape=np.shape(corners)
    if shape[0]>=4:

        camera_x=[corners[0][0][0],corners[1][0][0],corners[2][0][0],corners[3][0][0]]
        camera_y=[corners[0][0][1],corners[1][0][1],corners[2][0][1],corners[3][0][1]]   
    
        H=homography(world_x,world_y,camera_x,camera_y)  
         
        dst = cv2.warpPerspective(resize_lena,H,(frame.shape[1],frame.shape[0])) 
        
        dst3=imageontag(dst2,dst)
        
        cv2.imshow('f',dst3)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()