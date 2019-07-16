import cv2
import numpy as np

def id_detect(img):    
    max_size=480      
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    _,img1 = cv2.threshold(imgray,200,255,cv2.THRESH_BINARY)
    cv2.imshow('p',img1)    
    im = cv2.resize(img1, (max_size,max_size))
    width=int(max_size/8)
    im2=im[3*width:width*5,3*width:width*5]    
    identity=[]
    v=[int(im2[30,30]),int(im2[30,90]),int(im2[90,90]),int(im2[90,30])]
    for i in v:
        if i>200:
            identity.append(1)
        else:
            identity.append(0)
    return identity
    
    

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


cap=cv2.VideoCapture('Tag1.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,img1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    gray2 = np.float32(img1)    
    _,contours,_ = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    idx=1
    cnt=contours[idx]
    epsilon = 0.1*cv2.arcLength(cnt,True)
    corners = cv2.approxPolyDP(cnt,epsilon,True)  
    x, y, w, h = cv2.boundingRect(corners)
    cropped_img = frame[y+3:y+h-3,x+3:x+w-3]
    shape=np.shape(corners)
    if shape[0]>=4:
        camera_x=[corners[0][0][0],corners[1][0][0],corners[2][0][0],corners[3][0][0]]
        camera_y=[corners[0][0][1],corners[1][0][1],corners[2][0][1],corners[3][0][1]]     
        img5=cv2.drawContours(frame,cnt,-1,-1)
        H=homography(camera_x,camera_y,world_x,world_y)      
        dst = cv2.warpPerspective(img5,H,(300,300))
        identity=id_detect(dst)
        id1="Tag id is" + str(int(''.join(str(e) for e in identity),2))
        print(id1)
        cv2.putText(frame,id1,(corners[0][0][0],corners[0][0][1]), font, 0.6,(0,0,255),2)        
        cv2.imshow('img1',frame)    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

