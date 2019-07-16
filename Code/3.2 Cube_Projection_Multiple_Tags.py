import cv2
import numpy as np
import math

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

image_ref = cv2.imread('ref_marker.png')

world_x = [0, 0, 299, 299]
world_y = [0, 299, 299, 0]
world_zprojected = [-299,-299, -299, -299]
Ktranspose = [
            [1406.08415449821, 0, 0],
            [2.20679787308599, 1417.99930662800, 0],
            [1014.13643417416, 566.347754321696, 1]]
        
four_one = [1, 1, 1, 1]
K = np.transpose(Ktranspose)
Kinv = np.linalg.inv(K)
image_lena = cv2.imread('Lena.png')
resize_lena = cv2.resize(image_lena, (300, 300))
cap = cv2.VideoCapture('multipleTags.mp4')

count=0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, img1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    gray2 = np.float32(img1)
    _,contours, _ = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    idx = 1
    cnt = contours[idx]
    epsilon = 0.01*cv2.arcLength(cnt, True)
    corners = cv2.approxPolyDP(cnt, epsilon, True)
#    print(corners)
    shape=np.shape(corners)
    if shape[0]>=4:
        camera_x = [corners[0][0][0], corners[1][0][0],
                    corners[2][0][0], corners[3][0][0]]
        camera_y = [corners[0][0][1], corners[1][0]
                    [1], corners[2][0][1], corners[3][0][1]]
#        cv2.circle(frame, (camera_x[2],camera_y[2]) , 8, (0, 255, 255), -1)
        H = homography(world_x, world_y, camera_x, camera_y)          
  
        Btilda = np.matmul(Kinv, H)
    
        if np.linalg.det(Btilda) > 0:
            B = Btilda
        if np.linalg.det(Btilda) < 0:
            B = -1*Btilda
        b1 = B[:, 0]
        b2 = B[:, 1]
        b3 = B[:, 2]
        b1mag = math.sqrt((b1[0]*b1[0])
                          + (b1[1]*b1[1])+(b1[2]*b1[2]))
    
        b2mag = math.sqrt((b2[0]*b2[0])
                          + (b2[1]*b2[1])+(b2[2]*b2[2]))
        lembda = 2/(b1mag+b2mag)
        r1 = lembda*b1
        r2 = lembda*b2
        r3 = np.cross(r1.transpose(), r2.transpose())
        t = lembda*b3
        
        Ptilda = np.column_stack((r1, r2,r3,t))
        
        P = np.matmul(K, Ptilda)
        world_cordinate_projected = np.row_stack((world_x, world_y,world_zprojected,four_one))
        
        camera_coordinate_projected = np.matmul(P, world_cordinate_projected)  
    
        x1 = int((camera_coordinate_projected[0, 0])/camera_coordinate_projected[2,0])
        y1 = int((camera_coordinate_projected[1, 0])/camera_coordinate_projected[2,0])
#        cv2.circle(frame, (int(x1), int(y1)), 8, (255, 0, 0), -1)
        
        x2 = int((camera_coordinate_projected[0, 1])/camera_coordinate_projected[2,1])
        y2 =int( (camera_coordinate_projected[1, 1])/camera_coordinate_projected[2,1])
#        cv2.circle(frame, (int(x2), int(y2)), 8, (0, 255, 0), -1)
        
        x3 = int(camera_coordinate_projected[0, 2]/camera_coordinate_projected[2,2])
        y3= int(camera_coordinate_projected[1, 2]/camera_coordinate_projected[2,2])
#        cv2.circle(frame, (int(x3), int(y3)), 8, (0, 0, 255), -1)
        
        x4 = int(camera_coordinate_projected[0, 3]/camera_coordinate_projected[2,3])
        y4 = int(camera_coordinate_projected[1, 3]/camera_coordinate_projected[2,3])
#        cv2.circle(frame, (int(x4), int(y4)), 8, (0, 255, 255), -1)
        cv2.line(frame,(camera_x[0],camera_y[0]),(camera_x[1],camera_y[1]),(0,255,0),3)
        cv2.line(frame,(camera_x[1],camera_y[1]),(camera_x[2],camera_y[2]),(0,255,0),3)
        cv2.line(frame,(camera_x[2],camera_y[2]),(camera_x[3],camera_y[3]),(0,255,0),3)
        cv2.line(frame,(camera_x[0],camera_y[0]),(camera_x[3],camera_y[3]),(0,255,0),3)
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.line(frame,(x2,y2),(x3,y3),(0,255,0),3)
        cv2.line(frame,(x3,y3),(x4,y4),(0,255,0),3)
        cv2.line(frame,(x4,y4),(x1,y1),(0,255,0),3)
        cv2.line(frame,(camera_x[0],camera_y[0]),(x3,y3),(0,255,0),3)
        cv2.line(frame,(camera_x[1],camera_y[1]),(x4,y4),(0,255,0),3)
        cv2.line(frame,(camera_x[2],camera_y[2]),(x1,y1),(0,255,0),3)
        cv2.line(frame,(camera_x[3],camera_y[3]),(x2,y2),(0,255,0),3)
        
    idx = 4
    cnt = contours[idx]
    epsilon = 0.01*cv2.arcLength(cnt, True)
    corners = cv2.approxPolyDP(cnt, epsilon, True)
#    print(corners)
    shape=np.shape(corners)
    if shape[0]>=4:
        camera_x = [corners[0][0][0], corners[1][0][0],
                    corners[2][0][0], corners[3][0][0]]
        camera_y = [corners[0][0][1], corners[1][0]
                    [1], corners[2][0][1], corners[3][0][1]]
#        cv2.circle(frame, (camera_x[2],camera_y[2]) , 8, (0, 255, 255), -1)
        H = homography(world_x, world_y, camera_x, camera_y)          
  
        Btilda = np.matmul(Kinv, H)
    
        if np.linalg.det(Btilda) > 0:
            B = Btilda
        if np.linalg.det(Btilda) < 0:
            B = -1*Btilda
        b1 = B[:, 0]
        b2 = B[:, 1]
        b3 = B[:, 2]
        b1mag = math.sqrt((b1[0]*b1[0])
                          + (b1[1]*b1[1])+(b1[2]*b1[2]))
    
        b2mag = math.sqrt((b2[0]*b2[0])
                          + (b2[1]*b2[1])+(b2[2]*b2[2]))
        lembda = 2/(b1mag+b2mag)
        r1 = lembda*b1
        r2 = lembda*b2
        r3 = np.cross(r1.transpose(), r2.transpose())
        t = lembda*b3
        
        Ptilda = np.column_stack((r1, r2,r3,t))
        
        P = np.matmul(K, Ptilda)
        world_cordinate_projected = np.row_stack((world_x, world_y,world_zprojected,four_one))
        
        camera_coordinate_projected = np.matmul(P, world_cordinate_projected)  
    
        x1 = int((camera_coordinate_projected[0, 0])/camera_coordinate_projected[2,0])
        y1 = int((camera_coordinate_projected[1, 0])/camera_coordinate_projected[2,0])
#        cv2.circle(frame, (int(x1), int(y1)), 8, (255, 0, 0), -1)
        
        x2 = int((camera_coordinate_projected[0, 1])/camera_coordinate_projected[2,1])
        y2 =int( (camera_coordinate_projected[1, 1])/camera_coordinate_projected[2,1])
#        cv2.circle(frame, (int(x2), int(y2)), 8, (0, 255, 0), -1)
        
        x3 = int(camera_coordinate_projected[0, 2]/camera_coordinate_projected[2,2])
        y3= int(camera_coordinate_projected[1, 2]/camera_coordinate_projected[2,2])
#        cv2.circle(frame, (int(x3), int(y3)), 8, (0, 0, 255), -1)
        
        x4 = int(camera_coordinate_projected[0, 3]/camera_coordinate_projected[2,3])
        y4 = int(camera_coordinate_projected[1, 3]/camera_coordinate_projected[2,3])
#        cv2.circle(frame, (int(x4), int(y4)), 8, (0, 255, 255), -1)
        cv2.line(frame,(camera_x[0],camera_y[0]),(camera_x[1],camera_y[1]),(0,255,0),3)
        cv2.line(frame,(camera_x[1],camera_y[1]),(camera_x[2],camera_y[2]),(0,255,0),3)
        cv2.line(frame,(camera_x[2],camera_y[2]),(camera_x[3],camera_y[3]),(0,255,0),3)
        cv2.line(frame,(camera_x[0],camera_y[0]),(camera_x[3],camera_y[3]),(0,255,0),3)
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.line(frame,(x2,y2),(x3,y3),(0,255,0),3)
        cv2.line(frame,(x3,y3),(x4,y4),(0,255,0),3)
        cv2.line(frame,(x4,y4),(x1,y1),(0,255,0),3)
        cv2.line(frame,(camera_x[0],camera_y[0]),(x3,y3),(0,255,0),3)
        cv2.line(frame,(camera_x[1],camera_y[1]),(x4,y4),(0,255,0),3)
        cv2.line(frame,(camera_x[2],camera_y[2]),(x1,y1),(0,255,0),3)
        cv2.line(frame,(camera_x[3],camera_y[3]),(x2,y2),(0,255,0),3)
        
    idx = 8
    cnt = contours[idx]
    epsilon = 0.01*cv2.arcLength(cnt, True)
    corners = cv2.approxPolyDP(cnt, epsilon, True)
#    print(corners)
    shape=np.shape(corners)
    if shape[0]>=4:
        camera_x = [corners[0][0][0], corners[1][0][0],
                    corners[2][0][0], corners[3][0][0]]
        camera_y = [corners[0][0][1], corners[1][0]
                    [1], corners[2][0][1], corners[3][0][1]]
#        cv2.circle(frame, (camera_x[2],camera_y[2]) , 8, (0, 255, 255), -1)
        H = homography(world_x, world_y, camera_x, camera_y)          
  
        Btilda = np.matmul(Kinv, H)
    
        if np.linalg.det(Btilda) > 0:
            B = Btilda
        if np.linalg.det(Btilda) < 0:
            B = -1*Btilda
        b1 = B[:, 0]
        b2 = B[:, 1]
        b3 = B[:, 2]
        b1mag = math.sqrt((b1[0]*b1[0])
                          + (b1[1]*b1[1])+(b1[2]*b1[2]))
    
        b2mag = math.sqrt((b2[0]*b2[0])
                          + (b2[1]*b2[1])+(b2[2]*b2[2]))
        lembda = 2/(b1mag+b2mag)
        r1 = lembda*b1
        r2 = lembda*b2
        r3 = np.cross(r1.transpose(), r2.transpose())
        t = lembda*b3
        
        Ptilda = np.column_stack((r1, r2,r3,t))
        
        P = np.matmul(K, Ptilda)
        world_cordinate_projected = np.row_stack((world_x, world_y,world_zprojected,four_one))
        
        camera_coordinate_projected = np.matmul(P, world_cordinate_projected)  
    
        x1 = int((camera_coordinate_projected[0, 0])/camera_coordinate_projected[2,0])
        y1 = int((camera_coordinate_projected[1, 0])/camera_coordinate_projected[2,0])
#        cv2.circle(frame, (int(x1), int(y1)), 8, (255, 0, 0), -1)
        
        x2 = int((camera_coordinate_projected[0, 1])/camera_coordinate_projected[2,1])
        y2 =int( (camera_coordinate_projected[1, 1])/camera_coordinate_projected[2,1])
#        cv2.circle(frame, (int(x2), int(y2)), 8, (0, 255, 0), -1)
        
        x3 = int(camera_coordinate_projected[0, 2]/camera_coordinate_projected[2,2])
        y3= int(camera_coordinate_projected[1, 2]/camera_coordinate_projected[2,2])
#        cv2.circle(frame, (int(x3), int(y3)), 8, (0, 0, 255), -1)
        
        x4 = int(camera_coordinate_projected[0, 3]/camera_coordinate_projected[2,3])
        y4 = int(camera_coordinate_projected[1, 3]/camera_coordinate_projected[2,3])
#        cv2.circle(frame, (int(x4), int(y4)), 8, (0, 255, 255), -1)
        cv2.line(frame,(camera_x[0],camera_y[0]),(camera_x[1],camera_y[1]),(0,255,0),3)
        cv2.line(frame,(camera_x[1],camera_y[1]),(camera_x[2],camera_y[2]),(0,255,0),3)
        cv2.line(frame,(camera_x[2],camera_y[2]),(camera_x[3],camera_y[3]),(0,255,0),3)
        cv2.line(frame,(camera_x[0],camera_y[0]),(camera_x[3],camera_y[3]),(0,255,0),3)
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.line(frame,(x2,y2),(x3,y3),(0,255,0),3)
        cv2.line(frame,(x3,y3),(x4,y4),(0,255,0),3)
        cv2.line(frame,(x4,y4),(x1,y1),(0,255,0),3)
        cv2.line(frame,(camera_x[0],camera_y[0]),(x3,y3),(0,255,0),3)
        cv2.line(frame,(camera_x[1],camera_y[1]),(x4,y4),(0,255,0),3)
        cv2.line(frame,(camera_x[2],camera_y[2]),(x1,y1),(0,255,0),3)
        cv2.line(frame,(camera_x[3],camera_y[3]),(x2,y2),(0,255,0),3)
        
        cv2.imshow('img1', frame)
#        cv2.imwrite("frame%d.jpg", frame)
        count=count+1
        
    
    #q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
