# surfaceGenPolar.py

import cv2
import scipy
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, pearson3
import time
import imutils
import math

#def rotateImage(image, angle):
    #row,col = image.shape
    #center= tuple(np.array([row,col])/2)
    #rot_mat = cv2.getRotationMatrix2D(center,angle,1.) # 1. defualt, adjust to modift spikes
    #new_image = cv2.warpAffine(image, rot_mat, (col,row))
    
    #image_center = tuple(np.array(image.shape[1::-1]) / 2)
    #rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.11)
    #result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
 #   new_image = imutils.rotate(image, angle)
    
  #  return new_image
"""
def rotateImage(mat, angle):
  # angle in degrees

  height, width = mat.shape[:2]
  image_center = (width/2, height/2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat
"""
def rotateImage(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def Polarshort(a,stp):
    
    """
    INPUTS: a: surface/image matrix
    stp: Resolution of rotation
    gets info regarding the max size of the data that can be used to ensure
    only the original data is evaluated and not data filled in by the imrotate command
    """
    
    limit = round(len(a)/np.sqrt(2)/2) #limit = round(min(M,N)/np.sqrt(2)/2), where M = a.shape[0] N = a.shape[1]
    begining_angle =0
    end_angle = 180
    
    # generate list of angles based on assumed start angle of 0, end angle of 180, and user step size of stp
    angles = np.arange(begining_angle, end_angle+stp, stp)
    #print('num angles',len(angles))
    #print('angles',angles)
    
    # create empty matrix to store rq sigma columns
    rq_columns = np.empty((end_angle+1,1000)) #np.zeros((end_angle+1,1000)) 
    rq_columns[:] = np.NaN
   # print(rq_columns.shape)
  
    # loop through each angle value stored in "angles"
    for i in range(len(angles)): 
          
       # rotates the orginal data image based on current index of the list "angles" = 0,1,2.....180     
      #  c =scipy.ndimage.rotate(a, -angles[i]) # .22sec per iteration,- I think this is the preformance bottle neck
        c = rotateImage(a, -angles[i]) # 0.004sec per call, this reduces processing time by
        #c = imutils.rotate(a, -angles[i])
        #c = imutils.rotate_bound(a, -angles[i])
        
       # if i == 45:
        #    cv2.imshow(str(c.shape), c)
     #       print(angles[i])
    
        
        print('shape c',c.shape)  #shape will vary based on rotation 
        
        # makes sure only limit x limit data is evaluated for the polar plot data
        center_point = np.floor(len(c)/2)
        limit = int(limit)
        center_point = int(center_point)
        c = c[center_point-limit:center_point+limit,center_point-limit:center_point+limit]   #c = c[center_point-limit:center_point+limit-1,center_point-limit:center_point+limit-1]
        #print('cshape',c.shape) # should be 1000x1000
       # print(cv2.sumElems(c))
       # if i == 45:
        #    cv2.imshow(str(c.shape), c)
        
        #calcs stedev of each column limit*2 Rq values are stored
        
        rq_columns[i,:] = np.std(c,axis=0,ddof=1)
        #print(rq_columns) #181x1000
    
    # calcs the stdev for each column
    std_rq_matrix = np.std(rq_columns.T,axis=0,ddof=1)
    #print('std_rq_matrix',std_rq_matrix.shape) # 1x181
    
    # capitalizes on symmetry for the other half of the polar plot, (mirror and concante values?)
    std_rq_matrix = np.concatenate((std_rq_matrix, std_rq_matrix), axis=0) #std_rq_matrix = [std_rq_matrix, std_rq_matrix[2:end-1]];
    print('std_rq_matrix',std_rq_matrix.shape) # 1x360
    
    T = np.linspace(0,((2*end_angle)-stp)*np.pi/180,362) 
    #print('T',T)
    print('Tshape',T.shape)  # should be 1x360 
    plt.scatter(T,std_rq_matrix)
    plt.grid(True)
    plt.ylabel('Value')
    plt.xlabel('Angle (Radians)')
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.show
    
    
    return std_rq_matrix, T
    


mu = 0
sigma = 3
#skew = 0
kurt = 3
M = 1414
N = 1414


# generates a matrix with a gaussian distribution of values
sq = pearson3.rvs(skew=0, loc=mu, scale=sigma, size=(M, N)) # modify randomization with kurt?? np.zeros((M,N))  #
#sq = np.zeros((M,N))
# places three scratches on the surface
sq[450:450+100,695:695+10]= -15
sq[650:650+100,695:695+10]= -15
sq[850:850+100,695:695+10]= -15

# generate matrix by importing an image and converting to grayscale representation with openCV
#sq = cv2.imread(r"C:\Users\Jesse\Desktop\OpenCV\brass_cparri23_L02_fall2019_100x_25s.jpg",0)
#sq = cv2.resize(sq, (N, M))

#sq[450:450+100,695:695+10]= -15 #-15
#sq[250:650+100,600:695+10]= -15 #-15
#sq[850:850+100,695:695+10]= -15 #-15


# sends the surface data (sq) and the angle rotation increment (1 degree) to the
# subroutine - gets back the radii at the different angular rotations (TheataValues)
Radii,TheataValues = Polarshort(sq, 1);

#print('Radii',Radii)
#print('TheataValues',TheataValues)

fig = plt.figure(figsize=(10, 10))
surface = fig.add_subplot(121)
surface.set_title('Surfacemap')
surface.imshow(sq)

polar_plot = fig.add_subplot(122,projection='polar')
polar_plot.set_title('Stdev of Rq values versus angle')
polar_plot.plot(TheataValues, Radii)
plt.show()
 


