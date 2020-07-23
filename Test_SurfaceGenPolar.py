# Test_surfaceGenPolar.py

import cv2
import scipy
import numpy as np
from numpy import std
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, pearson3

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def generate_matrix(): # takes about 0.077sec 
    mu = 0
    sigma = 3
    #skew = 0
    #kurt = 3
    M = 1414
    N = 1414
    # generates a matrix with a gaussian distribution of values
    a = pearson3.rvs(skew=0, loc=mu, scale=sigma, size=(M, N)) # modify randomization with kurt??
    # places three scratches on the surface
    a[450:450+100,695:695+10]= -15
    a[650:650+100,695:695+10]= -15
    a[850:850+100,695:695+10]= -15
    return a

    
    

def Test_Polarshort():
    
    a = generate_matrix()
    stp = 1
    
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
    # print('num angles',len(angles))
    
    # create empty matrix to store rq sigma columns
    rq_columns = np.zeros((end_angle,1000)) 
  
    # loop through each angle value stored in "angles"
    for i in range(len(angles)-1):   
        
       # rotates the orginal data image based on current index of the list "angles" = 0,1,2.....180     
        #c =scipy.ndimage.rotate(a, -angles[i])
        c = rotateImage(a, -angles[i])
        #print(c)
        #print('shape c',c.shape) #shape will vary based on rotation 
        
        # makes sure only limit x limit data is evaluated for the polar plot data
        center_point = np.floor(len(c)/2)
        limit = int(limit)
        center_point = int(center_point)
        c = c[center_point-limit:center_point+limit,center_point-limit:center_point+limit]   #c = c[center_point-limit:center_point+limit-1,center_point-limit:center_point+limit-1]
        #print('cshape',c.shape) # should be 1000x1000
        
        #calcs stedev of each column limit*2 Rq values are stored
        rq_columns[i,:] = std(c,axis=0)
        #print(np.std(c,axis=0).shape) 181x1000
    
    # calcs the stdev for each column
    std_rq_matrix = std(rq_columns.T,axis=0)
    # print('std_rq_matrix',std_rq_matrix.shape) # 1x181
    
    # capitalizes on symmetry for the other half of the polar plot, (mirror and concante values?)
    std_rq_matrix = np.concatenate((std_rq_matrix, std_rq_matrix), axis=0) #std_rq_matrix = [std_rq_matrix, std_rq_matrix[2:end-1]];
    #print('std_rq_matrix',std_rq_matrix.shape) # 1x360
    
    T = np.linspace(0,((2*end_angle)-stp)*np.pi/180,360) 
    #print('T',T)
    #print(T.shape)  # should be 1x360 
    
    return std_rq_matrix, T
    
import cProfile
import pstats
profile = cProfile.Profile()
profile.runcall(Test_Polarshort)
ps = pstats.Stats(profile)
ps.sort_stats('cumulative').print_stats(10)

ps.print_stats()




