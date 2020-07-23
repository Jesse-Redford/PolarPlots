

import sys
sys.path.append(r'C:\Users\Jesse\Desktop\OpenCV\openh264-1.6.0-win64msvc.dll')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd



import face_recognition

def k_svd(matrix,k=1):
    matrix = np.matrix(matrix)
    U,s,V = svd(matrix,full_matrices = False,compute_uv=True)
    reconst_matrix = np.dot(U[:,:k],np.dot(np.diag(s[:k]),V[:k,:]))
    reconst_matrix = reconst_matrix.clip(0)
    reconst_matrix = reconst_matrix.clip(max=255)
    return(reconst_matrix)
    
def fc_svd(image,m=20,n=20):
    M=image.shape[0]
    N =image.shape[1]
    Ak = np.uint8(np.zeros((M,N)))
    face_locations = face_recognition.face_locations(image)
    """ Store feature bounding indices """
    X1,X2,Y1,Y2 = ([],[],[],[])
    for i in range(len(face_locations)):
        x1, y1, x2, y2 = face_locations[i]
        X1.append(x1) # top - row index 
        X2.append(x2) # bottom - row index
        Y2.append(y2) # left - column index
        Y1.append(y1) # right - column index

    for x in range(0, M, m):
        for y in range(0, N, n):
            block = image[x:x+m, y:y+n]
            fig = False
            for i in range(len(face_locations)):
                if X1[i] <= x <= X2[i] and Y2[i] <= y <= Y1[i]: #if X1[i]-m < x+m/2 < X2[i]+m and Y2[i]-n < y+n/2 < Y1[i]+n: # orginal 
                    fig = True
            if fig == True: 
                k = int(min(m,n) * (((m*n)*np.sqrt(m**2 + n**2)) / ((m+n+1)*(m**2+n**2))))
                if k == 0:
                    k = min(m,n)
            else:
                k=1
            Ak[x:x+m, y:y+n] = k_svd(block,k)

    return Ak
    
    

def get_time_series_matrix(video_file="test_video.mp4",scale_factor=1,processing=None):
    
    Time_Series_Matrix = []
    cap = cv2.VideoCapture(video_file)
    
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if ret == False:
            break
      
    
        # convert color frame to grayscale
        gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        
        if processing == 'fcsvd':
            gs_frame = fc_svd(gs_frame,32,32)
    
        # scale aspect ratio of grayscale frame by factor of 10
        M = int(gs_frame.shape[0]/scale_factor)
        N = int(gs_frame.shape[1]/scale_factor)
        gs_frame = cv2.resize(gs_frame, (N, M))
    
        # Convert image to 1D vector in "Fortran" (column-major) order, ravel is faster since it does not create a copy in memory
        vec_gs_frame =   gs_frame.ravel('F')  # gs_frame.flatten('F') #
    
        # Recover orginal frame from vector representation 
        vec_to_gs_frame = vec_gs_frame.reshape(N, -1).T
    
        # Check that the orginal image was fully recovered, and show on screen
        assert (vec_to_gs_frame == gs_frame).all(), 'orginal image was not recovered'
        #cv2.namedWindow('Recovered Frame', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Recovered Frame',480,640)
        #cv2.imshow('Recovered Frame', vec_to_gs_frame)
    
        # store vector frame in V
        Time_Series_Matrix.append(vec_gs_frame)
        
    Time_Series_Matrix = np.asarray(Time_Series_Matrix).T
    print('Shape of V:',Time_Series_Matrix.shape) 
    print('number of frames stored in V:',Time_Series_Matrix.shape[1])
    return Time_Series_Matrix
    

def plot_time_series_matrix(Time_Series_Matrix):
    # Plot V, notice the wavy lines represent movment (forground), and non changing horzontial lines are background (non-changing)
    plt.imshow(Time_Series_Matrix, cmap='grey')
    plt.colorbar()
    plt.show()
    return
    

    
def run(video_file,scale_factor=1,processing=None):
    Time_Series_Matrix = get_time_series_matrix(video_file=video_file,scale_factor=scale_factor,processing=None)
    Processed_Time_Series_Matrix = get_time_series_matrix(video_file=video_file,scale_factor=scale_factor,processing='fcsvd')
    #plot_time_series_matrix(Time_Series_Matrix)
    #plot_time_series_matrix(Processed_Time_Series_Matrix)

    T = Time_Series_Matrix.shape[1]
    step = 5
    offset = 10
    #fig = plt.figure(figsize=(10, 10))
    #norm_vs_S = fig.add_subplot(221)
    
    plt.title('Processed vs Non Time Series, Largest Singular Value')
   
    plt.scatter(0,0,marker='d',color='blue',label='sec_A-sec_AA')
    plt.scatter(0,0,marker='d',color='red',label='sec_B-sec_BB')
    plt.scatter(0,0,marker='d',color='green',label='sec_A-sec_BB') 
    plt.xlabel('index frame series +- offset')
    plt.ylabel('2-Norm')
    plt.legend()
            
    for t in range(0, T, step):
        sec_A = Time_Series_Matrix[:, t:t+step]
        sec_B = Processed_Time_Series_Matrix[:, t:t+step]
        if t < T-offset:
            sec_AA = Time_Series_Matrix[:, t+offset:t+step+offset]
            sec_BB = Processed_Time_Series_Matrix[:, t+offset:t+step+offset]
            
            """
            plt.scatter(t,np.linalg.norm(sec_A-sec_AA),marker='d',color='blue')
            plt.scatter(t,np.linalg.norm(sec_B-sec_BB),marker='d',color='red')
            
            plt.scatter(t,np.linalg.norm(sec_A-sec_AA,ord='fro'),marker='x',color='blue')
            plt.scatter(t,np.linalg.norm(sec_B-sec_BB,ord='fro'),marker='x',color='red')
            """
            #plt.scatter(t,np.linalg.norm(sec_A-sec_AA,ord='nuc'),marker='o',color='blue')
            #plt.scatter(t,np.linalg.norm(sec_A-sec_BB,ord='nuc'),marker='o',color='red')
            
            plt.scatter(t,np.linalg.norm(sec_A-sec_AA,ord=2),marker='d',color='blue',)
            plt.scatter(t,np.linalg.norm(sec_B-sec_BB,ord=2),marker='d',color='red')
            plt.scatter(t,np.linalg.norm(sec_A-sec_BB,ord=2),marker='d',color='green')
            
            plt.pause(0.001)
        
    plt.show()
    
    
    
    
    return
    
    
run(video_file="comp_outpy.mp4v",scale_factor=1,processing=None)
    
    
