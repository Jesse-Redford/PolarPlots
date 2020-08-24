# Polar Plot Project

### Project Description
This project is in Joint effort with Dr.Mullany from the Mechanical Engineering Deparment and the University of North Carolina Charlotte.
The scope of the project is to develop a method for characterizing material surfaces and provide a stastitical measure for estimating surface defects and quality using digital imagining. The repository includes beta versions of the desktop app which can be run on any windows machine with a camera. In addition we offer a library of modules that can be used to develop, modify, and integrate polar plotting into specific applications. 

### Contributors
- Jesse Redford
- Bridgit Mullany

### Basic Windows Application
If you want to test our a basic implentation download PolarPlot-v0.exe


### Beta Version of Polar Plot App
To compensate for lighting conditions and other external factors which may effect the surface image, version 1 of the application includes filtering options which can be modifed by the user in realtime. The new version also displays a comparision of the raw and filiterd images, in addition to there polar plots for comparison.
Download PolarPlot-v1.exe to use the application. 


### On going Research and Development
Although a model can be trained to detect surface defects, the question remains on wether there is enough information stored in the polar plot for a classifer to have the ability to regenerate what the captured surface looks like using only information from the polarplot.




## Polar Plot Examples
Assuming that a surface containing no defects can be described as a gussian surface, the resulting polar plot of this surface should appear as a relativley uniform circle.
However, in the presence of defects the polar plot should take on some new charactersitic.

### Polar Plot of Gussian Surface
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface.png)

### Polar Plot of Gussian Surface with 3 vertial scratches
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/gussian_surface_with_defects.png)

### Polar Plot of Real Gussian Surface with 3 vertial scratches, with and without filtering
![PolarPlot](https://github.com/Jesse-Redford/PolarPlots/blob/master/real_gussian_surface_with_defects.png)








Algorithms and Analytical tools for surface characterization and defect detection




'''

    import cv2
    import scipy
    import numpy as np
    from scipy.ndimage import rotate
    import matplotlib.pyplot as plt
    from scipy.stats import kurtosis, skew, pearson3
    import time
    import imutils
    import math

    def Polarshort(a,stp):

    limit = int(round(len(a)/np.sqrt(2)/2)) #limit = round(min(M,N)/np.sqrt(2)/2), where M = a.shape[0] N = a.shape[1]
    begining_angle = 0
    end_angle = 180
    
    # generate list of angles based on assumed start angle of 0, end angle of 180, and user step size of stp
    angles = np.arange(begining_angle, end_angle+stp, stp)  #print('num angles',len(angles)) #print('angles',angles)
    
    # create empty matrix to store rq sigma columns
    rq_columns = np.empty((end_angle+1,1000)) 
    rq_columns[:] = np.NaN
   
    # loop through each angle value stored in "angles"
    for i in range(len(angles)): 
    
       # rotates the orginal data image based on current index of the list "angles" = 0,1,2.....180     
       # c =scipy.ndimage.rotate(a, -angles[i]) # .22sec per iteration,- I think this is the preformance bottle neck
        c = imutils.rotate_bound(a, -angles[i]) # 0.004sec per call, this reduces processing time by
        
        #print('shape c',c.shape)  #shape will vary based on rotation #print(cv2.sumElems(c)) # check number of defect pixels in rotated matrix
        
        # makes sure only limit x limit data is evaluated for the polar plot data
        center_point = int(np.floor(len(c)/2))
        c = c[center_point-limit:center_point+limit,center_point-limit:center_point+limit]  # print('cshape',c.shape) # should be 1000x1000

        #calcs stedev of each column limit*2 Rq values are stored
        rq_columns[i,:] = np.std(c,axis=0,ddof=1)
    
    # calcs the stdev for each column
    std_rq_matrix = np.std(rq_columns.T,axis=0,ddof=1) # print('std_rq_matrix',std_rq_matrix.shape) # 1x181
    
    # capitalizes on symmetry for the other half of the polar plot, (mirror and concante values?)
    std_rq_matrix = np.concatenate((std_rq_matrix, std_rq_matrix), axis=0)
 
    T = np.linspace(0,((2*end_angle)-stp)*np.pi/180,362)
    
    return std_rq_matrix, T
    
    
    
    sq = np.zeros((M,N))
    # places three scratches on the surface
    sq[450:450+100,695:695+10]= 1
    sq[650:650+100,695:695+10]= 1
    sq[850:850+100,695:695+10]= 1 
    
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


    plt.scatter(TheataValues,Radii)
    plt.grid(True)
    plt.ylabel('Value')
    plt.xlabel('Angle (Radians)')
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.show
   
'''



