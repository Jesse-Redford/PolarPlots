# Materials_Imaging
Algorithms and Analytical tools for surface characterization and defect detection




'''
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
   
'''


We can call the function above for testing

'''
    sq = np.zeros((M,N))
    # places three scratches on the surface
sq[450:450+100,695:695+10]= 1
sq[650:650+100,695:695+10]= 1
sq[850:850+100,695:695+10]= 1 


    plt.scatter(T,std_rq_matrix)
    plt.grid(True)
    plt.ylabel('Value')
    plt.xlabel('Angle (Radians)')
    plt.grid(which='minor', alpha=0.2)
    plt.grid(which='major', alpha=0.5)
    plt.show
    
'''
