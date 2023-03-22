import numpy as np
import scipy. ndimage as nd
import time
from matplotlib import pyplot as plt

def distance_transform_np(input,vox_size):
    """
    Euclidean Distance Transform

    The Euclidean Distance between each '1' to the nearest '0' is measured.
    This function returns a transformed array.
    
    Parameters
    --------------
    input: array_like 
        A 2-D/3-D binary array to be transformed.

    vox_size: tuple (3)
        Voxel dimensions in the format of (Z,X,Y).
        Z being slice distnance; X and Y being axial spacing (x and y must be of the same value).

    Returns
    --------------
    sliced_im: array_like 
        A transformed array containing the Euclidean distance between each 1 
        to the nearest 0. 
        Array has the same shape as input array.
    """
    
    #load input .npy file.
    input = np.load(input)
    converted_input = input.astype(np.uint8)

    #compute the (z,x,y) coor-dinates of all locations of '1's.
    ones_loc = np.where(input ==1)

    #create empty array to append results.
    result = np.zeros(input.shape)

    #extract voxel dimensions from second input argument.
    vox_z,vox_x,vox_y = vox_size

    #compute the ratio of slice distance:axial spacing, vice versa.
    z_xy_ratio = vox_z/vox_x
    xy_z_ratio = vox_x/vox_z

    #COMPARE the computed ratio
    #n_ represents the increase in surrounding co-ordinate intervals during the Euclidean Diistance calculations.
    #m_ represents the increased interval +1 to achieve the same expansion of co-ordinates 
    #in the positive and negative directions.

    #both ratios are EQUAL
    if z_xy_ratio == xy_z_ratio:
        n_xy = 1
        m_xy = n_xy+1
        n_z = 1
        m_z = n_z+1

    #slice distance is GREATER than axial spacing
    #ratio of slice distance:axial spacing is taken as the expansion
    #interval.
    elif z_xy_ratio > xy_z_ratio:
        n_xy = int(np.ceil(z_xy_ratio))
        m_xy = n_xy+1
        n_z = 1
        m_z = n_z+1

    #slice distance is LESSER than axial spacing
    #ratio of axial spacing:slice distance is taken as the expansion
    #interval. 
    elif xy_z_ratio > z_xy_ratio:
        n_z = int(np.ceil(xy_z_ratio))
        m_z = n_z+1
        n_xy = 1
        m_xy = n_xy+1


    #for loop goes through all the co-ordinates of '1's 
    for z,x,y in zip(ones_loc[0], ones_loc[1],ones_loc[2]):
       
       #index for the centres of the expanded array in N_cords
        z_idx = n_z
        xy_idx = n_xy
        while True:
            N_cords=[]

            #Extract voxels from the surrounding of the selected '1's
            #expansion intervals are determined by n_ and m_ values computed above.
            N_cords = converted_input[z-n_z:z+m_z, x-n_xy:x+m_xy, y-n_xy:y+m_xy]
            #compute locations of '0's present in the expanded array.
            zero_loc = np.where(N_cords == 0)
            zero_loc = np.asarray(zero_loc)
            z0,x0,y0 = zero_loc[0],zero_loc[1],zero_loc[2]

            #In the case where there are '0'(s) present in the expanded array
            #Compute the Euclidean distance between the selected '1' and all the '0's
            #using the distance formula.
            #select the minimum distance from all distance computed
            #replace '1' at the selected co-ordinate with the min.distance value.
            if not zero_loc.size == 0 :
                Dist = ((xy_idx*vox_x)-(x0*vox_x))**2+((xy_idx*vox_y)-(y0*vox_y))**2+((z_idx*vox_z)-(z0*vox_z))**2
                min_dist= np.sqrt(Dist.min())
                result[z,x,y]=min_dist
                break

            #If no '0's present in the expanded array, continue to expand with intervals
            #n_ selected above.
            else:
                n_z += n_z
                m_z += n_z
                n_xy += n_xy
                m_xy += n_xy
                z_idx += n_z
                xy_idx += n_xy

    #This calculation optimises the expansion array by accounting the ratio of voxel dimensions.
    return (result)

def save_image(slice_num,name):
    """ 
    Saves 1 gray scale image with (1,2) subplots as PNG file.
    
    Parameters
    --------------
    slice_num: int.
        Number of slice in the z-axis.

    np_name: str.
        Name for slice computed by distance_transform_np.

    edt_name: str.
        Name for slice computed by distance_transform_edt.

    Returns
    --------------
    1 PNG file saved in the same path.

    """
    
    im = result[slice_num,:,:]
    im_edt = edt_result[slice_num,:,:]

    fig5, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.imshow(im,cmap='gray',extent=[0, 64, 0, 64])
    ax1.set_title('distance_transform_np\n slice_{}'.format(slice_num),fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.set_xlabel("x-axis (mm)", fontsize=20, labelpad=10)
    ax1.set_ylabel("y-axis (mm)", fontsize=20, labelpad=10)
    
    ax2.imshow(im_edt,cmap='gray',extent=[0, 64, 0, 64])
    ax2.set_title('distance_transform_edt\n slice_{}'.format(slice_num),fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax2.set_xlabel("x-axis (mm)", fontsize=20, labelpad=10)
    ax2.set_ylabel("y-axis (mm)", fontsize=20, labelpad=10)
    plt.savefig(name, facecolor='white')

    return

#input and voxel dimensions
input = "label_train00.npy"
vox_size=(2,0.5,0.5)

#Calculating computational time for distance_transform_np
start = time.time()
result = distance_transform_np(input,vox_size)
end = time.time()
result_time=np.around((end - start),3)
result= result.astype(np.float64)

#Calculating computational time for distance_transform_edt
start = time.time()
edt_result = nd.distance_transform_edt(np.load(input), vox_size, return_distances=True, return_indices=False, distances=None, indices=None)
end = time.time()
edt_result_time=np.around((end - start),3)
edt_result= edt_result.astype(np.float64)

#print time taken for both functions
#the time taken for the built-in edt function is about 35 to 40 times faster than the np function
#however, it is still working in the seconds for this size of input.
#The time taken for np function will increase with input size.
print("time taken for distance_transform_np :", result_time,"s")
print("time taken for distance_transform_edt :", edt_result_time,"s" )

#saving image of selected slices
save_image(10,"slice_10.png")
save_image(13,"slice_13.png")
save_image(16,"slice_16.png")
save_image(19,"slice_19.png")
save_image(22,"slice_22.png")

#Similarity comparison between results computed by np and edt function
Diff=[]
#Take the absolute difference between two 3-D arrays.
Diff=abs(np.subtract(edt_result,result))
#Compute the mean difference and standard deviation (sd) between two 3-D arrays
mean_diff=np.around((np.mean(Diff)),6)
stv_diff=np.around((np.std(Diff)),6)

#the computer mean between two arrays are in the magnitude of 10^-5.
#the computed sd between the two arrays is 0.0092.
#Thus, there is no significant difference between the computed results.
print("mean of absolute differences =", mean_diff)
print("standard deviation of differences =", stv_diff)


