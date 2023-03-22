import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from Transforms import Image3D, AffineTransform 

def plot(AFF_name, Affined_image,vox_size):
    """
    Returns a saved 3D plot in '.png' files.

    Parameters
    --------------
    AFF_name : str.
        Affine transform name.
    
    Affined_image : ndarray (3D)
        Warped/transformed image.

    vox_size : tuple (3)
        Voxel dimensions given by (Z,X,Y), where z is the slice distance; x and y are axial spacing.

    """
    #create empty array filled with '0's with the size of the transformed image
    image_plot = np.zeros(Affined_image.shape)

    #locate all non-zero voxels in the transformed array
    val = np.array(np.where(Affined_image>0)).T
    #compute centre z coordinate to locate midpoint of slice range
    m = val[0,0] +((val[-1,0] - val[0,0])//2)
    #taking 5 image slices
    slice_range = np.arange(m-10,m+15,5)
    slice_range = slice_range.astype(int)

    for i in slice_range:
        image_plot[i,:,:] = Affined_image[i,:,:]

    facecolors = cm.gray(image_plot)
    facecolors[:,:,:,-1] = image_plot
    filled = facecolors[:,:,:,-1] != 0

    ZZZ_1 = np.arange(0, (image_plot.shape[0]+1)*vox_size[0],vox_size[0])
    XXX_1 = np.arange(0, (image_plot.shape[1]+1)*vox_size[1],vox_size[1])
    YYY_1 = np.arange(0, (image_plot.shape[2]+1)*vox_size[2],vox_size[2])
    ZZZ_1,XXX_1,YYY_1=np.meshgrid(ZZZ_1,XXX_1,YYY_1,indexing='ij')

    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(projection='3d')

    ax.tick_params(axis='z', labelsize=60)
    ax.tick_params(axis='x', labelsize=60)
    ax.tick_params(axis='y', labelsize=60)

    ax.set_xlabel("z-axis (mm)", fontsize=100, labelpad=80)
    ax.set_ylabel("x-axis (mm)", fontsize=100, labelpad=80)
    ax.set_zlabel("y-axis (mm)", fontsize=100, labelpad=80)

    ax.set_title('{}'.format(AFF_name) , fontsize=120)

    ax.voxels(ZZZ_1, XXX_1, YYY_1, filled,facecolors=facecolors, shade=False)
    plt.savefig('3D_{}.png'.format(AFF_name),facecolor='white')

    return()

np.random.seed(1)
file = "image_train00.npy"
vox_size=[2, 0.5, 0.5]
IM = Image3D (file, vox_size=[2, 0.5, 0.5])
image = IM.load()

    #random - 1
R1 = AffineTransform(scale=1)
Image_R1 = IM.warp(R1.matrix)
P=plot("RT_S1_1",Image_R1,vox_size)

#random - 2
R2 = AffineTransform(scale=1)
Image_R2 = IM.warp (R2.matrix)
P=plot("RT_S1_2",Image_R2,vox_size)

#random - 3
R3 = AffineTransform()
r3 = R3.random_transfrom_generator(0.8)
Image_R3 = IM.warp (r3)
P=plot("RT_S0.8_1",Image_R3,vox_size)

#random - 4
R4 = AffineTransform()
r4 = R4.random_transfrom_generator(0.8)
Image_R4 = IM.warp (r4)
P=plot("RT_S0.8_2",Image_R4,vox_size)

#random - 5
R5 = AffineTransform()
r5 = R5.random_transfrom_generator(1.2)
Image_R5 = IM.warp (r5)
P=plot("RT_S1.2_1",Image_R5,vox_size)

#random - 6
R6 = AffineTransform()
r6 = R6.random_transfrom_generator(1.2)
Image_R6 = IM.warp (r6)
P=plot("RT_S1.2_2",Image_R6,vox_size)

#random - 7
R7 = AffineTransform()
r7 = R7.random_transfrom_generator(1.5)
Image_R7 = IM.warp (r7)
P=plot("RT_S1.5_1",Image_R7,vox_size)

#random - 8
R8 = AffineTransform()
r8 = R8.random_transfrom_generator(1.5)
Image_R8 = IM.warp (r8)
P=plot("RT_S1.5_2",Image_R8,vox_size)

#random - 9
R9 = AffineTransform()
r9 = R9.random_transfrom_generator(1.8)
Image_R9 = IM.warp (r9)
P=plot("RT_S1.8_1",Image_R9,vox_size)

#random - 10
R10 = AffineTransform()
r10 = R10.random_transfrom_generator(1.8)
Image_R10 = IM.warp (r10)
P=plot("RT_S1.8_2",Image_R10,vox_size)

#transformation-1
Transl = [0,0,0]
Rotation = [45,45,90]
M_1 = AffineTransform(transl=Transl,rot=Rotation)
Image_AFF_1 = IM.warp (M_1.matrix)
P=plot("Aff_1",Image_AFF_1,vox_size)

#transformation-2
Transl = [10,10,20]
Rotation = [0,0,0]
M_2 = AffineTransform(transl=Transl,rot=Rotation)
Image_AFF_2 = IM.warp (M_2.matrix)
P=plot("Aff_2",Image_AFF_2,vox_size)

#transformation-3
Transl = [20,0,0]
Rotation = [0,0,90]
M_3 = AffineTransform(transl=Transl,rot=Rotation)
Image_AFF_3 = IM.warp (M_3.matrix)
P=plot("Aff_3",Image_AFF_3,vox_size)

#transformation-4
Transl = [20,20,0]
Rotation = [0,0,0]
M_4 = AffineTransform(transl=Transl,rot=Rotation)
Image_AFF_4 = IM.warp (M_4.matrix)
P=plot("Aff_4",Image_AFF_4,vox_size)

#transformation-5
Transl = [0,0,0]
Rotation = [0,90,90]
M_5 = AffineTransform(transl=Transl,rot=Rotation)
Image_AFF_5 = IM.warp (M_5.matrix)
P=plot("Aff_5",Image_AFF_5,vox_size)

#transformation-6
Transl = [0,10,10]
Rotation = [0,0,0]
Scale = 2
M_6 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
Image_AFF_6 = IM.warp (M_6.matrix)
P=plot("Aff_6",Image_AFF_6,vox_size)

#transformation-7
Transl = [20,20,20]
Rotation = [0,0,0]
Scale = 5
M_7 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
Image_AFF_7 = IM.warp (M_7.matrix)
P=plot("Aff_7",Image_AFF_7,vox_size)

#transformation-8
Transl = [0,0,0]
Rotation = [45,45,45]
Scale = 2
M_8 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
Image_AFF_8 = IM.warp (M_8.matrix)
P=plot("Aff_8",Image_AFF_8,vox_size)

#transformation-9
Transl = [0,10,0]
Rotation = [45,0,0]
Scale = 4
M_9 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
Image_AFF_9 = IM.warp (M_9.matrix)
P=plot("Aff_9",Image_AFF_9,vox_size)

#transformation-10
Transl = [-20,10,0]
Rotation = [90,90,90]
Scale = 0.7
M_10 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
Image_AFF_10 = IM.warp (M_10.matrix)
P=plot("Aff_10",Image_AFF_10,vox_size)

