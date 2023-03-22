# %%
#FINAL
import nibabel as nib
import numpy as np
import skimage.metrics
import skimage.util
import scipy.ndimage
import medpy.filter 
import matplotlib.pyplot as plt
from matplotlib import cm
import time

#NORMALISED Sliced Images
def normalise(image):
    """
    This functions takes in an array of 2D or 3D. The minimum value present 
    in the array(min) is identified and set as a reference value for all other 
    values in the array. 
    The range of the array is identified as the difference between the 
    maximum (max) and minimum values.
    The difference is taken between each value in the array and the min.
    These difference are compared against the range of the array, resulting
    in a value between 0 and 1.

    Parameters
    --------------
        image: array_like 
            Array of which to normlise.
    
    Returns: 
    --------------
        normalise: array_like
            A normalised array (values ranging from 0 to 1).
    """
    arr_min = np.min(image)
    normalise = (image-arr_min)/(np.max(image)-arr_min)

    return(normalise)

def non_orthogonal(image, rot_angle, slice_y):
    """
    This functions takes in a 3D array. The array is rotated 90° along the y-axis.
    This rotation results in an upright image of the abdominal organs when viewed
    in 2D. This is followed by a rotation along the z-axis by (rot_angle)°. A is
    taken along the y-axis of the roated body. The slice is known as the non- 
    orthogonal plane of the image.
    
    Parameters
    --------------
        image: array_like (3D)
            A 3D Array containing voxel values.

        rot_angle: int.
            Angle to be rotated along the z-axis.

        slice_y: int.
            The number of slice to view along the y-axis of the 3D array.

    Returns
    --------------
        sliced_im: array_like (2D)
            A 2D array containing a slice along the y-axis, this is known as the
            non-orthogonal plane.

    """
    image_rotate = scipy.ndimage.interpolation.rotate(image, angle=90, axes=(0,2))
    image_rotate = scipy.ndimage.interpolation.rotate(image_rotate, angle=rot_angle, axes=(1,2),reshape=False)
    sliced_im = image_rotate[:,slice_y,:]

    return(sliced_im)

#Load NIFTI Image
image = nib.load('image.nii.gz').get_fdata()
#Specify slice to view
slice_y = 114

##varying angles of non-orthogonal plane
image_rotate_1 =  non_orthogonal(image, 0 , slice_y)
image_rotate_2 =  non_orthogonal(image, 4 , slice_y)
image_rotate_3 =  non_orthogonal(image, 8 , slice_y)
image_rotate_4 =  non_orthogonal(image, 12 , slice_y)
image_rotate_5 =  non_orthogonal(image, 16 , slice_y)

fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (12, 6))
#original
ax1.imshow(image_rotate_1,cmap='gray')
ax1.set_title('Orthogonal plane')
#Non-orthogonal plane (x+4°)
ax2.imshow(image_rotate_2,cmap='gray')
ax2.set_title('Rotated 4° \n along the z-axis ')
#Non-orthogonal plane (x+8°)
ax3.imshow(image_rotate_3,cmap='gray')
ax3.set_title('Rotated 8° \n along the z-axis')
#Non-orthogonal plane (x+12°)
ax4.imshow(image_rotate_4,cmap='gray')
ax4.set_title('Rotated 12° \n along the z-axis')
#Non-orthogonal plane (x+16°)
ax5.imshow(image_rotate_5,cmap='gray')
ax5.set_title('Rotated 16° \n along the z-axis')

plt.savefig('Varying_angles.png', facecolor='white')

##varying slices
image_slice_1 = non_orthogonal(image, 8 , 106)
image_slice_2 = non_orthogonal(image, 8 , 110)
image_slice_3 = non_orthogonal(image, 8 , 114)
image_slice_4 = non_orthogonal(image, 8 , 118)
image_slice_5 = non_orthogonal(image, 8 , 122)

fig2, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (12, 6))
#Slice 110 along the y-axis
ax1.imshow(image_slice_1,cmap='gray')
ax1.set_title('Slice 106 \n along the y-axis')
#Slice 112 along the y-axis
ax2.imshow(image_slice_2,cmap='gray')
ax2.set_title('Slice 110 \n along the y-axis ')
#Slice 114 along the y-axis
ax3.imshow(image_slice_3,cmap='gray')
ax3.set_title('Slice 114 \n along the y-axis')
#Slice 116 along the y-axis
ax4.imshow(image_slice_4,cmap='gray')
ax4.set_title('Slice 118 \n along the y-axis')
#Slice 118 along the y-axis
ax5.imshow(image_slice_5,cmap='gray')
ax5.set_title('Slice 122 \n along the y-axis')

plt.savefig('Varying_slices.png', facecolor='white')

#Plot 3D
image_rotate = scipy.ndimage.interpolation.rotate(image, angle=90, axes=(0,2))
image_rotate = scipy.ndimage.interpolation.rotate(image_rotate, angle=8, axes=(1,2),reshape=False)
image_rotate = normalise(image_rotate)

image_plot = np.zeros(image_rotate.shape)

slices = np.arange(106,126,4)

for s in slices:
    image_plot[:,s,:] = image_rotate[:,s,:]

facecolors = cm.gray(image_plot)
facecolors[:,:,:,-1] = image_plot
shape_arr = np.array(facecolors.shape)
size = shape_arr[:3]*2 - 1

filled = facecolors[:,:,:,-1] != 0
z, y, x = np.indices(np.array(filled.shape)+1)
z[1::2, :, :] += 1
y[:, 1::2, :] += 1
x[:, :, 1::2] += 1

fig3 = plt.figure(figsize=(100, 100))
ax = fig3.add_subplot(111, projection='3d')
ax.tick_params(axis='x', labelsize=40)
ax.tick_params(axis='y', labelsize=40)
ax.tick_params(axis='z', labelsize=40)

ax.set_xlabel("x-axis", fontsize=70, labelpad=40)
ax.set_ylabel("y-axis", fontsize=70, labelpad=40)
ax.set_zlabel("z-axis", fontsize=70, labelpad=40)
#ax.voxels(z,y,x, filled,facecolors=facecolors, shade=False)
ax.set_title("Visualization of slices 106, 110, 114, 118, & 122 along the y-axis" , fontsize=120)
#plt.savefig('Slice_3D visualisation.png',facecolor='white')

#Adding noise for filter comparison
norm_slice = image_rotate[:,114,:]
norm_slice = norm_slice.astype(np.float64)
Noise_volume = (skimage.util.random_noise(image_rotate, mode='s&p', amount=0.02))
Noise_slice = (Noise_volume[:,slice_y,:])
Noise_slice = Noise_slice.astype(np.float64)

#SSI and PSNR for noise image
SSI_2D_N = np.round(skimage.metrics.structural_similarity(norm_slice, Noise_slice),6)
PSNR_2D_N = np.round(skimage.metrics.peak_signal_noise_ratio(norm_slice, Noise_slice),6)

##2D FILTERING
#niter=1, option=1
#Varying Gamma
vox_size = (2,2,2)

Gamma = np.linspace(0.1,0.25,4)
Filter_2D_G = []
SSI_2D_G = []
PSNR_2D_G = []

for g in Gamma:
    Filter_2D_G1 = medpy.filter.smoothing.anisotropic_diffusion(Noise_slice, gamma=g, voxelspacing=vox_size)
    Filter_2D_G1 = Filter_2D_G1.astype(np.float64)
    Filter_2D_G.append(Filter_2D_G1)
    SSI_2D_G1 = np.round(skimage.metrics.structural_similarity(norm_slice, Filter_2D_G1),6)
    SSI_2D_G.append(SSI_2D_G1)
    PSNR_2D_G1 = np.round(skimage.metrics.peak_signal_noise_ratio(norm_slice, Filter_2D_G1),6)
    PSNR_2D_G.append(PSNR_2D_G1)

fig4, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (12, 6))
#Noisy image
ax1.imshow(Noise_slice,cmap='gray')
ax1.set_title('Noisy image\n SSIM = {}\n PSNR = {}'.format(SSI_2D_N, PSNR_2D_N))
#Gamma =0.10
ax2.imshow(Filter_2D_G[0],cmap='gray')
ax2.set_title('Gamma = 0.10\n SSIM = {}\n PSNR = {}'.format(SSI_2D_G[0], PSNR_2D_G[0]))
#Gamma =0.15
ax3.imshow(Filter_2D_G[1],cmap='gray')
ax3.set_title('Gamma = 0.15\n SSIM = {}\n PSNR = {}'.format(SSI_2D_G[1], PSNR_2D_G[1]))
#Gamma =0.20
ax4.imshow(Filter_2D_G[2],cmap='gray')
ax4.set_title('Gamma = 0.20\n SSIM = {}\n PSNR = {}'.format(SSI_2D_G[2], PSNR_2D_G[2]))
#Gamma =0.25
ax5.imshow(Filter_2D_G[3],cmap='gray')
ax5.set_title('Gamma =0.25\n SSIM = {}\n PSNR = {}'.format(SSI_2D_G[3], PSNR_2D_G[3]))
plt.savefig('Varying_Gamma.png', facecolor='white')

#Varying Kappa
Kappa = np.linspace(50,110,4)
Filter_2D_K = []
SSI_2D_K = []
PSNR_2D_K = []

for k in Kappa:
    Filter_2D_K1 = medpy.filter.smoothing.anisotropic_diffusion(Noise_slice,kappa=k, gamma=0.25, voxelspacing=vox_size)
    Filter_2D_K1 = Filter_2D_K1.astype(np.float64)
    Filter_2D_K.append(Filter_2D_K1)
    SSI_2D_K1 = np.round(skimage.metrics.structural_similarity(norm_slice, Filter_2D_K1),6)
    SSI_2D_K.append(SSI_2D_K1)
    PSNR_2D_K1 = np.round(skimage.metrics.peak_signal_noise_ratio(norm_slice, Filter_2D_K1),6)
    PSNR_2D_K.append(PSNR_2D_K1)

fig5, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (12, 6))
#Noisy image
ax1.imshow(Noise_slice,cmap='gray')
ax1.set_title('Noisy image\n SSIM = {}\n PSNR = {}'.format(SSI_2D_N, PSNR_2D_N))
#Kappa = 50
ax2.imshow(Filter_2D_K[0],cmap='gray')
ax2.set_title('Kappa = 50\n SSIM = {}\n PSNR = {}'.format(SSI_2D_K[0], PSNR_2D_K[0]))
#Kappa = 70
ax3.imshow(Filter_2D_K[1],cmap='gray')
ax3.set_title('Kappa = 70\n SSIM = {}\n PSNR = {}'.format(SSI_2D_K[1], PSNR_2D_K[1]))
#Kappa = 90
ax4.imshow(Filter_2D_K[2],cmap='gray')
ax4.set_title('Kappa = 90\n SSIM = {}\n PSNR = {}'.format(SSI_2D_K[2], PSNR_2D_K[2]))
#Kappa = 110
ax5.imshow(Filter_2D_K[3],cmap='gray')
ax5.set_title('Kappa =110\n SSIM = {}\n PSNR = {}'.format(SSI_2D_K[3], PSNR_2D_K[3]))
plt.savefig('Varying_Kappa.png', facecolor='white')

#TIMING 2D FILTERING
start = time.time()
Noise_slice = (Noise_volume[:,slice_y,:])
Filter_2D = medpy.filter.smoothing.anisotropic_diffusion(Noise_slice, kappa=50, gamma=0.25, voxelspacing=(2,2,2))
end = time.time()
time_2D=np.around((end - start),4)

#3D Filtering
#TIMING 3D FILTERING
start = time.time()
Filter_3D = medpy.filter.smoothing.anisotropic_diffusion(Noise_volume, kappa=50, gamma=0.25, voxelspacing=(2,2,2))
Filter_3D_slice = Filter_3D[:,slice_y,:]
end = time.time()
time_3D=np.around((end - start),4)
Filter_3D_slice = Filter_3D_slice.astype(np.float64)
SSI_3D= np.round(skimage.metrics.structural_similarity(norm_slice, Filter_3D_slice),6)
PSNR_3D= np.round(skimage.metrics.peak_signal_noise_ratio(norm_slice, Filter_3D_slice),6)

#Plot
fig6, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (12, 6))
#original
ax1.imshow(image_rotate_1,cmap='gray')
ax1.set_title('Image')
#rotated
ax2.imshow(image_rotate_3,cmap='gray')
ax2.set_title('Non-Orthogonal Plane\nRotated 8° \n along the z-axis')
#Noise image
ax3.imshow(Noise_slice,cmap='gray')
ax3.set_title('S&P Noise Image \n SSIM = {}\n PSNR = {}'.format(SSI_2D_N,PSNR_2D_N))
#2D filter
ax4.imshow(Filter_2D_K[0],cmap='gray')
ax4.set_title('2D Filter\n  SSIM = {}\n PSNR = {}'.format(SSI_2D_K[0], PSNR_2D_K[0]))
#3D filter
ax5.imshow(Filter_3D_slice,cmap='gray')
ax5.set_title('3D Filter\n  SSIM = {}\n PSNR = {}'.format(SSI_3D, PSNR_3D))
plt.savefig('Comparing_Filters.png', facecolor='white')

#percentage comparison
pct_SSI_2D = (abs(SSI_2D_K[0]-SSI_2D_N)/SSI_2D_N)*100
pct_SSI_3D = (abs(SSI_3D-SSI_2D_N)/SSI_2D_N)*100
pct_PSNR_2D = (abs(PSNR_2D_K[0]-PSNR_2D_N)/PSNR_2D_N)*100
pct_PSNR_3D = (abs(PSNR_3D-PSNR_2D_N)/PSNR_2D_N)*100

# %%
