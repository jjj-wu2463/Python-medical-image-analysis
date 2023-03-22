import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interpn


class Image3D():

    def __init__(self, file=None, Transf_matrix=None, vox_size=None):
        """
        Attributes
        ----------
        file : str.
            Input file name as string.

        Transf_matrix : (4x4) ndarray 
            Transformation matrix from ``AffineTransform``.

        vox_size : tuple (3)
            Voxel dimensions given by (Z,X,Y), where z is the slice distance; x and y are axial spacing.

        """

        self.file = file
        self.Transf_matrix = Transf_matrix
        self.vox_size = vox_size

    def load (self):
        """
        Loads input image file.

        Returns
        --------------
        input: ndarray 
            Returns as array with size same as image size.

        """
        root,ext=os.path.splitext(self.file)

        if (ext=='.npy'):
            self.input = np.load(self.file)

        elif (ext=='.dcm'):
            input = dicom.read_file(self.path)

        elif (ext=='.nii'):
            input = nib.load(self.pathpath)

        return self.input

    def warp (self, Transf_matrix):
        """
        Computes a  warped 3D image.

        Parameters
        --------------
        Transf_matrix: (4x4) ndarray  
            Transformation matrix.

        Returns
        --------------
        Affined_image: ndarray 
            Returns a padded transformed image.

        Attributes
        --------------
        Affined_image: ndarray 
            Returns a padded transformed image.

        """
        self.input
        z,x,y = self.vox_size
        self.Transf_matrix = Transf_matrix
        arr_min = np.min(self.input)
        self.normalised_im = (self.input-arr_min)/(np.max(self.input)-arr_min)

    #pad image for transformation
        image_pad = np.pad(self.normalised_im,[(60,60),(60,60),(60,60)])
        points_i_pad = np.arange(0, image_pad.shape[0]*z,z)
        points_j_pad = np.arange(0, image_pad.shape[1]*x,x)
        points_k_pad = np.arange(0, image_pad.shape[2]*y,y)

    #shape = np.shape(im)
        ZZZ_pad = np.arange(0, image_pad.shape[0]*z,z)
        XXX_pad = np.arange(0, image_pad.shape[1]*x,x)
        YYY_pad = np.arange(0, image_pad.shape[2]*y,y)
        ZZZ, XXX, YYY = np.meshgrid(ZZZ_pad, XXX_pad, YYY_pad, indexing='ij')
        image_coordinates_pad = np.concatenate((ZZZ.reshape(-1,1), XXX.reshape(-1,1), YYY.reshape(-1,1), np.ones((np.prod(image_pad.shape), 1))), axis=1).T

    # centering transformation
        
        centre = (np.array(image_pad.shape))//2

        centering_tf_pad = np.eye(4)
        centering_tf_pad[0, 3] = 0 - ZZZ_pad[centre[0]]
        centering_tf_pad[1, 3] = 0 - XXX_pad[centre[1]]
        centering_tf_pad[2, 3] = 0 - YYY_pad[centre[2]]


    # inverse centering transformation
        centering_tf_inv_pad = np.eye(4)
        centering_tf_inv_pad[0, 3] = -centering_tf_pad[0, 3]
        centering_tf_inv_pad[1, 3] = -centering_tf_pad[1, 3]
        centering_tf_inv_pad[2, 3] = -centering_tf_pad[2, 3]

    # compute padded matrix
        M_pad = np.dot(centering_tf_inv_pad, np.dot(self.Transf_matrix, centering_tf_pad))

        points = (points_i_pad, points_j_pad, points_k_pad)
        values = image_pad
        Mi=np.dot(M_pad, image_coordinates_pad)[:3].T
        Mi=np.asarray(Mi)

        image_interpn_flatten = interpn(points, values, Mi, bounds_error=False, fill_value=0)
        self.Affined_image = image_interpn_flatten.reshape(image_pad.shape)

        return(self.Affined_image)
                
        

class AffineTransform():

    def __init__(self, transl=[0,0,0], rot=[0,0,0], scale=1, shear=0):
        """
        Attributes
        ----------
        transl : tuple or a list (3,)
            Translation parameters (z,x,y).
        
        rot : tuple or a list (3,)
            Rotation angle along the (z,x,y) aixs.

        scale : int or float
            Uniform scale along the z,x,y axis.

        shear : int or float
            Uniform shear along the z,x,y axis.

        """
        self.transl = transl
        self.rot = rot
        self.scale = scale
        self.shear = shear

        if transl == [0,0,0] and rot == [0,0,0] and shear == 0:
            self.random_transfrom_generator(self.scale)
        
        else:
            self.affine_transform(transl=self.transl, rot=self.rot, scale=self.scale, shear=self.shear)


    def shape_transl(self):
        """
        Translation input shape check.

        """
        if np.shape(self.transl) != (3,):
            raise ValueError("Invalid length for translation vector.")            

    def shape_rot(self):
        """
        Rotation input shape check.

        """
        if np.shape(self.rot) != (3,):
            raise ValueError("Invalid length for rotational vector.")

    def shape(self):
        """
        Translation and rotation input shape check.
        """
        self.shape_transl()
        self.shape_rot()

    #def affine_transform(self, transl=None, rot=None, scale=None, shear=None):
    def affine_transform(self, transl=[0,0,0], rot=[0,0,0], scale=1, shear=0):
        """
        Computes a (4x4) transformation matrix.

        Attributes
        ----------
        transl : tuple or a list (3,)
            Translation parameters (z,x,y).
        
        rot : tuple or a list (3,)
            Rotation angle along the (z,x,y) aixs.

        scale : int or float
            Uniform scale along the z,x,y axis.

        shear : int or float
            Uniform shear along the z,x,y axis.

        matrix: ndarray (4x4)
            Returns a transformation matrix.

        Parameters
        -----------
        transl : tuple or a list (3,)
            Translation parameters (z,x,y).
        
        rot : tuple or a list (3,)
            Rotation angle along the (z,x,y) aixs.

        scale : int or float
            Uniform scale along the z,x,y axis.

        shear : int or float
            Uniform shear along the z,x,y axis. 

        Returns
        --------------
        matrix: ndarray (4x4)
            Returns a transformation matrix.

        """
        self.shape()
        self.scale = scale
        self.shear = shear
        self.transl = transl
        self.rot = rot

        #rotation matrix
        Rx, Ry, Rz = np.array(np.radians(self.rot))

        rx = np.matrix([[1, 0, 0, 0], [0, np.cos(Rx), -np.sin(Rx), 0], [0, np.sin(Rx), np.cos(Rx), 0], [0, 0, 0, 1]])
        ry = np.matrix([[np.cos(Ry), 0, np.sin(Ry), 0], [0, 1, 0, 0], [-np.sin(Ry), 0, np.cos(Ry), 0], [0, 0, 0, 1]])
        rz = np.matrix([[np.cos(Rz), -np.sin(Rz), 0, 0], [np.sin(Rz), np.cos(Rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        M_rot = np.dot(rx, np.dot(ry, rz))
        
        #translation matrix
        Tx, Ty, Tz = self.transl
        M_transl = np.matrix([[1,0,0,Tx],[0,1,0,Ty],[0,0,1,Tz],[0,0,0,1]])

        #scalar matrix
        M_scalar=np.matrix([[1/self.scale,0,0,0],[0,1/self.scale,0,0],[0,0,1/self.scale,0],[0,0,0,1]])

        #shear matrix
        M_shear=np.matrix([[1,self.shear,self.shear,0],[self.shear,1,self.shear,0],[self.shear,self.shear,1,0],[0,0,0,1]])
        
        #combined transformation
        self.matrix=np.dot(M_shear,np.dot(M_scalar,np.dot(M_transl, M_rot)))

        return(self.matrix)

    def rigid_transform(self, transl=[0,0,0], rot=[0,0,0], scale=1):
        """
        Computes a (4x4) transformation matrix.

        Attributes
        ----------
        transl : tuple or a list (3,)
            Translation parameters (z,x,y).
        
        rot : tuple or a list (3,)
            Rotation angle along the (z,x,y) aixs.

        scale : int or float
            Uniform scale along the z,x,y axis.

        matrix: ndarray (4x4)
            Returns a transformation matrix.
        
        Parameters
        --------------
        transl : tuple or a list (3,)
            Translation parameters (z,x,y).
        
        rot : tuple or a list (3,)
            Rotation angle along the (z,x,y) aixs.

        scale : int or float
            Uniform scale along the z,x,y axis.

        Returns
        --------------
        matrix: ndarray (4x4)
            Returns a transformation matrix.

        """
        self.shape()
        self.scale = scale
        self.transl = transl
        self.rot = rot

        #rotation matrix
        Rx, Ry, Rz = np.radians(self.rot)

        rx = np.matrix([[1, 0, 0, 0], [0, np.cos(Rx), -np.sin(Rx), 0], [0, np.sin(Rx), np.cos(Rx), 0], [0, 0, 0, 1]])
        ry = np.matrix([[np.cos(Ry), 0, np.sin(Ry), 0], [0, 1, 0, 0], [-np.sin(Ry), 0, np.cos(Ry), 0], [0, 0, 0, 1]])
        rz = np.matrix([[np.cos(Rz), -np.sin(Rz), 0, 0], [np.sin(Rz), np.cos(Rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        M_rot = np.dot(rx, np.dot(ry, rz))
        
        #translation matrix
        Tx, Ty, Tz = self.transl
        M_transl = np.matrix([[1,0,0,Tx],[0,1,0,Ty],[0,0,1,Tz],[0,0,0,1]])

        #scalar matrix
        M_scalar=np.matrix([[(1/self.scale),0,0,0],[0,(1/self.scale),0,0],[0,0,(1/self.scale),0],[0,0,0,1]])
        
        #combined transformation
        self.matrix=np.dot(M_scalar,np.dot(M_transl, M_rot))

        return(self.matrix)

    def random_transfrom_generator(self, scale=1):   
        """
        Computes a (4x4) transformation matrix.

        Attributes
        ----------
        scale : int or float
            Uniform scale along the z,x,y axis.

        matrix: ndarray (4x4)
            Returns a transformation matrix.

        Parameters
        --------------
        scale : int or float
            Uniform scale along the x,y,z axis.

        Returns
        --------------
        matrix: ndarray (4x4)
            Returns a randomly generated transformation matrix.

        """
        self.shape()
        self.scale = scale
        self.transl = np.random.randint(0,30,(3,))    
        self.rot = np.random.randint(0,180,(3,))

        #rotation matrix
        Rx, Ry, Rz = np.array(np.radians(self.rot))
       
        rx = np.matrix([[1, 0, 0, 0], [0, np.cos(Rx), -np.sin(Rx), 0], [0, np.sin(Rx), np.cos(Rx), 0], [0, 0, 0, 1]])
        ry = np.matrix([[np.cos(Ry), 0, np.sin(Ry), 0], [0, 1, 0, 0], [-np.sin(Ry), 0, np.cos(Ry), 0], [0, 0, 0, 1]])
        rz = np.matrix([[np.cos(Rz), -np.sin(Rz), 0, 0], [np.sin(Rz), np.cos(Rz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        M_rot = np.dot(rx, np.dot(ry, rz))
        
        #translation matrix
        Tx, Ty, Tz = self.transl
        M_transl = np.matrix([[1,0,0,Tx],[0,1,0,Ty],[0,0,1,Tz],[0,0,0,1]])

        #scalar matrix
        M_scalar=np.matrix([[1/self.scale,0,0,0],[0,1/self.scale,0,0],[0,0,1/self.scale,0],[0,0,0,1]])
        
        #combined transformation
        self.matrix=np.dot(M_scalar,np.dot(M_transl, M_rot))

        return(self.matrix)


def plot(AFF_name, Affined_image):
    """
    Saves 5 selected image slices in '.png' files and returns a sparsely sampled image array.

    Parameters
    --------------
    AFF_name : str.
        Affine transform name.
    
    Affined_image : ndarray
        Warped/transformed image.

    Returns
    --------------
    image_plot : ndarray 
        sparsely sampled image array.

    """

    image_plot = np.zeros(Affined_image.shape)
    val = np.array(np.where(Affined_image>0)).T
    m = val[0,0] +((val[-1,0] - val[0,0])//2)


    slice_range = np.arange(m-10,m+15,5)
    slice_range = slice_range.astype(int)

    for i in slice_range:
        image_plot[i,:,:] = Affined_image[i,:,:]
        plt.imsave('{}_slice_{}.png'.format(AFF_name,i), Affined_image[i,:,:], cmap='gray')

    return(image_plot)

if __name__ == "__main__" :
    #generate a constant set of randomly generated values.
    np.random.seed(1)

    file = "image_train00.npy"
    IM = Image3D (file, vox_size=[2, 0.5, 0.5])
    image = IM.load()

    #random - 1
    R1 = AffineTransform(scale=1)
    Image_R1 = IM.warp(R1.matrix)
    Plot_R1 = plot("RT_S1_1",Image_R1)

    #random - 2
    R2 = AffineTransform(scale=1)
    Image_R2 = IM.warp (R2.matrix)
    Plot_R2 = plot("RT_S1_2",Image_R2)

    #random - 3
    R3 = AffineTransform()
    r3 = R3.random_transfrom_generator(0.8)
    Image_R3 = IM.warp (r3)
    Plot_R3 = plot("RT_S0.8_1",Image_R3)

    #random - 4
    R4 = AffineTransform()
    r4 = R4.random_transfrom_generator(0.8)
    Image_R4 = IM.warp (r4)
    Plot_R4 = plot("RT_S0.8_2",Image_R4)

    #random - 5
    R5 = AffineTransform()
    r5 = R5.random_transfrom_generator(1.2)
    Image_R5 = IM.warp (r5)
    Plot_R5 = plot("RT_S1.2_1",Image_R5)

    #random - 6
    R6 = AffineTransform()
    r6 = R6.random_transfrom_generator(1.2)
    Image_R6 = IM.warp (r6)
    Plot_R6 = plot("RT_S1.2_2",Image_R6)

    #random - 7
    R7 = AffineTransform()
    r7 = R7.random_transfrom_generator(1.5)
    Image_R7 = IM.warp (r7)
    Plot_R7 = plot("RT_S1.5_1",Image_R7)

    #random - 8
    R8 = AffineTransform()
    r8 = R8.random_transfrom_generator(1.5)
    Image_R8 = IM.warp (r8)
    Plot_R8 = plot("RT_S1.5_2",Image_R8)

    #random - 9
    R9 = AffineTransform()
    r9 = R9.random_transfrom_generator(1.8)
    Image_R9 = IM.warp (r9)
    Plot_R9 = plot("RT_S1.8_1",Image_R9)

    #random - 10
    R10 = AffineTransform()
    r10 = R10.random_transfrom_generator(1.8)
    Image_R10 = IM.warp (r10)
    Plot_R10 = plot("RT_S1.8_2",Image_R10)

    #transformation-1
    Transl = [0,0,0]
    Rotation = [45,45,90]
    M_1 = AffineTransform(transl=Transl,rot=Rotation)
    Image_AFF_1 = IM.warp (M_1.matrix)
    Plot_1 = plot("Aff_1",Image_AFF_1)

    #transformation-2
    Transl = [10,10,20]
    Rotation = [0,0,0]
    M_2 = AffineTransform(transl=Transl,rot=Rotation)
    Image_AFF_2 = IM.warp (M_2.matrix)
    Plot_2 = plot("Aff_2",Image_AFF_2)

    #transformation-3
    Transl = [20,0,0]
    Rotation = [0,0,90]
    M_3 = AffineTransform(transl=Transl,rot=Rotation)
    Image_AFF_3 = IM.warp (M_3.matrix)
    Plot_3 = plot("Aff_3",Image_AFF_3)

    #transformation-4
    Transl = [20,20,0]
    Rotation = [0,0,0]
    M_4 = AffineTransform(transl=Transl,rot=Rotation)
    Image_AFF_4 = IM.warp (M_4.matrix)
    Plot_4 = plot("Aff_4",Image_AFF_4)

    #transformation-5
    Transl = [0,0,0]
    Rotation = [0,90,90]
    M_5 = AffineTransform(transl=Transl,rot=Rotation)
    Image_AFF_5 = IM.warp (M_5.matrix)
    Plot_5 = plot("Aff_5",Image_AFF_5)

    #transformation-6
    Transl = [0,10,10]
    Rotation = [0,0,0]
    Scale = 2
    M_6 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
    Image_AFF_6 = IM.warp (M_6.matrix)
    Plot_6 = plot("Aff_6",Image_AFF_6)

    #transformation-7
    Transl = [20,20,20]
    Rotation = [0,0,0]
    Scale = 5
    M_7 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
    Image_AFF_7 = IM.warp (M_7.matrix)
    Plot_7 = plot("Aff_7",Image_AFF_7)

    #transformation-8
    Transl = [0,0,0]
    Rotation = [45,45,45]
    Scale = 2
    M_8 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
    Image_AFF_8 = IM.warp (M_8.matrix)
    Plot_8 = plot("Aff_8",Image_AFF_8)

    #transformation-9
    Transl = [0,10,0]
    Rotation = [45,0,0]
    Scale = 4
    M_9 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
    Image_AFF_9 = IM.warp (M_9.matrix)
    Plot_9 = plot("Aff_9",Image_AFF_9)

    #transformation-10
    Transl = [-20,10,0]
    Rotation = [90,90,90]
    Scale = 0.7
    M_10 = AffineTransform(transl=Transl,rot=Rotation,scale=Scale)
    Image_AFF_10 = IM.warp (M_10.matrix)
    Plot_10 = plot("Aff_10",Image_AFF_10)

    #Consider what the precomputing can be done during construction of a new image object, and implement them with comments
    print('The precomputing that can be done to construct a new image includes loading the original image,resampling or reslicing (if necessary). To construct a transformed image, precomputing includes defining the transformation parameters, such as translation and rotation, and combine it into a 4x4 matrix. Furthermore, precomputing includes sufficient padding around the image to allow interpolation during transformation.')

    #Consider what the best way to specify local image coordinates, and implement them with brief comments explain the rationale.
    print('The best way to specify local image coordinates is by redefining the image coordinates, hence, centering the origin of the new coordinate system to the centre of the padded image. Through this method, the centre of the image is set to (0,0,0). Arrays of Z,X,Y can be created and scaled, according to the voxel dimensions, thus, accounting for voxel size. The Z,X,Y arrays can be concatenated and meshed to form a scaled and centred coordinate system. In additon, the transformation matrix can be applied to the center(origin) of the image to transform the image about image centre point.')



