import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure

def surface_normals_np(verts, faces):
    """ 
    Compute the vertex and surface normals from given faces and vertex co-ordinates,
    computed from ``marching_cubes``.
    The surface normals can be computed by taking the two vector differences between three points, 
    ensuring both vectors are converging to tge same point. The cross product of both vectors give
    the surface normal of the triangle face. 
    The sum of surface vectors where the particular vertex involved results in the vertex vector.
    
    Parameters
    --------------
    verts: (V,3) array 
        An array containing all vertex co-ordinates.

    faces: (V,3) array
        An array containing the index of vertices in each faces.

    Returns
    --------------
    norm_V: (V,3) array 
        An array consisting vertex normals at each vertex, respectively. 

    norm_S: (S,3) array
        An array consisting surface normals at each surface, repsectively.
    """
    #assign an empty array for vertex normals
    norm_V = np.zeros(verts.shape, np.float32)

    #compute vector difference between three points of the triangle 
    #both vectors converge to the same vertex.
    V1 = verts[faces[:,1]] - verts[faces[:,0]]
    V2 = verts[faces[:,2]] - verts[faces[:,0]]

    #calculate the surface normal
    norm_S = np.cross(V2,V1)

    # compute vertex normals
    # face normals of the faces where the respective vertex is indexed, is summed.
    sx, sy = verts.shape
    range = np.arange(0,sx)

    for i in range:
        idx_x = np.asarray(np.where(faces[:,0]==i))
        idx_y = np.asarray(np.where(faces[:,1]==i))
        idx_z = np.asarray(np.where(faces[:,2]==i))

        idx = np.concatenate ((idx_x, idx_y, idx_z), axis=1)

        for j in idx:
            norm_V [i] = (np.sum (norm_S[j,0]), np.sum (norm_S[j,1]), np.sum (norm_S[j,2]))

    return (norm_V,norm_S)

def gaussian (input,sigma,vox_size):
    """
    Computes vertex and surface normals of a smoothened surface. The input image is smoothened and filtered with
    with ``gaussian_filter``. The strength of this filter can be controlled by sigma. The voxel dimensions are 
    considered in the Gaussian filter. The function then computes the vertices, faces, and vertex normals
    with ``marching_cubes``. The array of vertices and faces acts as an input for ``surface_normals_np`` to compute
    vertex and surface normals.
    
    Parameters
    --------------
    input: array_like 
        Array to be filtered.

    sigma: scalar 
        Controls the strength of Gaussian filter.

    vox_size: tuple (3)
        Voxel spacing in spatial dimensions with the format (Z,X,Y).
        Z being slice distance; X and Y being the axial spacing.

    Returns
    --------------
    normals: (V,3) array
        An array consisting vertex normals at each vertex, respectively. 

    norm_V: (V,3) array
        An array consisting vertex normals at each vertex, respectively. 

    norm_S: (S,3) array 
        An array consisting surface normals at each surface, repsectively.
    """
    volume=input.astype(np.float32)
    volume= gaussian_filter(volume,sigma)
    verts, faces, normals, values = measure.marching_cubes(volume, spacing=vox_size)

    norm_V, norm_S = surface_normals_np(verts,faces)

    return (normals, norm_V, norm_S)


#computing triangluated surface as input
input = np.load("label_train00.npy")
vox_size =  (2, 0.5, 0.5)
verts, faces, normals, values = measure.marching_cubes(input, spacing=vox_size)

#function
norm_V,norm_S = surface_normals_np(verts,faces)

#Calculate triangle centre co-ordinates
centres = np.divide((verts[faces[:,0]]+ verts[faces[:,1]]+ verts[faces[:,2]]),3)

if __name__ == "__main__" :
    #Normalising vectors: Divide the vector by its magnitude
    #marching_cubes vertex vectors
    M_m = np.sqrt(normals[:,0]**2+normals[:,1]**2+normals[:,2]**2)
    unit_norms = normals/M_m.reshape(-1,1)

    #surface_normals_np vertex vectors
    M_v = np.sqrt(norm_V[:,0]**2+norm_V[:,1]**2+norm_V[:,2]**2)
    unit_V = norm_V/M_v.reshape(-1,1)

    #surface_normals_np surface vectors
    M_s = np.sqrt(norm_S[:,0]**2+norm_S[:,1]**2+norm_S[:,2]**2)
    unit_S = norm_S/M_s.reshape(-1,1)

    # METRIC COMPARISON between vertex vectors : COSINE SIMILARITY
    # Calculate the cosine similarity between the both unit vectors originating from the same vertex 
    # Dot product is 1 when both unit vectors are aligned as cos(0) = 1.
    # Returns the mean angle difference (Alpha) between the respective vertex vector comparisons.
    Diff_alpha = []

    # Loops through each vertex in verts.
    for i in range(len(verts)):

        alpha = np.dot(unit_norms[i],unit_V[i])
        Diff_alpha.append(alpha)

    Mean_alpha = np.mean(Diff_alpha)
    Alpha = np.arccos(Mean_alpha)
    Alpha = np.round(np.degrees(Alpha),2)
    print('The mean difference in angle is', Alpha,'°. This difference could be caused by the different methods of calculating the vertex vectors. However, this angle difference is insignificant.\n')

    # Comparison between outputs (vertex normals and surface normals) of surface_normal_np 
    # Dot product between the each vertex vector and its surface vector is calculated to compute the angle difference. 
    # Returns the mean angle difference (Theta) between each vertec vector and its respective surface vector.
    Diff_theta = []
    n=0

    #index vertices involved in each face
    for P1, P2, P3 in faces:
        
        vert_1, vert_2, vert_3 = unit_V[P1], unit_V[P2], unit_V[P3]
        
        #dot product between vertex normals and its respective surface normal
        theta = np.dot((vert_1,vert_2,vert_3),unit_S[n])
        Mean_theta = np.mean (theta)
        Diff_theta.append(Mean_theta)
        n +=1

    Mean_Theta = np.mean(Diff_theta)    
    Theta = np.arccos(Mean_Theta)
    Theta = np.round(np.degrees(Theta),2)

    print('The mean difference in angle between the surface vectors and vertex vectors is', Theta,'°. This difference is because the vertex vector is a sum of all surface vectors where the particular vertex is involved. However, this angle difference is insignificant.\n')

    #gaussian filters
    #set a range for sigma values
    range_s = np.linspace(3,8,11)
    #number of vertex normals computed from marching cubes
    #this represents the number of vertices present in filtered image.
    Vmc_val = []
    #number of vertices computed from 
    V_val = []
    #number of faces computed from
    #this represents the number of faces present in filtered image.
    S_val = []

    G_normal = []
    G_norm_v = []
    G_norm_s = []

    for n in range_s:
        G_normals, G_norm_V, G_norm_S = gaussian(input,n,vox_size)
        G_normal.append(G_normals)
        G_norm_v.append(G_norm_V)
        G_norm_s.append(G_norm_S)
        x,y = G_normals.shape
        m,n = G_norm_V.shape
        p,q = G_norm_S.shape
        Vmc_val.append(x)
        V_val.append(m)
        S_val.append(p)

    #find the minimum number of vertices as image is filtered with different values of sigma
    S_idx = V_val.index(min(V_val))
    #the minimum value is indexed to compute the value of sigma which allows maximum filtering.
    S = range_s[S_idx]

    Vmc_val = np.asarray(Vmc_val)
    V_val = np.asarray(V_val)
    S_val = np.asarray(S_val)

    #results are compiled into an array for comparison purposes.
    table = np.vstack((range_s, Vmc_val, V_val,S_val)).T
    title = ("sigma","no. of vertex normals from mc", "no. of vertex normals from np", "no. of surface normals from np")
    table = np.vstack((title,table))

    #print sigma value.
    print('As the value of sigma increases, the number of faces and vertices decreases. However, as sigma increases further (above',S,'), the number of faces and vertices increases.')
    print('The value of sigma that gives the least number of vertices and faces is',S,'.')

