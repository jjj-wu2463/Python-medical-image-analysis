# Python-medical-image-analysis


## Euclidean-distance
This file compares the speed and accuracy of calcuating the Euclidean distance within a 3-Dimensional binary image manually (using only Numpy) and via the built-in function "distance_transform_edt".

## SurfaceVertices
Takes the output from "marching_cubes" to manually compute (using Numpy) the surface normals and vertices of a 3D image.
These results are compared to the output of the built-in function "measure.marching_cubes".
The dot product of each corresponding vertex vector is calculated to compare the difference in angle between the two vectors. When two vectors are parallel, there is no angle difference. 

## Transformation
This implements a Class that takes 3D medical images as an input and performs affine and rigid transformation. A random transformation can be generated when there are no inputs for any type of transformation.

## Filters
This file compares the efficiency of 2D and 3D filtering. 2D filtering can be done by applying suitable filters after slicing the image, while 3D filtering filters the image prior to slicing.  Salt and pepper (S&P) noise are added to the 3D image, then filtered via 2D and 3D.
The efficiency of these 2 methods are compared through measuring the Structura Similarity Index Measurement(SSIM) and peak-signalg-to-noise ratio (PSNR) of the filtered images to the original image. 
Due to time constraints, more comparison metrics could be applied and experimented.
