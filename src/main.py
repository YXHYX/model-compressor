#MODEL COMPRESSOR USING PYTHOON

"""
Im trying to build a 3D model compressor using python numpy and for visualization mayavi. The idea is based on taking a closed surface 3D object,
 then creating a function which maps each point on the sphere with a complex number which the norm of that number corresponds to the distance from
  the sphere center to the points on the model closest to the line passing by the sphere's point and center. Then decomposing this functions using 
  Fourier series and spherical harmonics to obtain the coefficients of the series. The compression happens by choosing the degree (N) of the series,
   thereby cutting away certain information while still retaining the shape's form. Also by using inverse mapping to re obtain the original shape 
   ( we decomposed the mapping function ). 

"""
import core

shape = CompressedShape()