from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

#opencv_inc_dir = '/data/tools/opencv/include/' # directory containing OpenCV header files
#opencv_lib_dir = '/usr/local/lib/' # directory containing OpenCV library files

if 'opencv_inc_dir' in os.environ.keys():
	opencv_inc_dir = os.environ['opencv_inc_dir'] #'' # directory containing OpenCV header files
else:
	opencv_inc_dir = ''

print('opencv inc dir', opencv_inc_dir)

if 'opencv_lib_dir' in os.environ.keys():
	opencv_lib_dir = os.environ['opencv_lib_dir'] #'' # directory containing OpenCV library files
else:
	opencv_lib_dir = ''

print('opencv lib dir', opencv_lib_dir)

#if not explicitly provided, we try to locate OpenCV in the current Conda environment
if 'CONDA_PREFIX' in os.environ.keys():
	conda_env = os.environ['CONDA_PREFIX'] #'' # directory containing OpenCV library files
else:
	conda_env = ''



if len(conda_env) > 0 and len(opencv_inc_dir) == 0 and len(opencv_lib_dir) == 0:
	print("Detected active conda environment:", conda_env)
	
	opencv_inc_dir = conda_env + '/include'
	opencv_lib_dir = conda_env + '/lib'

	print("Assuming OpenCV dependencies in:")
	print(opencv_inc_dir)
	print(opencv_lib_dir)

if len(opencv_inc_dir) == 0:
	print("Error: You have to provide an OpenCV include directory. Edit this file.")
	exit()
if len(opencv_lib_dir) == 0:
	print("Error: You have to provide an OpenCV library directory. Edit this file.")
	exit()

setup(
	name='ngransac',
	ext_modules=[CppExtension(
		name='ngransac', 
		sources=['ngransac.cpp','thread_rand.cpp'],
		#include_dirs=[opencv_inc_dir],
		#library_dirs=[opencv_lib_dir],
		libraries=['opencv_core','opencv_calib3d'],
		extra_compile_args=['-fopenmp']
		)],		
	cmdclass={'build_ext': BuildExtension})
