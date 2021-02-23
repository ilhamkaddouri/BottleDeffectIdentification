# organize dataset into a useful structure
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random
# create directories
dataset_home = 'bottle/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['normals/', 'nolabel/','labelnotstraight/','overfilled/','underfilled/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)
# seed random number generator
seed(1)
# define ratio of pictures to use for validation
val_ratio = 0.3
# copy training dataset images into subdirectories
src_directory = 'bottles/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_ratio:
		dst_dir = 'test/'
	if file.startswith('normal'):
		dst = dataset_home + dst_dir + 'normals/'  + file
		copyfile(src, dst)
	elif file.startswith('nolabel'):
		dst = dataset_home + dst_dir + 'nolabel/'  + file
		copyfile(src, dst)
	elif file.startswith('labelnotstraight'):
		dst = dataset_home + dst_dir + 'labelnotstraight/'  + file
		copyfile(src, dst)
	elif file.startswith('overfilled'):
		dst = dataset_home + dst_dir + 'overfilled/'  + file
		copyfile(src, dst)
	elif file.startswith('underfilled'):
		dst = dataset_home + dst_dir + 'underfilled/'  + file
		copyfile(src, dst)
