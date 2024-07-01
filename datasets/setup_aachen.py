# Based on https://github.com/vislearn/esac/blob/master/datasets/setup_aachen.py

# The images of the Aachen Day-Night dataset are licensed under a 
# [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and are intended for non-commercial academic use only. 
# All other data also provided as part of the Aachen Day-Night dataset, including the 3D model 
# and the camera calibrations, is derived from these images. Consequently, all other data is 
# also licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) and intended for non-commercial academic use only.

import os
import gdown
import zipfile
from urllib.request import urlretrieve

import torch

import math

import numpy as np

import cv2 as cv

# name of the folder where we download the original aachen dataset to
src_folder = 'aachen_source'

# destination folder that will contain the dataset in our format
dst_folder = 'aachen'

# source files and folders
image_file = 'database_and_query_images.zip' # https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/images/database_and_query_images.zip
image_folder = 'images_upright/'
recon_file = 'aachen_cvpr2018_db.nvm' # https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/3D-models/aachen_cvpr2018_db.nvm
db_file = 'aachen_cvpr2018_db.list.txt' # https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/3D-models/aachen_cvpr2018_db.list.txt
day_file = 'day_time_queries_with_intrinsics.txt' # https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/queries/day_time_queries_with_intrinsics.txt
night_file = 'night_time_queries_with_intrinsics.txt'# https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/queries/night_time_queries_with_intrinsics.txt

def mkdir(directory):
	"""Checks whether the directory exists and creates it if necessacy."""
	if not os.path.exists(directory):
		os.makedirs(directory)

mkdir(dst_folder)
mkdir(src_folder)
os.chdir(src_folder)

print("\n###############################################################################")
print("# Please make sure to check this dataset's license before using it!           #")
print("# https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/README_Aachen-Day-Night.md #")
print("###############################################################################\n\n")

license_response = input('Please confirm with "yes" or abort. ')
if not (license_response == "yes" or license_response == "y"):
	print(f"Your response: {license_response}. Aborting.")
	exit()


# download the aachen dataset
print("=== Downloading Aachen Data ===============================")

def dl_file(urls, file):
	if os.path.isfile(file):
			return True
	for url in urls:
		print(f"Try downloading to {file} from {url}")
		if "drive.google.com" in url:
			try:
				gdown.download(url, file, quiet=False)
			except gdown.exceptions.FileURLRetrievalError as e:
				print(f"Failed to download {file}: {e}")
				continue
		else:
			urlretrieve(url, file)
		if os.path.isfile(file):
			return True
	print(f"Failed to download {file}")
	exit(1)

dl_file(["https://drive.google.com/uc?id=18vz-nCBFhyxiX7s-zynGczvtPeilLBRB",
		 "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/images/database_and_query_images.zip"], image_file)
dl_file(["https://drive.google.com/uc?id=0B7s5ESv70mytamRSY0J1dWs4aE0",
		 "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/3D-models/aachen_cvpr2018_db.nvm"], recon_file)
dl_file(["https://drive.google.com/uc?id=0B7s5ESv70mytQldWbEdrODBlOFE",
		 "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/3D-models/aachen_cvpr2018_db.list.txt"], db_file)
dl_file(["https://drive.google.com/uc?id=0B7s5ESv70mytQS1MSmlIVVZzaGM",
		 "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/queries/day_time_queries_with_intrinsics.txt"], day_file)
dl_file(["https://drive.google.com/uc?id=0B7s5ESv70mytTWZmTFoxUkNYZW8",
		 "https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/queries/night_time_queries_with_intrinsics.txt"], night_file)
	
# unpack and delete image zip file
f = zipfile.PyZipFile(image_file)
f.extractall()
os.system('rm ' + image_file)

def create_training_set(image_file):

	variant = 'train'

	mkdir('../' + dst_folder + '/' + variant + '/rgb')
	mkdir('../' + dst_folder + '/' + variant + '/calibration')
	mkdir('../' + dst_folder + '/' + variant + '/poses')
	mkdir('../' + dst_folder + '/' + variant + '/init')

	f = open(recon_file)
	reconstruction = f.readlines()
	f.close()

	f = open(image_file)
	camera_list = f.readlines()
	f.close()

	image_list = [camera.split()[0] for camera in camera_list]

	num_cams = int(reconstruction[2])
	num_pts = int(reconstruction[num_cams + 4])

	print("Num Cams:", num_cams)
	print("Num Pts:", num_pts)

	camera_list = reconstruction[3:3+num_cams]
	camera_list = [c.split() for c in camera_list]

	# read points
	pts_dict = {}
	for cam_idx in range(0, num_cams):
		pts_dict[cam_idx] = []

	pt = pts_start = num_cams + 5
	pts_end = pts_start + num_pts

	while pt < pts_end:

		pt_list = reconstruction[pt].split()
		pt_3D = [float(x) for x in pt_list[0:3]]
		pt_3D.append(1.0)

		for pt_view in range(0, int(pt_list[6])):
			cam_view = int(pt_list[7 + pt_view * 4])
			pts_dict[cam_view].append(pt_3D)
		
		pt += 1

	for cam_idx in range(0, num_cams):

		print("Processing camera %d of %d." % (cam_idx, num_cams))
		image_file = reconstruction[3 + cam_idx].split()[0]

		if image_file not in image_list:
			print("Skipping image " + image_file + ". Not part of set: " + mode + ".")
			continue

		# read cameras
		image_idx = image_list.index(image_file)
		camera = camera_list[image_idx]
		cam_rot = [float(r) for r in camera[2:6]]
		focal_length = float(camera[1])

		#quaternion to axis-angle
		angle = 2 * math.acos(cam_rot[0])
		x = cam_rot[1] / math.sqrt(1 - cam_rot[0]**2)
		y = cam_rot[2] / math.sqrt(1 - cam_rot[0]**2)
		z = cam_rot[3] / math.sqrt(1 - cam_rot[0]**2)

		cam_rot = [x * angle, y * angle, z * angle]
		
		cam_rot = np.asarray(cam_rot)
		cam_rot, _ = cv.Rodrigues(cam_rot)

		cam_trans = [float(r) for r in camera[6:9]]
		cam_trans = np.asarray([cam_trans])
		cam_trans = np.transpose(cam_trans)
		cam_trans = - np.matmul(cam_rot, cam_trans)

		cam_pose = np.concatenate((cam_rot, cam_trans), axis = 1)
		cam_pose = np.concatenate((cam_pose, [[0, 0, 0, 1]]), axis = 0)

		cam_pose = torch.tensor(cam_pose).float()

		#load image
		image = cv.imread(image_folder + image_file)
		image_file = image_file.replace('/', '_')

		pts_3D = torch.tensor(pts_dict[cam_idx])

		img_aspect = image.shape[0] / image.shape[1]

		target_height = 480 # shorter image side re-scaled to this value
		nn_subsampling = 8 # scene coordinate output sub-sampled by this factor wrt. the image 

		if  img_aspect > 1:
			#portrait
			img_w = target_height
			img_h = int(math.ceil(target_height * img_aspect))	
		else:
			#landscape
			img_w = int(math.ceil(target_height / img_aspect))
			img_h = target_height

		out_w = int(math.ceil(img_w / nn_subsampling))
		out_h = int(math.ceil(img_h / nn_subsampling))

		out_scale = out_w / image.shape[1]
		img_scale = img_w / image.shape[1]

		out_tensor = torch.zeros((3, out_h, out_w))
		out_zbuffer = torch.zeros((out_h, out_w))

		camMat = np.eye(3)
		camMat[0, 0] = focal_length
		camMat[1, 1] = focal_length
		camMat[0, 2] = image.shape[1] / 2
		camMat[1, 2] = image.shape[0] / 2

		dist = np.asarray([-float(camera[9]), 0, 0, 0])

		image = cv.undistort(image, camMat, dist)
		image = cv.resize(image, (img_w, img_h))
		cv.imwrite('../' + dst_folder + '/' + variant + '/rgb/' + image_file, image)

		with open('../' + dst_folder + '/' + variant + '/calibration/' + image_file[:-3] + 'txt', 'w') as f:
			f.write(str(focal_length * img_scale))

		for pt_idx in range(0, pts_3D.size(0)):

			scene_pt = pts_3D[pt_idx]
			scene_pt = scene_pt.unsqueeze(0)
			scene_pt = scene_pt.transpose(0, 1)

			cam_pt = torch.mm(cam_pose, scene_pt)

			img_pt = cam_pt[0:2, 0] * focal_length / cam_pt[2, 0] * out_scale
					
			y = img_pt[1] + out_h / 2
			x = img_pt[0] + out_w / 2

			x = int(torch.clamp(x, min=0, max=out_tensor.size(2)-1))
			y = int(torch.clamp(y, min=0, max=out_tensor.size(1)-1))

			if out_zbuffer[y, x] == 0 or out_zbuffer[y, x] > cam_pt[2, 0]:
				out_zbuffer[y, x] = cam_pt[2, 0]
				out_tensor[:, y, x] = pts_3D[pt_idx, 0:3]

		torch.save(out_tensor, '../' + dst_folder + '/' + variant + '/init/' + image_file[:-4] + '.dat')

		cam_pose = cam_pose.inverse()

		with open('../' + dst_folder + '/' + variant + '/poses/' + image_file[:-3] + 'txt', 'w') as f:
			f.write(str(float(cam_pose[0, 0])) + ' ' + str(float(cam_pose[0, 1])) + ' ' + str(float(cam_pose[0, 2])) + ' ' + str(float(cam_pose[0, 3])) + '\n')
			f.write(str(float(cam_pose[1, 0])) + ' ' + str(float(cam_pose[1, 1])) + ' ' + str(float(cam_pose[1, 2])) + ' ' + str(float(cam_pose[1, 3])) + '\n')
			f.write(str(float(cam_pose[2, 0])) + ' ' + str(float(cam_pose[2, 1])) + ' ' + str(float(cam_pose[2, 2])) + ' ' + str(float(cam_pose[2, 3])) + '\n')
			f.write(str(float(cam_pose[3, 0])) + ' ' + str(float(cam_pose[3, 1])) + ' ' + str(float(cam_pose[3, 2])) + ' ' + str(float(cam_pose[3, 3])) + '\n')

def create_test_set(image_file):

	variant = 'test'

	mkdir('../' + dst_folder + '/' + variant + '/rgb')
	mkdir('../' + dst_folder + '/' + variant + '/calibration')
	mkdir('../' + dst_folder + '/' + variant + '/poses')

	f = open(image_file)
	camera_list = f.readlines()
	f.close()

	image_list = [camera.split()[0] for camera in camera_list]
	camera_list = [c.split() for c in camera_list]


	print("Num Cams: ", len(image_list))


	for cam_idx in range(0, len(image_list)):

		print("Processing camera %d of %d." % (cam_idx, len(image_list)))
		image_file = image_list[cam_idx]
		camera = camera_list[cam_idx]
		
		#load image
		image = cv.imread(image_folder + image_file)
		image_file = image_file.replace('/', '_')

		# read cameras
		focal_length = float(camera[4])

		cam_pose = torch.eye(4).float()

		target_height = 480
		nn_subsampling = 8

		img_aspect = image.shape[0] / image.shape[1]

		if  img_aspect > 1:
			#portrait
			img_w = target_height
			img_h = int(math.ceil(target_height * img_aspect))	
		else:
			#landscape
			img_w = int(math.ceil(target_height / img_aspect))
			img_h = target_height

		out_w = int(math.ceil(img_w / nn_subsampling))
		out_h = int(math.ceil(img_h / nn_subsampling))

		out_scale = out_w / image.shape[1]
		img_scale = img_w / image.shape[1]


		camMat = np.eye(3)
		camMat[0, 0] = focal_length
		camMat[1, 1] = focal_length
		camMat[0, 2] = image.shape[1] / 2
		camMat[1, 2] = image.shape[0] / 2

		dist = np.asarray([float(camera[7]), 0, 0, 0])
		image = cv.undistort(image, camMat, dist)

		image = cv.resize(image, (img_w, img_h))
		cv.imwrite('../' + dst_folder + '/' + variant + '/rgb/' + image_file, image)

		with open('../' + dst_folder + '/' + variant + '/calibration/' + image_file[:-3] + 'txt', 'w') as f:
			f.write(str(focal_length * img_scale))

		with open('../' + dst_folder + '/' + variant + '/poses/' + image_file[:-3] + 'txt', 'w') as f:
			f.write(str(float(cam_pose[0, 0])) + ' ' + str(float(cam_pose[0, 1])) + ' ' + str(float(cam_pose[0, 2])) + ' ' + str(float(cam_pose[0, 3])) + '\n')
			f.write(str(float(cam_pose[1, 0])) + ' ' + str(float(cam_pose[1, 1])) + ' ' + str(float(cam_pose[1, 2])) + ' ' + str(float(cam_pose[1, 3])) + '\n')
			f.write(str(float(cam_pose[2, 0])) + ' ' + str(float(cam_pose[2, 1])) + ' ' + str(float(cam_pose[2, 2])) + ' ' + str(float(cam_pose[2, 3])) + '\n')
			f.write(str(float(cam_pose[3, 0])) + ' ' + str(float(cam_pose[3, 1])) + ' ' + str(float(cam_pose[3, 2])) + ' ' + str(float(cam_pose[3, 3])) + '\n')



print("=== Processing Dataset Images ===============================")
create_training_set(db_file)

print("=== Processing Day Queries ===============================")
create_test_set(day_file)

print("=== Processing Night Queries ===============================")
create_test_set(night_file)
