import cv2
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict
from tqdm import tqdm


def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        # black
        return 0, 0, 0, CMYK_SCALE

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    return c * CMYK_SCALE, m * CMYK_SCALE, y * CMYK_SCALE, k * CMYK_SCALE
# test = df['img'][6][300,:,:,:]/255

def rgb_to_cmyk(rgb):
	r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
	k = 1 - np.max(rgb, axis=-1)
	c = (1-r-k)/(1-k)
	m = (1-g-k)/(1-k)
	y = (1-b-k)/(1-k)
	return np.dstack([c, m, y, k])


def rgb_to_cmyk(rgb):
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    k = 1 - np.max(rgb, axis=-1)
    c = (1-r-k)/(1-k)
    m = (1-g-k)/(1-k)
    y = (1-b-k)/(1-k)
    return np.dstack([c, m, y, k])


def calculate_K_mean(vm):
    K_means = []
    for frame in range(vm.shape[0]):
        # Convert the frame to CMYK
        cmyk_image = rgb_to_cmyk(vm[frame, :, :, :]/255)
        
        # Extract the K channel
        K_channel = cmyk_image[:,:,3]
        
        # Calculate the 2.5th and 97.5th percentiles (95% CI)
        ci_low, ci_high = np.percentile(K_channel, [2.5, 97.5])
        
        # Mask the values outside the CI
        masked_K_channel = np.ma.masked_outside(K_channel, ci_low, ci_high)
        
        # Calculate the mean of the values within the CI
        K_mean = np.ma.mean(masked_K_channel)
        
        K_means.append(K_mean)
        
    return K_means


def apply_perspective_transform(frame, M, cols, rows):
    return cv2.warpPerspective(frame, M, (cols, rows))


def get_image_creation_date(image_path):
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag, value in exif_data.items():
                    tag_name = TAGS.get(tag, tag)
                    if tag_name == 'DateTimeOriginal':
                        return value
    except Exception as e:
        print(f"Error reading EXIF data for {image_path}: {e}")
    return None

def crop_image(img, crop_params):
    x0, x1 = crop_params[0]
    y0, y1 = crop_params[1]
    return img[x0:x1, y0:y1]

def create_video_matrix(folder_name, slice_dict, fids_0):
    
    # first_image = cv2.imread(fids_0[folder_name][0][0])
    # height, width, channels = first_image.shape

    # # Initialize the video matrix
    num_frames = len(fids_0[folder_name])
    video_matrix = None

    # Read each image, crop it, and populate the video matrix
    for i, info in enumerate(fids_0[folder_name]):

        image_path = info[0]
        frame = cv2.imread(image_path)
        

        if folder_name not in slice_dict:
            crop_params = slice_dict['113GOPRO']
        else:
            crop_params = slice_dict[folder_name]
        frame = crop_image(frame, crop_params)
        height, width, channels = frame.shape
        if video_matrix is None:
            video_matrix = np.zeros((num_frames, height, width, channels), dtype=np.uint8)

        video_matrix[i] = frame

    return video_matrix

def transform_vm(video_matrix):
    # Define the points in the original image (corners of the original image)
    transformed = np.zeros_like(video_matrix)
    rows, cols = video_matrix.shape[1:3]
    pts1 = np.float32([[cols//31, rows//8.75], [cols-cols//20, rows//35], [0, rows], [cols, rows-rows//4]])  # top left, top right, bottom left, bottom right
    #pts1 = np.float32([[20, 15], [cols-30, 5], [0, rows], [cols, rows-40]])  
    # Define the magnitude of the transformation (e.g., 0.05 for a 5% stretch at the top)
    magnitude = 0.01

    # Define where those points will be in the transformed image (stretching the top)
    pts2 = np.float32([[0, -rows * magnitude], [cols - 1, -rows * magnitude], [0, rows - 1], [cols - 1, rows - 1]])

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts1, pts2)

    # If the transformed image looks good, uncomment the following lines to apply the transformation to the whole video_matrix
    for i in tqdm(range(video_matrix.shape[0])):
        transformed[i, :, :, :] = apply_perspective_transform(video_matrix[i, :, :, :], M, cols, rows)
    
    return transformed

def temp_name(image_file_path):
    creation_time = []
    image_folder_0 = image_file_path
    go_pro_folders = os.listdir(image_folder_0)[::-1]
    fids_0 = {}
    for folder in go_pro_folders:
        if folder not in fids_0:
            fids_0[folder] = []
        folder_path = os.path.join(image_folder_0, folder)
        images = os.listdir(folder_path)
        for image in images:
            if 'control' not in image and '_rgb' not in image and 'ipynb' not in image:
                image_path = os.path.join(folder_path, image)
                creation_time = get_image_creation_date(image_path)
                fids_0[folder].append((image_path, creation_time))

    slice_dict = {}
    # Folder_name : ((x0,x1),(y0,y1))
    # try to get sizes to 175 x 620 pixels
    slice_dict['100GOPRO'] = ((1080,1255),(1390,2010))
    slice_dict['101GOPRO'] = ((1095,1280),(1365,1985))
    slice_dict['102GOPRO'] = ((980,1165),(1300,1920))
    slice_dict['103GOPRO'] = ((980,1165),(1300,1920))
    slice_dict['104GOPRO'] = ((980,1165),(1300,1920))
    slice_dict['105GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['106GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['107GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['108GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['109GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['110GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['111GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['112GOPRO'] = ((955,1140),(1280,1900))
    slice_dict['113GOPRO'] = ((955,1140),(1280,1900))
    
    temp_matrix = create_video_matrix(go_pro_folders[0], slice_dict = slice_dict, fids_0=fids_0)

    center_points_0 = np.full((4, 10), None, dtype=object)
    
    rows, cols = temp_matrix.shape[1:3]
    
    for i, y in enumerate(np.linspace(0+15,rows-rows//12,4)):
        for j, x in enumerate(np.linspace(0 + 30, cols-cols//21, 10)):
            center_points_0[i,j] = (int(x),int(y))
    
    sample_names_0 = np.arange(0,40).reshape(4,10)
    
    go_pro_folders = np.sort(np.array(go_pro_folders))
    
    window_size = 25
    samples_to_ignore = [17,22,38,39]
    used_sample_names = []
    for s in sample_names_0.flatten():
        if s not in samples_to_ignore:
            used_sample_names.append(s)

    c_data = pd.DataFrame(columns=used_sample_names)

    # Create a slice matrix to hold the slices
    slice_shape = (2 * window_size, 2 * window_size, temp_matrix.shape[3])
    slice_matrix = np.empty((*sample_names_0.shape, *slice_shape))

    K_means = defaultdict(list)

    K_means['Hour'] = []


    s_time = datetime.datetime.strptime(fids_0[go_pro_folders[0]][0][1], '%Y:%m:%d %H:%M:%S')
    
    # Extract the slices from the video_matrix using the center points
    for f in go_pro_folders:
        video_matrix = create_video_matrix(f, slice_dict, fids_0)
        video_matrix = transform_vm(video_matrix)
        num_frames = video_matrix.shape[0]
        for p in range(num_frames):
            picture_time = datetime.datetime.strptime(fids_0[f][p][1], '%Y:%m:%d %H:%M:%S')
            K_means['Hour'].append(np.round(((picture_time-s_time).total_seconds() / 60**2),3))
            for i in range(sample_names_0.shape[0]):
                for j in range(sample_names_0.shape[1]):
                    if sample_names_0[i][j] in samples_to_ignore:
                        continue
                    x, y = center_points_0[i, j]

                    # Boundary checks
                    x_min = max(0, x - window_size)
                    x_max = min(video_matrix.shape[2], x + window_size)
                    y_min = max(0, y - window_size)
                    y_max = min(video_matrix.shape[1], y + window_size)

                    temp_slice = rgb_to_cmyk(video_matrix[p, y_min:y_max, x_min:x_max, :])/255
                    # print(temp_slice)
                    K_channel = temp_slice[:,:,3]
                    # Calculate the 2.5th and 97.5th percentiles (95% CI)
                    ci_low, ci_high = np.percentile(K_channel, [2.5, 97.5])

                    # Mask the values outside the CI
                    masked_K_channel = np.ma.masked_outside(K_channel, ci_low, ci_high)
                    # Calculate the mean of the values within the CI
                    K_mean = np.ma.mean(masked_K_channel)


                    # Padding in case the slice is smaller than the window
                    pad_x_min = window_size - (x - x_min)
                    pad_x_max = window_size + (x_max - x)
                    pad_y_min = window_size - (y - y_min)
                    pad_y_max = window_size + (y_max - y)


                    K_means[sample_names_0[i][j]].append(1-K_mean)
                    # slice_matrix[i, j, pad_y_min:pad_y_max, pad_x_min:pad_x_max, :] = temp_slice
    return pd.DataFrame(K_means)



c_data = temp_name('gopro_images/')