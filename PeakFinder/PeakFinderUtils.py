from __future__ import division
import lxml
from lxml import etree
import pykml
from pykml import parser
from glob import glob
from bs4 import BeautifulSoup
from tqdm import tqdm

import exifread
import pandas as pd

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from IPython.display import display, HTML, Javascript, clear_output
import ipywidgets as widgets
import IPython
from io import BytesIO

import re
import json
import urllib
from urllib.request import Request, urlopen
import requests
from typing import Union
from haversine import haversine, Unit

import matplotlib.pyplot as plt
import numpy as np
import cv2

import operator
from geopy import distance as geopy_distance
import overpass
import geocoder
import geopy
from geopy.distance import VincentyDistance

import numpy as np
import matplotlib.mlab as ml
from mpl_toolkits.mplot3d.axes3d import *
import matplotlib.pyplot as plt
from matplotlib import cm

import pyproj
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
from scipy.interpolate import griddata


import random
import pprint
import sys
import time
import numpy as np
import pickle
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras import backend as K
from keras.layers import Input
from keras.models import Model
sys.path.append('/content/drive/MyDrive/CV Final Project/Keras-FasterRCNN')
from keras_frcnn import config
import keras_frcnn.roi_helpers as roi_helpers
import warnings
warnings.filterwarnings("ignore")
from joblib import load, dump

import PIL.Image
from io import BytesIO
import cv2
from tqdm.auto import tqdm, trange

# Set learning phase to 0 for model.predict. Set to 1 for training
K.set_learning_phase(0)
C = config.Config()
C.model_path ='/content/drive/MyDrive/CV Final Project/models/resnet/resnet516_300roi_best_loss.ckpt'
C.network = 'resnet50'
num_rois = 300
C.im_size = 516
C.num_rois = int(num_rois)

if C.network == 'resnet50': import keras_frcnn.resnet as nn
elif C.network == 'vgg': import keras_frcnn.vgg as nn

try:   
  class_mapping = load('/content/drive/MyDrive/CV Final Project/data/dumped_data_gen_vgg516/class_mapping.pkl')
  print('--- LOADED DATA DICTIONARIES ---')
except: print('--- FAILED TO LOAD DATABASE FILES ---')

if 'bg' not in class_mapping: class_mapping['bg'] = len(class_mapping)
class_mapping = {v: k for k, v in class_mapping.items()}
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

if C.network == 'resnet50': num_features = 1024
elif C.network == 'vgg': num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet)
shared_layers = nn.nn_base(img_input, trainable=True)
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)
classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)
model_rpn = Model(img_input, rpn_layers)
model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)
model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')
classes = {}
clear_output()


def get_bboxes_in_photo_overlap(filepath, bbox_threshold=0.8, 
                                background_threshold=0.95,
                                rpn_overlap_threshold=0.9,
                                classifier_overlap_threshold=0.5):
    if not filepath.lower().endswith(('.bmp', '.jpeg', 
                                      '.jpg', '.png', 
                                      '.tif', '.tiff')): return False
    img = cv2.imread(filepath)
    X, ratio = format_img(img, C)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=rpn_overlap_threshold)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    possible_peaks = []
    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0: break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier.predict([F, ROIs])
        for ii in range(P_cls.shape[1]):
            # Just mountains/ No background
            # if np.max(P_cls[0, ii, :]) > background_threshold and not np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            #     continue
                
            if np.max(P_cls[0, ii, :]) < bbox_threshold:continue
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
            if cls_name == 'bg' and np.max(P_cls[0, ii, :]) > background_threshold: continue
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    peak_bboxes = []
    for key in bboxes:
        if key == 'mountain': 
            print("\n=== Peak Detected ===\n")
            
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=classifier_overlap_threshold)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            peak_bboxes.append([real_x1, real_y1, real_x2, real_y2])
            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    return all_dets, peak_bboxes
    
    
def get_bboxes_in_photo(filepath, bbox_threshold=0.8, background_threshold=0.95):
    if not filepath.lower().endswith(('.bmp', '.jpeg', 
                                      '.jpg', '.png', 
                                      '.tif', '.tiff')): return False
    img = cv2.imread(filepath)
    X, ratio = format_img(img, C)
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.999)

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    possible_peaks = []
    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
        if ROIs.shape[1] == 0: break

        if jk == R.shape[0]//C.num_rois:
            #pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier.predict([F, ROIs])
        for ii in range(P_cls.shape[1]):
            # Just mountains/ No background
            # if np.max(P_cls[0, ii, :]) > background_threshold and not np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            #     continue
                
            if np.max(P_cls[0, ii, :]) < bbox_threshold:continue
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
            if cls_name == 'bg' and np.max(P_cls[0, ii, :]) > background_threshold: continue
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except:
                pass
            bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []
    peak_bboxes = []
    for key in bboxes:
        if key == 'mountain': 
            print("\n=== Peak Detected ===\n")
            
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.001)
        for jk in range(new_boxes.shape[0]):
            (x1, y1, x2, y2) = new_boxes[jk,:]

            (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
            peak_bboxes.append([real_x1, real_y1, real_x2, real_y2])
            cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

            textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
            all_dets.append((key,100*new_probs[jk]))

            (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
            textOrg = (real_x1, real_y1-0)

            cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
            cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
            cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

    return all_dets, peak_bboxes


#Use 'jpeg' instead of 'png' (~5 times faster)
def imdisplay(img, fmt='jpeg',width=500):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_p = PIL.Image.fromarray(img)    
    f = BytesIO()
    if new_p.mode != 'RGB':
        new_p = new_p.convert('RGB')
    new_p.save(f, fmt)
    return IPython.display.Image(data=f.getvalue(), width=width)




def preventScrolling():
    disable_js = """
    IPython.OutputArea.prototype._should_scroll = function(lines) {
        return false;
    }
    """
    display(Javascript(disable_js))

import warnings
warnings.filterwarnings("ignore")

clear_output()

def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width ,_) = img.shape
    if width <= height:
        ratio = img_min_side/width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side/height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))
    return (real_x1, real_y1, real_x2 ,real_y2)


# Takes lat, long, height off the ground, and a name
class Location:

    def __init__(self, latitude, longitude, height_off_ground=6, name="", 
                 get_elevation=False, 
                 GOOGLE_API_KEY='AIzaSyDFH9DM-lNJai_3bpepD1YIAjzCOzu_Rw0'):
        self.latitude = latitude
        self.longitude = longitude
        self.height = height_off_ground
        self.name = name
        self.GOOGLE_API_KEY = GOOGLE_API_KEY
        self.coordinates_lat_long_height = self.latitude, self.longitude, self.height
        self.coordinates_lat_long_as_string = str(self.latitude) + ',' + str(self.longitude)
        self.coordinates_long_lat_height = self.longitude, self.latitude, self.height
        self.coordinates_lat_long = self.latitude, self.longitude
        if get_elevation: self.elevation = self.set_elevation_google()

    # Set in Meters
    def set_elevation_google(self):
        url = f'''https://maps.googleapis.com/maps/api/elevation/json?locations={self.latitude},{self.longitude}&key={self.GOOGLE_API_KEY}'''
        response = urlopen(Request(url, headers={'Content-Type': 'application/json'}))
        json_response = json.loads(response.read().decode("utf8"))
        response.close()
        self.elevation = self.process_elevation_response(json_response)
        return self.elevation

    def process_elevation_response(self, received_request):
        data = float(received_request['results'][0]['elevation'])
        return np.round(data,15)

class Photo:
    
    def __init__(self, path_to_photo, height_off_ground=25):
        self.filepath = path_to_photo
        if self.filepath.find("https://") != -1:
            self.image  = np.asarray(bytearray(urlopen(self.filepath).read()), dtype="uint8")
            self.cv_image = get_CV_image()
        else:
            self.image = Image.open(path_to_photo)
            self.cv_image = cv2.imread(path_to_photo)

        self.set_exif()
        self.set_AOV()
        self.height_off_ground = height_off_ground
        self.location = Location(self.latitude, self.longitude, self.height_off_ground, self.filepath)
        self.set_intrinsic_cam_matrix()
        self.set_homography()
        self.set_extrinstic_cam_matrix()

    def __repr__(self):
        return f"""<class Photo #id: {self.id}; #path: {self.filepath}>"""


    def __str__(self):
        return f''' 
        \nPhoto Details:
        \t\tFilepath:\t{self.filepath}
        \t\tWidth:\t{self.width}px
        \t\tHeight:\t{self.height}\n
        GPS Data:
        \t\tLongitutde:\t{self.longitude}
        \t\tLatitude:\t{self.latitude}
        \t\tAltitude:\t{self.altitude}m
        \t\tHeading:\t{self.heading}Â°\n
        Image Details:
        \tFocal Length:\t{self.focal_length}mm
        \tImage Sensor Width:\t{self.x_resolution}px
        \tImage Sensor Height:\t{self.y_resolution}px
        \t\tAOV:\t{self.aov}Â°
        '''


    def set_Location_height(self, height):
        self.location.height = height


    def get_CV_image(self):
        image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
        return image
    

    #Use 'jpeg' instead of 'png' (~5 times faster)
    def imdisplay(self, img, format='jpeg', width=500):
        new_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
        if new_img.mode != 'RGB': new_img = new_img.convert('RGB')
        new_img.save(BytesIO(), format)
        return display.Image(data=imageIO.getvalue(), width=width)


    def display(self, width=800):
        display_ratio = width/self.width
        img = Image.fromarray((cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2RGB)).astype(np.uint8))
        image_to_display = img.resize((int(width), int(self.height*display_ratio)))
        display(image_to_display)


    def set_exif(self):
        try:
            data = exifread.process_file(open(self.filepath, 'rb'))
            latitudes = data["GPS GPSLatitude"].values
            self.latitude = np.round(float(latitudes[0]) + float(latitudes[1])/60 + float(latitudes[2])/3600, 15)
            longitudes = data["GPS GPSLongitude"].values
            self.longitude = np.round(float(longitudes[0]) + float(longitudes[1])/60 + float(longitudes[2])/3600,15)
            self.latitude *= (-1 if data["GPS GPSLatitudeRef"].values[0] in ['S','W'] else 1)
            self.longitude *= (-1 if data["GPS GPSLongitudeRef"].values[0] in ['S','W'] else 1)
            altitude = str(data["GPS GPSAltitude"].values[0]).split("/")
            self.altitude = np.round(float(int(altitude[0])/int(altitude[1])), 15)
            self.height = data["EXIF ExifImageLength"].values[0]
            self.width = data["EXIF ExifImageWidth"].values[0]
            self.x_resolution = int(data["Image XResolution"].values[0])
            self.y_resolution = int(data["Image XResolution"].values[0])
            focal_lengths = str(data["EXIF FocalLength"].values[0]).split("/")
            self.focal_length = float(int(focal_lengths[0])/int(focal_lengths[1]))*0.001
            heading = str(data["GPS GPSImgDirection"].values[0]).split("/")
            self.heading = np.round(float(int(heading[0])/int(heading[1])),15)
        except: print("Didn't find all needed EXIF information")

    #TODO: Change to y_resolution if orientation is vertical
    def set_AOV(self):
        self.aov = np.arctan((self.x_resolution/2)/self.focal_length)*180/np.pi
        
    def set_intrinsic_cam_matrix(self):
        self.K = np.array([[self.focal_length/self.x_resolution, 0, self.width/2], 
                            [0, self.focal_length/self.y_resolution, self.height/2], 
                             [0, 0, 1]])
        
    def set_homography(self):
        self.H_c_w, self.H_w_c = getHomoTransforms(getRotationMatrix(ax=0,ay=0,az=self.heading+180,in_rad=False,order="ZYX"), np.array([(latlon_to_xyz(self.location))]).T)

    def set_extrinstic_cam_matrix(self):
        self.Mext = self.H_w_c[0:3, :]

    def get_xy_coord(self, location):
        point = np.append(latlon_to_xyz(location), 1.0)
        new_point = self.K @ self.Mext @ point
        new_point = new_point/new_point[2] # normalizing the points
        return new_point[0], new_point[1]
        
    def draw_marker(self, x, y):
        cv2.drawMarker(self.cv_image, (int(x),int(y)), color=(0,255,0), markerType=cv2.MARKER_CROSS, thickness=50)
        
    def draw_bbox(self, bbox, color=(0,255,0)):
        self.cv_image = cv2.rectangle(self.cv_image,(bbox[0], bbox[1]), (bbox[2], bbox[3]), color, int(self.width/100))


class Peak:

    def __init__(self, name, latitude, longitude, elevation=0):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.set_Location()

    def __repr__(self):
        return f"""<class Peak #id:{self.name}>"""

    def __str__(self):
        description = f'''\n{self.name}
        \tLatitude: {self.latitude}
        \tLongitude: {self.longitude}
        \tElevation: {self.elevation}m'''
        return description
    
    def set_Location(self):
        self.location = Location(self.latitude, self.longitude, 6, self.name) 
        if self.elevation == 0: self.elevation = self.location.set_elevation_google()
        return self.location

    def display(self):
        if self.ref_img is not None:
            display(Image.open(BytesIO(requests.get(self.ref_img).content)))
            
class MountainDatabase():
    
    def __init__(self, csv_file_path):
        self.load_peaks_from_csv(csv_file_path)


    def load_peaks_from_csv(self, csv_file_path='extract_mountain_data.csv'):
        self.df = pd.read_csv('extract_mountain_data.csv')
        self.mountains = []
        for index, row in self.df.iterrows():
            self.mountains.append(Peak(row["Name"],row["Latitude"],row["Longitude"], row["Elevation"]))


    def get_peaks_in_photo(self, photo: Photo, within_distance=50, error_threshold_distance=100):
        peaks = self.filter_peaks_aov(photo.aov, photo.heading, photo.location, 
                                self.query_peaks_by_distance(photo.location, within_distance, "miles"), 
                                error_threshold_distance=error_threshold_distance)
        if len(peaks) == 0: peaks = self.filter_peaks_aov(photo.aov, photo.heading, photo.location,
                                                         self.get_peaks_in_photo_OVERPASS(photo, max_range=within_distance, num_results=1000),
                                                         error_threshold_distance=error_threshold_distance)
        return peaks

    def get_peaks_in_photo_OVERPASS(self, photo: Photo, min_range=1, max_range=100, range_inc=10, num_results=50, verbose=0):
        OVERPASS_API = overpass.API()
        latitude = photo.location.latitude
        longitude = photo.location.longitude
        RANGE = max_range*1609.344
        overpass_query = f'''node(around:{RANGE},{latitude},{longitude})[natural=peak];'''
        results = OVERPASS_API.get(overpass_query)
        return self.create_nearby_peaks(format_peak_results(results, (latitude, longitude)), max_range, num_results)


    def format_peak_results(self, results, current_location):
        formatted_results={}
        for feature in results['features']:
            try:
                coordinates = tuple(reversed(feature["geometry"]["coordinates"]))
                peak_name = feature['properties']['name']
                distance = geopy_distance.geodesic(current_location, coordinates).miles
                formatted_results[distance] = {"name": peak_name, 
                                                "coordinates": coordinates, 
                                                "id": id}
            except KeyError: pass
        return formatted_results
    

    def create_nearby_peaks(self, results, distance=80, num_results=1000, printer=False):
        peaks = []
        for i, peak in enumerate(sorted(results.items(), key=operator.itemgetter(0)),1):
            peaks.append(Peak(peak[1]['name'], peak[1]['coordinates'][0], peak[1]['coordinates'][1]))
            if i == num_results:
                return peaks
        return peaks

    # Find corners by changing bearing and calculating new lat long edge
    # Takes a Location Object and distance in kilometers
    def query_peaks_by_distance(self, location: Location, distance=20, UNITS="miles"):
        origin = geopy.Point(location.latitude, location.longitude)
        latitude_min = 0
        latitude_max = 0
        longitude_min = 0
        latitude_max = 0
        for bearing in [0, 90, 180, 270]:
            if UNITS == "miles": destination = VincentyDistance(miles=distance).destination(origin, bearing)
            if UNITS == "kilometers": destination = VincentyDistance(kilometers=distance).destination(origin, bearing)
            elif UNITS == "meters": destination = VincentyDistance(meters=distance).destination(origin, bearing)
            else: destination = VincentyDistance(miles=distance).destination(origin, bearing)
            lat2, lon2 = destination.latitude, destination.longitude
            if bearing == 0: latitude_max = lat2
            if bearing == 180: latitude_min = lat2
            if bearing == 90: longitude_max = lon2
            if bearing == 270: longitude_min = lon2
        return self.df.loc[(self.df['Latitude'] >= latitude_min) & 
                           (self.df['Latitude'] <= latitude_max) & 
                           (self.df['Longitude'] >= longitude_min) & 
                           (self.df['Longitude'] <= longitude_max)]

    def in_aov(self, origin: Location, dest: Location, aov, heading, error_threshold=0.2):
        aov = np.deg2rad(aov)/2
        heading = np.deg2rad(heading+180)
        origin = np.array([float(origin.latitude), float(origin.longitude)])
        peak = np.array([float(dest.latitude), float(dest.longitude)])
        theta = np.arccos(origin*(origin-peak))
        # if np.tan(np.abs(origin*(origin-peak)))[1] + error_threshold < 0: return False
 
        if np.cos(origin*(origin-peak))[1] + error_threshold < 0: return False
        if theta[0] < (heading+aov)/2 and theta[0] > (heading-aov)/2: return True
        else: return False

    def filter_peaks_aov(self, aov, heading, origin, df, error_threshold_distance=250):
        mountains=[]
        try:
            for index, row in df.iterrows():
                mountains.append(Peak(row["Name"],row["Latitude"],row["Longitude"], row["Elevation"]))
            in_view = []
            for mountain in mountains:
                if self.in_aov(origin, mountain, aov, heading) and self.check_line_of_sight(origin, mountain.location, error_threshold_distance):
                    in_view.append(mountain)
            return in_view
        except: return mountains


    def in_line_of_sight(self, x_values,elevation_data, error_threshold=500):
        i = 0
        intercept = True
        while i < len(x_values):
            if x_values[i] + error_threshold < elevation_data[i]:
                intercept = False
                break
            else: i += 1
        return intercept

    def create_los_path(self, x_values, y_start, y_end):
        return np.linspace(y_start, y_end, len(x_values))


    def select_color(self, los_y, terrain_y):
        if self.in_line_of_sight(los_y, terrain_y): return 'green'
        else: return 'red'


    def check_line_of_sight(self, photo: Location, peak: Location, error_threshold_distance, plot=False):
        GOOGLE_API_KEY = 'AIzaSyDFH9DM-lNJai_3bpepD1YIAjzCOzu_Rw0' # For ar-peak-finder Project
        NUMBER_SAMPLES = 500
        DISTANCE_UNITS = "Kilometres" # 'Kilometres', 'Feet', 'Nautical miles'
        HEIGHT_UNITS = "Metres" # 'Kilometres', 'Metres', 'Nautical miles'


        great_circle_distance = calculate_great_circle_distance((float(photo.latitude), 
                                                                float(photo.longitude)), 
                                                                (float(peak.latitude), 
                                                                float(peak.longitude)), 
                                                                DISTANCE_UNITS)
        
        EARTH = Circle(define_earth_radius(DISTANCE_UNITS), great_circle_distance)
        angle_list = np.linspace(EARTH.calc_start_angle(0, 0), EARTH.calc_end_angle(0, great_circle_distance), NUMBER_SAMPLES)
        distance_x_values = np.linspace(0, great_circle_distance, NUMBER_SAMPLES)
        earth_surface_y_values = EARTH.calculate_earth_surface_y_values(distance_x_values, 
                                                                        angle_list, 
                                                                        HEIGHT_UNITS, 
                                                                        DISTANCE_UNITS)

        elevation_data, lat_data, long_data = send_and_receive_data_google_elevation(photo.coordinates_lat_long_as_string,
                                                                peak.coordinates_lat_long_as_string,
                                                                NUMBER_SAMPLES, 
                                                                GOOGLE_API_KEY, 
                                                                earth_surface_y_values, 
                                                                HEIGHT_UNITS)
        if HEIGHT_UNITS == "Feet":
            elevation_data[0] = elevation_data[0] + 20
            elevation_data[-1] = elevation_data[-1] + 20
        else:
            elevation_data[0] = elevation_data[0] + 5
            elevation_data[-1] = elevation_data[-1] + 5
        
        los_path = self.create_los_path(distance_x_values, elevation_data[0], elevation_data[-1])
        intersection_color = self.select_color(los_path, elevation_data)
        if self.in_line_of_sight(los_path, elevation_data, error_threshold=error_threshold_distance):
            peak.elevation = elevation_data[-1]
            if plot:
                x_values = distance_x_values
                plt.figure(figsize=(20, 10))
                plt.plot(x_values, elevation_data)  # Terrain path
                plt.plot(x_values, earth_surface_y_values)  # Earth curvature path
                plt.plot(x_values, los_path, color=intersection_color)  # Line of sight path
                plt.fill_between(x_values, elevation_data, 0, alpha=0.1)
                plt.text(x_values[0], elevation_data[0], photo.name + ": " + str(photo.height))
                plt.text(x_values[-1], elevation_data[-1], peak.name + ": " + str(peak.height))
                plt.xlabel("Distance (" + DISTANCE_UNITS + ")"),
                plt.ylabel("Elevation (" + HEIGHT_UNITS + ")"),
                plt.grid()
                plt.legend(fontsize='small')
                plt.show()
            return True
        else: return False
        
        
        
"""
Accepts two arguments: the radius of the circle and the length 
of arc between the locations which are being assessed. This generates a 
representation of the Earth's curvature which will be represented on the final 
graph. This is important to represent as the curvature of the Earth makes a big 
difference to whether or not, line of sight exists between locations.
"""
class Circle:

    def __init__(self, radius_of_circle, length_of_arc):
        self.radius = radius_of_circle
        self.arc_length = length_of_arc

    def calc_degrees(self) -> float:
        return self.calc_radians() * 180 / np.pi

    def calc_radians(self) -> float: 
        return self.arc_length / self.radius

    """
    Returns the chord lengths of the arc, taking theta (angle in radians) as 
    it's argument The chord is the horizontal line which separates the arc 
    segment from the rest of the circle. Formula for theta (radians) only, not 
    degrees. Confirmed using http://www.ambrsoft.com/Trigocalc/Sphere/Arc_.htm
    """
    def calc_chord_length(self) -> float:
        return 2 * self.radius * np.sin(self.calc_radians() / 2)

    """
    Calculates the length of arc, taking theta (angle in radians) as its 
    argument. Confirmed using http://www.ambrsoft.com/Trigocalc/Sphere/Arc_.htm
    """
    def calc_arc_length(self) -> float:
        return self.arc_length

    """
    Calculates the Sagitta of the arc segment.  The Sagitta is the distance from
    the centre of the arc to the centre of the chord. Confirmed correct against 
    online calculator https://www.liutaiomottola.com/formulae/sag.htm
    """
    def calc_sagitta(self) -> float:
        return float(self.radius) - (np.sqrt((float(self.radius) ** 2) - ((float(self.calc_chord_length() / 2)) ** 2)))

    """
    Calculate the distance between the chord of the segment and the centre of 
    the circle
    """
    def calc_arc_apothem(self) -> float:
        return round(self.radius - self.calc_sagitta(), 8)

    def calc_circular_centre_x(self) -> float:
        return self.calc_chord_length() / 2

    def calc_circular_centre_y(self) -> float:
        return self.calc_sagitta() - self.radius

    def calc_diameter(self) -> float:
        return self.radius * 2

    """
    Takes two arguments, starting y and x coordinates
    Returns the starting angle of the circular arc as float in radians
    """
    def calc_start_angle(self, start_y, start_x) -> float:
        centre_y = self.calc_circular_centre_y()
        centre_x = self.calc_circular_centre_x()
        return np.arctan2(start_y - centre_y, start_x - centre_x)

    """
    Takes two arguments, ending y and x coordinates
    Returns the ending angle of the circular arc as float in radians
    """
    def calc_end_angle(self, end_y, end_x) -> float:
        centre_y = self.calc_circular_centre_y()
        centre_x = self.calc_circular_centre_x()
        return np.arctan2(end_y - centre_y, end_x - centre_x)

    """
    Returns a numpy array of y-axis values for mapping on matplotlib graph.  
    x values list is a list of distances in nautical miles. Each y-axis value 
    represents the rising and falling of the earth to simulate 'curvature' which
    effects line of sight visibility.
    """
    def calculate_earth_surface_y_values(self, list_of_x_axis_values, 
                                         list_of_angles, height_units, 
                                         distance_units) -> np.ndarray:
        y_values_list = []
        for x in range(len(list_of_x_axis_values)):
            """
            Calculate the y axis value (height) for the corresponding x value 
            (distance). Subtract the apothem of the circle to ensure the arc 
            starts at coordinates 0,0 and ends at zero again on the y axis
            """
            y = self.radius * np.sin(list_of_angles[x]) - self.calc_arc_apothem()
            y = round(convert_y_values(y, distance_units, height_units), 5)
            y_values_list.append(y)

        return np.array(y_values_list)
        
       
def getRotationMatrixX(ax, in_rad=False):
    if not in_rad:
        ax = np.radians(ax)
    sx = np.sin(ax)
    cx = np.cos(ax)
    return np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))

def getRotationMatrixY(ay, in_rad=False):
    if not in_rad:
        ay = math.radians(ay)
    sy = np.sin(ay)
    cy = np.cos(ay)
    return np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))

def getRotationMatrixZ(az, in_rad=False):
    if not in_rad:
        az = math.radians(az)
    sz = np.sin(az)
    cz = np.cos(az)
    return np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))

def getRotationMatrix(ax,ay,az,in_rad=False,order="XYZ"):
    X = getRotationMatrixX(ax, in_rad)
    Y = getRotationMatrixY(ay, in_rad)
    Z = getRotationMatrixZ(az, in_rad)

    if order == "XYZ":
        return Z @ Y @ X
    elif order == "ZYX":
        return X @ Y @ Z
    elif order == "YXZ":
        return Z @ X @ Y
    elif order == "YZX":
        return X @ Z @ Y
    elif order == "XZY":
        return Y @ Z @ X
    elif order == "ZXY":
        return Y @ X @ Z

def getTranslationMatrix(latitude, longitude):
    T = np.array([1.0, 0.0 -latitude],
                 [0.0, 1.0, -longitude],
                 [0.0, 0.0, 1.0])
    


# Return camera to world and world to camera
def getHomoTransforms(rotation_matrix, translation_matrix=np.array([[0,0,0]]).T):
    H_c_w = np.block([[rotation_matrix, translation_matrix], [0,0,0,1]]) 
    H_w_c = np.linalg.inv(H_c_w)
    return H_c_w, H_w_c


def put_text(img, text, org, font_face, font_scale, color, thickness=1, line_type=8, bottom_left_origin=False):
    """Utility for drawing text with line breaks
    :param img: Image.
    :param text: Text string to be drawn.
    :param org: Bottom-left corner of the first line of the text string in the image.
    :param font_face: Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
                          FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL,
                          FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font IDÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢s
                          can be combined with FONT_ITALIC to get the slanted letters.
    :param font_scale: Font scale factor that is multiplied by the font-specific base size.
    :param color: Text color.
    :param thickness: Thickness of the lines used to draw a text.
    :param line_type: Line type. See the line for details.
    :param bottom_left_origin: When true, the image data origin is at the bottom-left corner.
                               Otherwise, it is at the top-left corner.
    :return: None; image is modified in place
    """
    # Break out drawing coords
    x, y = org

    # Break text into list of text lines
    text_lines = text.split('\n')

    # Get height of text lines in pixels (height of all lines is the same)
    _, line_height = cv2.getTextSize('', font_face, font_scale, thickness)[0]
    # Set distance between lines in pixels
    line_gap = line_height // 3

    for i, text_line in enumerate(text_lines):
        # Find total size of text block before this line
        line_y_adjustment = i * (line_gap + line_height)

        # Move text down from original line based on line number
        if not bottom_left_origin:
            line_y = y + line_y_adjustment
        else:
            line_y = y - line_y_adjustment

        # Draw text
        cv2.putText(img,
                    text=text_lines[i],
                    org=(x, line_y),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=line_type,
                    bottomLeftOrigin=bottom_left_origin)


def put_centered_text(img, text, org, font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale=4, color=(0,0,255), thickness=4, line_type=8):
    """Utility for drawing vertically & horizontally centered text with line breaks
    :param img: Image.
    :param text: Text string to be drawn.
    :param font_face: Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
                          FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL,
                          FONT_HERSHEY_SCRIPT_SIMPLEX, or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font IDÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢s
                          can be combined with FONT_ITALIC to get the slanted letters.
    :param font_scale: Font scale factor that is multiplied by the font-specific base size.
    :param color: Text color.
    :param thickness: Thickness of the lines used to draw a text.
    :param line_type: Line type. See the line for details.
    :return: None; image is modified in place
    """
    # Save img dimensions
    img_h, img_w = img.shape[:2]

    # Break text into list of text lines
    text_lines = text.split(' ')

    # Get height of text lines in pixels (height of all lines is the same; width differs)
    _, line_height = cv2.getTextSize('', font_face, font_scale, thickness)[0]
    # Set distance between lines in pixels
    line_gap = line_height // 3

    # Calculate total text block height for centering
    text_block_height = len(text_lines) * (line_height + line_gap)
    text_block_height -= line_gap  # There's one less gap than lines

    for i, text_line in enumerate(text_lines):
        # Get width of text line in pixels (height of all lines is the same)
        line_width, _ = cv2.getTextSize(text_line, font_face, font_scale, thickness)[0]

        # Center line with image dimensions
        # Break out drawing coords
        x, y = org
        # x -= int(img_w/10)
        y -= int(line_height*len(text_lines)) -10
        if line_width > 300: x -= 100
        else: x += 50
        # Find total size of text block before this line
        line_adjustment = i * (line_gap + line_height)

        # Adjust line y and re-center relative to total text block height
        y += line_adjustment - text_block_height // 2 + line_gap

        # Draw text
        cv2.putText(img,
                    text=text_lines[i],
                    org=(x, y),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=color,
                    thickness=thickness,
                    lineType=line_type)
                    
"""
Carson's Google Maps API key:
AIzaSyDFH9DM-lNJai_3bpepD1YIAjzCOzu_Rw0 
For ar-peak-finder Project on GCP
"""


"""Returns the elevation data in a list"""
def send_and_receive_data_google_elevation(pos_1, pos_2, 
                                           number_samples, api_key, 
                                           y_values, height_units):
    url = construct_url_google_elevation(pos_1, pos_2, number_samples, api_key)
    sent_request = send_request_google_elevation(url)
    received_data = receive_request_google_elevation(sent_request)
    return process_response(received_data, y_values, height_units)

def construct_url_google_elevation(pos_1, pos_2, number_samples, api_key):
    return f'''https://maps.googleapis.com/maps/api/elevation/json?path={pos_1}|{pos_2}&samples={number_samples}&key={api_key}'''

def send_request_google_elevation(url):
    return urlopen(Request(url, headers={'Content-Type': 'application/json'}))

def receive_request_google_elevation(response_from_send_request):
    json_response = json.loads(response_from_send_request.read().decode("utf8"))
    response_from_send_request.close()
    return json_response

"""
This method extracts the 'elevation' values and adds the value corresponding 
to the rise or fall of the pseudo earth ellipsoid created by the Circle class.
A list of values are returned representing the final elevation values to be 
processed.
"""
def process_response(response, earth_surface_values, height_units):
    response_len = len(response['results'])
    elev_list = []
    lat_list = []
    long_list = []
    for j in range(response_len):
        """
        This manipulates the elevations so that they sit a correct distance 
        above the earth curve. In this instance the earth's curve represents the
        sea-level or '0' in terms of returned elevation values
        """
        lat_list.append(float(response['results'][j]['location']['lat']))
        long_list.append(float(response['results'][j]['location']['lng']))
        if height_units == "Metres":
            elev_list.append(response['results'][j]['elevation'] + earth_surface_values[j])
        elif height_units:
            elev_list.append((metres_to_feet(response['results'][j]['elevation'])) +
                             metres_to_feet(earth_surface_values[j]))
    return elev_list, lat_list, long_list

def metres_to_feet(value_in_metres):
    return value_in_metres * 3.281

def miles_to_metres(value_in_miles):
    return value_in_miles * 1609.34

def miles_to_feet(value_in_miles) -> float:
    return value_in_miles * 5280

def kilometres_to_metres(value_in_kilometres) -> float:
    return value_in_kilometres * 1000

def kilometres_to_feet(value_in_kilometres) -> float:
    return value_in_kilometres * 3281

def nautical_miles_to_metres(value_in_nautical_miles) -> float:
    return value_in_nautical_miles * 1852

def nautical_miles_to_feet(value_in_nautical_miles) -> float:
    return value_in_nautical_miles * 6076

def nautical_miles_to_statute_miles(value_in_nautical_miles):
    return value_in_nautical_miles * 1.151

def nautical_miles_to_kilometres(value_in_nautical_miles):
    return value_in_nautical_miles * 1.852

def define_earth_radius(unit_of_distance):
    if unit_of_distance == "Nautical miles": return 3440.065
    elif unit_of_distance == "Miles": return 3958.8
    elif unit_of_distance == "Kilometres": return 6371.0
    else: return Exception("A unit of distance was not specified or there was a typo in the code.  Get help!")

"""
Takes arguments of units of distance and two positions with long/lat
Returns the great circle distance in the correct units of measurement
"""
def calculate_great_circle_distance(pos_1, pos_2, unit_of_distance):
    if unit_of_distance == "Nautical miles": return haversine(pos_1, pos_2, unit=Unit.NAUTICAL_MILES)
    elif unit_of_distance == "Miles": return haversine(pos_1, pos_2, unit=Unit.MILES)
    elif unit_of_distance == "Kilometres": return haversine(pos_1, pos_2, unit=Unit.KILOMETERS)
    elif unit_of_distance == "Metres": return haversine(pos_1, pos_2, unit=Unit.METERS)
    else: return Exception("A unit of distance was not specified or there was a typo in the code.  Get help!")

"""
Returns a converted height value based upon the distance and height units of 
measurement selected by the user
"""
def convert_y_values(y_value, distance_units, height_units):
    if distance_units == "Nautical miles":
        if height_units == "Feet": return nautical_miles_to_feet(y_value)
        elif height_units == "Metres": return nautical_miles_to_metres(y_value)
        else: return Exception("Something went wrong converting nautical miles to the selected height units")
    elif distance_units == "Kilometres":
        if height_units == "Feet": return kilometres_to_feet(y_value)
        elif height_units == "Metres": return kilometres_to_metres(y_value)
        else: return Exception("Something went wrong converting Kilometres to the selected height units")
    elif distance_units == "Miles":
        if height_units == "Feet": return miles_to_feet(y_value)
        elif height_units == "Metres": return miles_to_metres(y_value)
    else: return Exception("Something went wrong converting y values from the distance units to the height units.")


def check_photo_exif(path_to_photo, print_data=False):
    data = exifread.process_file(open(path_to_photo, 'rb'))
    try:
        latitudes = data["GPS GPSLatitude"].values
        latitude = np.round(float(latitudes[0]) + float(latitudes[1])/60 + float(latitudes[2])/3600,15)
        longitudes = data["GPS GPSLongitude"].values
        longitude = np.round(float(longitudes[0]) + float(longitudes[1])/60 + float(longitudes[2])/3600,15)
        latitude *= (-1 if data["GPS GPSLatitudeRef"].values[0] in ['S','W'] else 1)
        longitude *= (-1 if data["GPS GPSLongitudeRef"].values[0] in ['S','W'] else 1)
        altitude = str(data["GPS GPSAltitude"].values[0]).split("/")
        altitude = np.round(float(int(altitude[0])/int(altitude[1])), 15)
        height = data["EXIF ExifImageLength"].values[0]
        width = data["EXIF ExifImageWidth"].values[0]
        x_resolution = int(data["Image XResolution"].values[0])
        y_resolution = int(data["Image YResolution"].values[0])
        focal_lengths = str(data["EXIF FocalLength"].values[0]).split("/")
        focal_length = np.round(float(int(focal_lengths[0])/int(focal_lengths[1])), 15)
        heading = str(data["GPS GPSImgDirection"].values[0]).split("/")
        heading = np.round(float(heading[0])/int(heading[1]), 6)
    except:
        if print_data:
            if len(data) < 1: print("Image {} doesn't have any GPS data".format(path_to_photo))
            else: print("Image {} is Missing Required EXIF data".format(path_to_photo))
        return False
    return True


def check_line_of_sight(photo: Location, peak: Location, error_threshold_distance, plot=True):
    GOOGLE_API_KEY = 'AIzaSyDFH9DM-lNJai_3bpepD1YIAjzCOzu_Rw0' # For ar-peak-finder Project
    NUMBER_SAMPLES = 500
    DISTANCE_UNITS = "Kilometres" # 'Kilometres', 'Feet', 'Nautical miles'
    HEIGHT_UNITS = "Metres" # 'Kilometres', 'Metres', 'Nautical miles'


    great_circle_distance = calculate_great_circle_distance((float(photo.latitude), 
                                                            float(photo.longitude)), 
                                                            (float(peak.latitude), 
                                                            float(peak.longitude)), 
                                                            DISTANCE_UNITS)
    
    EARTH = Circle(define_earth_radius(DISTANCE_UNITS), great_circle_distance)
    angle_list = np.linspace(EARTH.calc_start_angle(0, 0), EARTH.calc_end_angle(0, great_circle_distance), NUMBER_SAMPLES)
    distance_x_values = np.linspace(0, great_circle_distance, NUMBER_SAMPLES)
    earth_surface_y_values = EARTH.calculate_earth_surface_y_values(distance_x_values, 
                                                                    angle_list, 
                                                                    HEIGHT_UNITS, 
                                                                    DISTANCE_UNITS)

    elevation_data, lat_data, long_data = send_and_receive_data_google_elevation(photo.coordinates_lat_long_as_string,
                                                            peak.coordinates_lat_long_as_string,
                                                            NUMBER_SAMPLES, 
                                                            GOOGLE_API_KEY, 
                                                            earth_surface_y_values, 
                                                            HEIGHT_UNITS)
    if HEIGHT_UNITS == "Feet":
        elevation_data[0] = elevation_data[0] + 20
        elevation_data[-1] = elevation_data[-1] + 20
    else:
        elevation_data[0] = elevation_data[0] + 5
        elevation_data[-1] = elevation_data[-1] + 5
    
    los_path = create_los_path(distance_x_values, elevation_data[0], elevation_data[-1])
    intersection_color = select_color(los_path, elevation_data)
    if in_line_of_sight(los_path, elevation_data, error_threshold=error_threshold_distance):
        if plot:
            x_values = distance_x_values
            plt.figure(figsize=(20, 10))
            plt.plot(x_values, elevation_data)  # Terrain path
            plt.plot(x_values, earth_surface_y_values)  # Earth curvature path
            plt.plot(x_values, los_path, color=intersection_color)  # Line of sight path
            plt.fill_between(x_values, elevation_data, 0, alpha=0.1)
            plt.text(x_values[0], elevation_data[0], photo.name + ": " + str(photo.height))
            plt.text(x_values[-1], elevation_data[-1], peak.name + ": " + str(peak.height))
            plt.xlabel("Distance (" + DISTANCE_UNITS + ")"),
            plt.ylabel("Elevation (" + HEIGHT_UNITS + ")"),
            plt.grid()
            plt.legend(fontsize='small')
            plt.show()
        return True
    else: return False

def get_peaks_in_photo(photo: Photo, within_distance=20, error_threshold_distance=100):
    location = photo.location
    photo.display()
    return filter_peaks_aov(photo.aov, photo.heading, location, query_peaks_by_distance(location, within_distance, "miles"), error_threshold_distance=error_threshold_distance)


def get_current_location():
    return geocoder.ip('me').latlng

def format_peak_results(results, current_location):
    formatted_results={}
    for feature in results['features']:
        try:
            coordinates = tuple(reversed(feature["geometry"]["coordinates"]))
            peak_name = feature['properties']['name']
            distance = geopy_distance.geodesic(current_location, coordinates).miles
            formatted_results[distance] = {"name": peak_name, 
                                             "coordinates": coordinates, 
                                             "id": id}
        except KeyError: pass
    return formatted_results

def display_nearby_peaks(results, distance, num_results=5, printer=False):
    if printer:
        print('\n\nMountain Peaks Within {} Miles of You:'.format(distance))
        print("_"*100)
    coords = []
    peaks = []
    for i, peak in enumerate(sorted(results.items(), key=operator.itemgetter(0)),1):
        coords.append([peak[1]['coordinates'][0], peak[1]['coordinates'][1]])    
        peaks.append(Peak(peak[1]['name'], peak[1]['coordinates'][0], peak[1]['coordinates'][1]))
        if printer:
            print("{}. {} ({} miles): {}, {}".format(i, peak[1]['name'], 
                                                    round(peak[0], 1), 
                                                    peak[1]['coordinates'][0], 
                                                    peak[1]['coordinates'][1]))
        if i == num_results:
            
            if printer: 
                print("\n***Stopped search due to maximum peaks {} found***\n".format(num_results))
                plot_mountains(coords, title="Mountain Peaks Within {} Miles of You:".format(distance))
            return True
    
def get_nearby_peaks(location=get_current_location(), min_range=1, max_range=100, range_inc=10, num_results=50):
    OVERPASS_API = overpass.API()
    if isinstance(location, Location):
        latitude = location.latitude
        longitude = location.longitude
    else:
        latitude = location[0]
        longitude = location[1]
    iterator = tqdm(range(min_range, max_range, range_inc))

    for radius in iterator:
        clear_output()
        print('\n\nStarting Search for Peaks Within {} Miles of You:'.format(max_range))
        RANGE = radius*1609.344
        overpass_query = f'''node(around:{RANGE},{latitude},{longitude})[natural=peak];'''
        results = OVERPASS_API.get(overpass_query)
        if len(results['features']) > 1:
            max_results = display_nearby_peaks(format_peak_results(results, (latitude, longitude)), radius, num_results, printer=False)
            if max_results: 
                iterator.close()
                clear_output()
                display_nearby_peaks(format_peak_results(results, (latitude, longitude)), radius, num_results)
                break
        elif radius == max_range:
            print("No Peaks found winthin {} mile radius".format(max_range))

def plot_mountains(list_of_coords, title="Mountains"):
    plt.figure(figsize=(10, 10))
    X = np.array(list_of_coords)
    plt.plot(X[:, 0], X[:, 1], 'o')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.grid()
    plt.show()   
    
    
def get_lat_long_by_distance(distance, bearing, origin: Location):
    R = 6378.1 #Radius of the Earth

    lat1 = math.radians(float(origin.latitude)) #Current lat point converted to radians
    lon1 = math.radians(float(origin.longitude)) #Current long point converted to radians

    lat2 = math.asin( math.sin(lat1)*math.cos(distance/R) +
        math.cos(lat1)*math.sin(distance/R)*math.cos(bearing))

    lon2 = lon1 + math.atan2(math.sin(bearing)*math.sin(distance/R)*math.cos(lat1),
                math.cos(distance/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return Location(lat2,lon2)

# Mount_Elbert = Location( "39.1178", "-106.4454", 6, "Mount Elbert")
# Mount_Elbert = Location( "39.749668", "-105.2206394", 6, "Mount Elbert")


def surrounding_topology(origin: Location, distance=20, detail=1):

    GOOGLE_API_KEY = 'AIzaSyDFH9DM-lNJai_3bpepD1YIAjzCOzu_Rw0' # For ar-peak-finder Project
    NUMBER_SAMPLES = 500
    DISTANCE_UNITS = "Kilometres" # 'Kilometres', 'Feet', 'Nautical miles'
    HEIGHT_UNITS = "Metres" # 'Kilometres', 'Metres', 'Nautical miles'
    EARTH_RADIUS = define_earth_radius(DISTANCE_UNITS)
    HEIGHT_OFF_GROUND = origin.height

    elevation_data = []
    lat_data = []
    long_data = []
    for bearing in tqdm(np.linspace(0,360,int(360/detail))):
        dest = get_lat_long_by_distance(20, bearing, origin)

        great_circle_distance = calculate_great_circle_distance((float(origin.latitude), 
                                                            float(origin.longitude)), 
                                                            (float(dest.latitude), 
                                                            float(dest.longitude)), 
                                                            DISTANCE_UNITS)

        # Circle object to simulate curvature of the earth.
        EARTH = Circle(EARTH_RADIUS, great_circle_distance)

        # Start and end points for the earths curvature
        angle_list = np.linspace(EARTH.calc_start_angle(0, 0), EARTH.calc_end_angle(0, great_circle_distance), NUMBER_SAMPLES)

        distance_x_values = np.linspace(0, great_circle_distance, NUMBER_SAMPLES)

        earth_surface_y_values = EARTH.calculate_earth_surface_y_values(distance_x_values, 
                                                                        angle_list, 
                                                                        HEIGHT_UNITS, 
                                                                        DISTANCE_UNITS)


        elevation_data1, lat_data1, long_data1 = send_and_receive_data_google_elevation(origin.coordinates_lat_long_as_string,
                                                                dest.coordinates_lat_long_as_string,
                                                                NUMBER_SAMPLES, 
                                                                GOOGLE_API_KEY, 
                                                                earth_surface_y_values, 
                                                                HEIGHT_UNITS)
        elevation_data.extend(elevation_data1)
        lat_data.extend(lat_data1)
        long_data.extend(long_data1)

    # transform to numpy arrays
    x = np.array(lat_data)
    y = np.array(long_data)
    z = np.array(elevation_data)
    zmin = np.min(z)

    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    xi, yi = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (xi, yi), method='linear', fill_value=zmin)

    fig = plt.figure(figsize=(20, 10))
    ax = Axes3D(fig)
    ax.plot_surface(xi, yi, Z, rstride=1, cstride=1, cmap = plt.get_cmap('rainbow'),linewidth=0.75)
    ax.set_zlim(zmin, 4500)
    ax.view_init(60, 25)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation')
    fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('rainbow')))
    plt.show()
    
    

def projection_coordinates(photo: Photo, dest):
    if isinstance(dest, Peak): dest = latlon_to_xyz(dest.location)
    elif isinstance(dest, Location): dest = latlon_to_xyz(dest)
    else: assert(isinstance(dest, np.array))
    origin = latlon_to_xyz(photo.location)
    rel_dest = np.append(dest - origin, 1.0)

    rel_cam = photo.K @ photo.Mext @ rel_dest
    x = rel_cam[0]/rel_cam[2]
    y = rel_cam[1]/rel_cam[2]
    return x, y

def latlon_to_xyz(location: Location):
    try:elevation = location.elevation
    except:elevation = location.set_elevation_google()
    cos_lat = np.cos((location.latitude*np.pi)/180.0)
    sin_lat = np.sin((location.latitude*np.pi)/180.0)
    cos_lon = np.cos((location.longitude*np.pi)/180.0)
    sin_lon = np.sin((location.longitude*np.pi)/180.0)

    r = 6378137.0
    f = 1.0/2982.57224
    C = 1.0/(np.sqrt(cos_lat * cos_lat + (1 - f) * (1 - f) * sin_lat * sin_lat))
    S = (1.0-f) * (1.0-f) * C

    x = (r * C + elevation) * cos_lat * cos_lon
    y = (r * C + elevation) * cos_lat * sin_lon
    z = (r * S + elevation) * sin_lat

    return np.array([x, y, z])        
        