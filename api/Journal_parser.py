import numpy as np 
import matplotlib.pylab as plt
from glob import glob
import os
from scipy.stats import mode
import skimage
from skimage.transform import hough_line, hough_line_peaks
from skimage import filters
from tqdm import tqdm
import sys

class Journal_parser():

    def treshold_image(self, img, gamma, small_object_th):
        adjusted = skimage.exposure.adjust_gamma(img, gamma = gamma)
        gray_img_for_th = skimage.color.rgb2gray(adjusted)
        th_value = filters.threshold_sauvola(gray_img_for_th, window_size=25)
        thresholded_img = gray_img_for_th < th_value
        thresholded_img_clean = skimage.morphology.remove_small_objects(thresholded_img, small_object_th)

        return thresholded_img_clean

    def rotate_image(self, image, angles, center):
        if len(angles) == 0:
            angle = 0
        else:
            angle = np.rad2deg(mode(angles[abs(angles) > 2])[0][0])
            
        if (angle < 0):
            angle = angle + 180
        else:
            angle = angle - 180
        rotated_image = skimage.transform.rotate(image, angle = angle, resize=False, center = center)

        return rotated_image
    
    def filter_lines(self, h_lines, v_lines):
        h_lines = np.unique(np.array(h_lines, dtype=np.int))
        v_lines = np.unique(np.array(v_lines, dtype=np.int))

        horisontal_lines_filtred = []
        h_line_prew = 0
        for h_line in h_lines:
            if h_line - h_line_prew > 5:
                horisontal_lines_filtred.append(h_line)
                h_line_prew = h_line

        vertical_lines_filtred = []
        v_line_prew = 0
        for v_line in v_lines:
            if v_line - v_line_prew > 5:
                vertical_lines_filtred.append(v_line)
                v_line_prew = v_line

        return horisontal_lines_filtred, vertical_lines_filtred
    
    def hough_line_selection(self, input_img, eps = 0.1, bins = 50, gamma = 3, small_object_th = 10000, save_result = False):
        thresholded_img_clean = self.treshold_image(input_img, gamma, small_object_th)

        tested_angles = np.concatenate((np.linspace(-np.pi/2 - eps, -np.pi/2 + eps, bins), 
                                        np.linspace(np.pi/2 - eps, np.pi/2 + eps, bins),
                                        np.linspace(-np.pi - eps, -np.pi + eps, bins),
                                        np.linspace(np.pi - eps, np.pi + eps, bins)))
        hspace, theta, dist = hough_line(thresholded_img_clean, tested_angles)
        
        vertical_lines = []
        horisontal_lines = []
        v_origin = 0
        h_origin = input_img.shape[0]
        _, angles, distances = hough_line_peaks(hspace, theta, dist)
        rotated_image = self.rotate_image(input_img, angles, np.array((input_img.shape[1], 0)))
        for angle, dist in zip(angles, distances):
            if abs(angle) > 2:
                #vertical lines
                y0 = (dist - v_origin * np.sin(angle)) / np.cos(angle)
                vertical_lines.append(y0)
                
            else:
                #horisontal lines
                y0 = (dist - h_origin * np.cos(angle)) / np.sin(angle)
                horisontal_lines.append(y0)
                
        horisontal_lines, vertical_lines  = self.filter_lines(horisontal_lines, vertical_lines)
        
        return np.array(horisontal_lines), np.array(vertical_lines), rotated_image
