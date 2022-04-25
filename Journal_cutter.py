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

class Journal_cutter():

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
    
    def hough_line_selection(self, input_img, output_result_path, eps = 0.1, bins = 50, gamma = 3, small_object_th = 10000, save_result = False):
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

        if save_result:
            fig, ax = plt.subplots(1, 2, figsize = (18, 8))
            ax[0].imshow(thresholded_img_clean, cmap='gray')
            ax[1].imshow(rotated_image)

            for v_line in vertical_lines:
                ax[1].axvline(x=v_line, color='r', linestyle='-')
                #print(f"y0: {v_line[0]}, y1: {v_line[1]}, angle: {angle}")

            for h_line in horisontal_lines:
                ax[1].axhline(y=h_line, color='r', linestyle='-')
                #print(f"y0: {h_line[0]}, y1: {h_line[1]}, angle: {angle}")
                
            plt.tight_layout()
            plt.savefig(output_result_path)
        
        return np.array(horisontal_lines), np.array(vertical_lines), rotated_image

    def transform(self, imgs_dir, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        for img_path in tqdm(glob('{}/*'.format(imgs_dir))):
            print(img_path)
            test_img = plt.imread(img_path)
            img_name = img_path.split('/')[-1].split('.')[0]

            if not os.path.isdir(os.path.join(output_dir, img_name)):
                os.mkdir(os.path.join(output_dir, img_name))
                os.mkdir(os.path.join(output_dir, img_name, 'names'))
                os.mkdir(os.path.join(output_dir, img_name, 'grades'))

            output_result_path = os.path.join(output_dir, img_name, 'line_deteciton_result.png')
            horisontal_lines, vertical_lines, rotated_image = self.hough_line_selection(test_img, output_result_path, save_result = True)

            for i in range(2, horisontal_lines.shape[0] - 1):
                for j in range(vertical_lines.shape[0] - 1):
                    if (horisontal_lines[i + 1] - horisontal_lines[i] > 5) and (vertical_lines[j + 1] - vertical_lines[j] > 5):
                        cell = rotated_image[horisontal_lines[i]:horisontal_lines[i + 1],
                                               vertical_lines[j]:vertical_lines[j + 1]]
                        if cell.shape[0] > 0 and cell.shape[1] > 0:                
                            if j == 0:
                                output_path = os.path.join(output_dir, img_name, 'names', '{}{}_{}.png'.format(img_name[-2:], i,j))
                                plt.imsave(output_path, cell)
                            else:
                                output_path = os.path.join(output_dir, img_name, 'grades', '{}{}_{}.png'.format(img_name[-2:], i,j))
                                plt.imsave(output_path, cell)

if __name__ == "__main__":
    cutter = Journal_cutter()
    if len(sys.argv) == 3:
        cutter.transform(sys.argv[1], sys.argv[2])
    else:
        intput_img_dir = os.path.join('.', 'data', 'original_images')
        output_dir = os.path.join('.', 'data', 'cutt_result')
        cutter.transform(intput_img_dir, output_dir)