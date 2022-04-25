import matplotlib.pylab as plt
from Journal_parser import Journal_parser
from Use_transformet import Model
#from Use_crnn import Model
import pandas as pd
from tqdm import tqdm

class Transform():
    def __init__(self):
        self.parser = Journal_parser()
        self.model = Model()

    
    def transform(self, img_path):
        test_img = plt.imread(img_path)
        output_talbe = pd.DataFrame()
        horisontal_lines, vertical_lines, rotated_image = self.parser.hough_line_selection(test_img, save_result = True)

        for i in tqdm(range(2, horisontal_lines.shape[0] - 1)):
            for j in range(vertical_lines.shape[0] - 1):
                if (horisontal_lines[i + 1] - horisontal_lines[i] > 5) and (vertical_lines[j + 1] - vertical_lines[j] > 5):
                    cell = rotated_image[horisontal_lines[i]:horisontal_lines[i + 1],
                                            vertical_lines[j]:vertical_lines[j + 1]]
                    if cell.shape[0] > 0 and cell.shape[1] > 0:
                        output_string = self.model.predict(cell)
                        output_talbe.loc[i, j] = output_string
        
        plt.imsave('./Test_results/input_img.jpeg', test_img)
        output_talbe.to_json('./Test_results/output_table.json')
        output_talbe.to_csv('./Test_results/output_table.csv', sep='|', index=False, header=True)

if __name__ == "__main__":
    journal_processor = Transform()
    journal_processor.transform('../journal_cutter/data/original_images/scann-7.jpg')
        