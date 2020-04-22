# AUTHOR: 
# DATE: 11:08
# FILE NEME: split_dataset.py.py
# TOOL: PyCharm
import pandas as pd
from sklearn.model_selection import train_test_split


annotation_path = '/Users/apple1/Desktop/2020iqiyi/dataset/personai_icartoonface_detrain/train_v1.csv'
dataset = pd.read_csv(annotation_path, header=None)
dataset.columns = ['image_name','x_min', 'y_min','x_max','y_max']

RANDOM_SEED = 42
train_df, test_df = train_test_split(dataset, test_size =0.1, random_state= RANDOM_SEED)

Train_file = 'train_dataset.txt'
Test_file = 'test_dataset.txt'
Class_file = '/Users/apple1/Desktop/2020iqiyi/yolo/keras-yolo3-master/model_data/classes.txt'

train_df.to_csv(Train_file, sep='\t', index = False, header = None)
test_df.to_csv(Test_file, sep='\t', index = False, header = None)
classes = set(['icartoonface'])

with open(Class_file, 'w') as f:
	for i, line in enumerate(sorted(classes)):
		f.write('{},{}\n'.format(line,i))
