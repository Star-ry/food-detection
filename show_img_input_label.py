import json
import yaml
import os
from PIL import Image

json_path = 'food-detection/data_info.json'

with open(json_path) as json_file:
    json = json.load(json_file)
    
folder_name_list = ['train', 'valid', 'test']
data_yaml_path = 'food-detection/my_data.yaml'

with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)
labels_list = data_config['labels_list']

num_classes = 68

for id in range(num_classes):
    # input_id = input("Enter the class_id to display images for: ")
    label = labels_list[int(id)]


    for folder_name in folder_name_list:
        temp_root_path = f'sorted_data_in_classes/{id}_{label}/{folder_name}'
        os.makedirs(temp_root_path, exist_ok=True)
        
        for img_file_name in json[folder_name]:
            in_image = False
            for obj_info in json[folder_name][img_file_name]:
                if obj_info['class_id'] == int(id):
                    in_image = True
            if in_image == True:
                img_path = f'bbox_images/{folder_name}/images/{img_file_name}'
                temp_path = f'{temp_root_path}/{img_file_name}'
                image = Image.open(img_path)
                image.save(temp_path)
                
            