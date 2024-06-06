import os
from PIL import Image, ImageDraw, ImageFont
import yaml
import json

folder_name_list = ['train', 'valid', 'test']
json_path = 'data_info.json'
data_yaml_path = 'my_data.yaml'

with open(data_yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)
labels_list = data_config['labels_list']
labels_list_kr = data_config['labels_list_kr']

data_info = {folder: {} for folder in folder_name_list}

for folder_name in folder_name_list:
    img_root_path = f'data/{folder_name}/images'
    labels_root_path = f'data/{folder_name}/labels'
    save_root_path = f'bbox_images/{folder_name}/images'
    os.makedirs(save_root_path, exist_ok=True)

    for text_file_name in os.listdir(labels_root_path):
        text_file_path = os.path.join(labels_root_path, text_file_name)
        img_file_name = text_file_name.replace('.txt', '.jpg')
        img_file_path = os.path.join(img_root_path, img_file_name)
        save_img_path = os.path.join(save_root_path, img_file_name)
        
        if os.path.isfile(text_file_path) and os.path.isfile(img_file_path):
            with open(text_file_path, 'r') as text_file:
                text = text_file.read()
            
            obj_info_list = text.strip().split('\n')
            image = Image.open(img_file_path)
            draw = ImageDraw.Draw(image)
            
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            

            for obj_info in obj_info_list:
                if obj_info:
                    parts = obj_info.split()
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))

                    img_width, img_height = image.size
                    x_center, y_center, width, height = bbox
                    left = (x_center - width / 2) * img_width
                    top = (y_center - height / 2) * img_height
                    right = (x_center + width / 2) * img_width
                    bottom = (y_center + height / 2) * img_height
                    
                    draw.rectangle([left, top, right, bottom], outline='red', width=8)
                    
                    label = labels_list[class_id]
                    text_bbox = draw.textbbox((left, top), label, font=font)
                    text_background = (text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3])
                    draw.rectangle(text_background, fill='red')
                    draw.text((left, top), label, fill='white', font=font)
                    

            
            image.save(save_img_path)

