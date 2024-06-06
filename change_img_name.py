import os
from PIL import Image, ImageDraw, ImageFont

folder_name = 'test'

img_root_path = f'data/{folder_name}/images'
labels_root_path = f'data/{folder_name}/labels'

text_files = sorted(os.listdir(labels_root_path))
img_files = sorted(os.listdir(img_root_path))

for index, text_file_name in enumerate(text_files):
    text_file_path = os.path.join(labels_root_path, text_file_name)
    img_file_name = text_file_name.replace('.txt', '.jpg')
    img_file_path = os.path.join(img_root_path, img_file_name)
    
    new_text_file_name = f'{index}.txt'
    new_img_file_name = f'{index}.jpg'
    
    new_text_file_path = os.path.join(labels_root_path, new_text_file_name)
    new_img_file_path = os.path.join(img_root_path, new_img_file_name)
    
    os.rename(text_file_path, new_text_file_path)
    os.rename(img_file_path, new_img_file_path)
