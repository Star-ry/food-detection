import os


def seg_to_bbox(seg_info):
    if not seg_info:
        return ""

    # Example input: 5 0.046875 0.369141 0.0644531 0.384766 0.0800781 0.402344 ...
    class_id, *points = seg_info.split()
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = min(points[0::2]), min(points[1::2]), max(points[0::2]), max(points[1::2])
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    bbox_info = f"{int(class_id)} {x_center} {y_center} {width} {height}"
    return bbox_info


label_root_path = 'DATA/valid/labels_ori'
output_root_path = 'data/valid/labels'
os.makedirs(output_root_path, exist_ok=True)


for text_file_name in os.listdir(label_root_path):
    text_file_path = os.path.join(label_root_path, text_file_name)
    output_file_path = os.path.join(output_root_path, text_file_name)
    
    if os.path.isfile(text_file_path):
        with open(text_file_path, 'r') as text_file:
            text = text_file.read()
        
        obj_info_list = text.split('\n')
        obj_cnt = len(obj_info_list)
        
        with open(output_file_path, 'w') as output_file:
            for i in range(obj_cnt):
                obj_info = obj_info_list[i]
                temp = seg_to_bbox(obj_info)
                output_file.write(temp + '\n')



