import sys
import os
from PIL import Image

import specified_helper_functions as helper
import HWFinder as model_handling

# Configure logging
import logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def _classify_height_and_weight(height_or_weight: float) -> str:
    """
    Groups height or weight into classes of 5.
    """
    if height_or_weight is None:
        return height_or_weight

    lower_bound = (int(height_or_weight) // 3) * 3
    upper_bound = lower_bound + 2
    return f"{lower_bound}-{upper_bound}"

def _subtract(coordinate):
    """
    Mainly to check if the coordinates are within the bounds of 
    """
    if coordinate - 15 >= 0:
        return coordinate - 15
    return 0

def _add(coordinate, width_or_height):
    """
    Mainly to check if the coordinates are within the bounds of
    """
    if coordinate + 15 <= width_or_height:
        return coordinate + 15
    return width_or_height

def _enlarge_body_crop(original_img_path, body_crop_bbox):
    original_img = Image.open(original_img_path)
    width, height = original_img.size
    new_body_crop_bbox = [
        _subtract(body_crop_bbox[0]),
        _subtract(body_crop_bbox[1]),
        _add(body_crop_bbox[2], width),
        _add(body_crop_bbox[3], height)
    ]
    temporary_path = "data/temporary_enlarged_body.png"
    enlarged_body_crop = original_img.crop(new_body_crop_bbox)
    enlarged_body_crop.save(temporary_path)
    return temporary_path

def age_and_gender_classification(anon_type):
    sist_basis = helper.get_anon_type_sist_basis(anon_type)
    all_unedited_cities = helper.collect_all_footage_dfs(anon_type)
    model_h, model_w = model_handling.get_models()

    height_data_dict = {}
    weight_data_dict = {}
    for city in all_unedited_cities:
        for index, row in all_unedited_cities[city].iterrows():
            temporary_image_path = _enlarge_body_crop(row["original_img_path"], row["body_crop_bbox"])
            try:
                height, weight = model_handling.get_height_and_weight(model_h, model_w, temporary_image_path)
            except RuntimeError:
                logging.warning(f"Model was unable to obtain height and weight information on person with id {row['person_id']}")
                height, weight = None, None
            height_data_dict[row["person_id"]] = _classify_height_and_weight(height)
            weight_data_dict[row["person_id"]] = _classify_height_and_weight(weight)
            os.remove(temporary_image_path)

    new_column_data_dict = {
        "weight": height_data_dict,
        "height": weight_data_dict
    }

    helper.update_sist_df(sist_basis, new_column_data_dict, anon_type)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        age_and_gender_classification(sys.argv[1])
    else:
        logging.warning(f"Height and Weight Estimator weren't able to be executed due to missing anon_type argument!")