import pandas as pd
import os

def _get_root_data_folder():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    attr_id_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))
    real_root_dir = os.path.abspath(os.path.join(attr_id_dir, os.pardir))
    return os.path.join(real_root_dir, "data")

def _get_anon_type_folder(anon_type):
    """
    For every image we create a new folder where we save all crops.
    """
    root_data_folder = _get_root_data_folder()
    all_crop_folder = os.path.join(root_data_folder, "all_crop_versions")
    unedited_folder = os.path.join(all_crop_folder, anon_type)
    return unedited_folder

def _get_all_city_folders(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    return subfolders

def collect_all_footage_dfs(anon_type):
    """
    Return-value structure {city_name: unedited_city_df}
    """
    unedited_folder = _get_anon_type_folder(anon_type)
    get_all_city_folders = _get_all_city_folders(unedited_folder)
    unedited_footage_df = {}

    for city_path in get_all_city_folders:
        city_name = str(city_path).split("/")[-1]
        city_df = pd.read_excel(os.path.join(city_path, f"{city_name}_pre_processed.xlsx"))
        unedited_footage_df[city_name] = city_df

    return unedited_footage_df


def _create_single_structure_basis(anon_type):
    unedited_footage_dfs = collect_all_footage_dfs(anon_type)
    person_id_list = []
    lowest, highest = 1, 0
    for city in unedited_footage_dfs:
        for index, row in unedited_footage_dfs[city].iterrows():
            if row["person_id"] < lowest:
                lowest = row["person_id"]
            if row["person_id"] > highest:
                highest = row["person_id"]
        
    person_id_list = list(range(lowest, highest+1))
    
    return pd.DataFrame({"person_id": person_id_list})

def _get_savepath_single_structure_basis(anon_type):
    root_data_folder = _get_root_data_folder()
    sist_rel_folder = os.path.join(root_data_folder, "single_structured_relations")
    savepath = os.path.join(sist_rel_folder, f"{anon_type}.xlsx")
    return savepath

def initialise_single_structure_basis(anon_type: str):
    single_structure_basis = _create_single_structure_basis(anon_type)
    save_path = _get_savepath_single_structure_basis(anon_type)
    single_structure_basis.to_excel(save_path)

def get_anon_type_sist_basis(anon_type: str):
    return pd.read_excel(_get_savepath_single_structure_basis(anon_type))


def update_sist_df(current_sist, new_column_data_dict, anon_type):
    """
    Updates the current sist DataFrame with a new column containing data from the provided dictionary.
    Works in-place.

    new_column_data_dict has the following structure:
    {
        "new_column_name": {
            "person_id": <data_for_that_person>,
            ...,
            "person_id": <data_for_that_preson>
        }
    }
    """

    if "person_id" not in current_sist.columns:
        raise ValueError("The DataFrame must contain a 'person_id' column.")
    
    for column in new_column_data_dict:
        new_data_df = pd.DataFrame(list(new_column_data_dict[column].items()), columns=["person_id", column])
        current_sist = current_sist.merge(new_data_df, on="person_id", how="left")

    updated_sist = current_sist

    if "Unnamed: 0" in updated_sist.columns:
        updated_sist = updated_sist.drop("Unnamed: 0", axis='columns')
        
    save_path = _get_savepath_single_structure_basis(anon_type)
    updated_sist.to_excel(save_path)