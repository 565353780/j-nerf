import sys

sys.path.append("../camera-manage/")
sys.path.append("../colmap-manage/")

from colmap_manage.Module.colmap_manager import COLMAPManager
from colmap_manage.Module.dataset_manager import DatasetManager
from j_nerf.Module.trainer import Trainer

data_folder_name = "NeRF/wine"
video_file_path = "None"
down_sample_scale = 1

scale = 1
show_image = False
print_progress = True
remove_old = False
remain_db = True
valid_percentage = 0.8
method_dict = {
    "aabb_scale": 2,
}

data_folder_path = "/home/chli/Dataset/" + data_folder_name + "/"
dataset_folder_path = (
    "../colmap-manage/output/" + data_folder_name.replace("/", "_") + "/"
)
output_folder_path = "../j-nerf/output/" + data_folder_name.replace("/", "_") + "/"

COLMAPManager(
    data_folder_path, video_file_path, down_sample_scale, print_progress=True
).autoGenerateData(remove_old, remain_db, valid_percentage)
DatasetManager().generateDataset(
    "jn", data_folder_path, dataset_folder_path, method_dict
)
Trainer(data_folder_name.replace("/", "_"), dataset_folder_path).train()