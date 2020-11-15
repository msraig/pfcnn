import pySurfaceHierarchy
from read_data import DataReader
import os
mesh_dir_path = "D:/data/MPI-FAUST_training/normalized/registrations"
mesh_name = "tr_reg_000"
output_path = "D:/data/MPI-FAUST_training/normalized/reg_dataset/"
top_level_size = 6890
level_size_factor = 1.0
start_level = 0
level_num = 1
sym_N = 4
grid_length = [0.01]
grid_size = 5
if not os.path.exists(output_path):
    os.mkdir(output_path)

ext = ".obj"
for filename in os.listdir(mesh_dir_path):
    if ext in filename:
        mesh_name = '.'.join(filename.split('.')[:-1])
        pySurfaceHierarchy.hierarchy(mesh_dir_path, mesh_name + ext, output_path, top_level_size, level_size_factor, level_num, sym_N)
        pySurfaceHierarchy.conv_para(output_path, mesh_name, start_level, level_num, grid_length, grid_size)

