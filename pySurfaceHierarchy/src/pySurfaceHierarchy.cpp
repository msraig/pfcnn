// pySurfaceHierarchy.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "MeshHierarchy.hpp"
#include "calc_para.h"

namespace py = pybind11;
using namespace surface_hierarchy;

void hierarchy(std::string mesh_path, std::string mesh_name, std::string output_path, int top_level_size, double level_size_factor, int level_num, int sym_N)
{
	std::string in_path = mesh_path + "/" + mesh_name;
	std::string out_path = output_path + "/" + mesh_name.substr(0, mesh_name.find("."));;

	SurfaceHierarchyOptions options;
	if (top_level_size > 0)options.top_level_size = top_level_size;
	options.level_size_factor = level_size_factor;
	options.level_num = level_num;
	options.fields_option.sym_N = sym_N;
	options.normalize_radius = -1;
	MeshHierarchy mesh_hircy(options);
	mesh_hircy.build(in_path.c_str());
	mesh_hircy.write(out_path.c_str());
}

void conv_para(std::string mesh_dir_path, std::string filename, int start_level, int level_num, py::array_t<double> grid_length, int grid_size)
{
	py::buffer_info buff = grid_length.request();
	double *ptr = (double *)buff.ptr;
	double *grid_length_array = new double[level_num];
	for (int i = 0; i < level_num; i++)
	{
		grid_length_array[i] = ptr[i];
	}
	store_tan_para_singlefile(mesh_dir_path, filename, start_level, level_num, grid_length_array, grid_size);
}

PYBIND11_MODULE(pySurfaceHierarchy, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring

	m.def("hierarchy", &hierarchy, "write hierarchy mesh and frames");
	m.def("conv_para", &conv_para, "write convolution parameters");

}
