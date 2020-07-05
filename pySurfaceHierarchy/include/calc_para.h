#include <ostream>
#include <iostream>
#include "TangentParam/tangent_param.h"
#include "TriMeshDef.hpp"
#include "io.h"
#include "HierarchyBuilder.hpp"
#include "TriMeshDef.hpp"
#include "TNT/tnt_matrix.h"

using namespace surface_hierarchy;

struct VertexFrame {
	int vid;
	std::vector<TriMesh::Point> axis;
	TriMesh::Point pos;

	VertexFrame() {
		vid = -1;
		axis.resize(0);
	}

};

void store_tan_para(std::string dataset_path, std::string ext, int strat_level, int level_num, double *grid_length);
void store_tan_para_from_pc(std::string dataset_path, std::string ext, int strat_level, int level_num, double *grid_length);
void store_tan_para_singlefile(std::string mesh_dir_path, std::string filename, int start_level, int level_num, double *grid_length, int grid_size);