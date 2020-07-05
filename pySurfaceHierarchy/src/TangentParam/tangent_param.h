#ifndef _TANGENT_PARAM_
#define _TANGENT_PARAM_

#include <vector>
#include "TriMeshDef.hpp"
#include "Geex\graphics\opengl.h"

using namespace surface_hierarchy;
// Defines how to map the neighborhood of a vertex to the tangent space, and
//  sample the tangent space with regular grid that is coupled with the neighboring
//  vertices.
// Input: 
//	mesh - surface mesh,
//	axes - axis for each vertex, 
//	scale - side length of the sampling grid
//	gm,gn - resolution of the sampling grid
// Output:
//	vtx_indices - an integer tensor of shape V*A*(gm*gn)*3, indexing the 3 vertices for each grid point
//	axis_indices - an integer tensor of the same shape as vtx_indices, indexing the axis of patch vts
//	baryweights - a double tensor with the same shape of indices, giving the barycentric weights


// Does local patch parameterization and regular grid sampling, as well as patch-wise input data generation.
// The interface is similar to tangent_param(), with additional output patch feature vector.
// The implementation is different from tangent_param(), with improvements in:
//  1. Out-of-patch grid points will be masked out with zero bary-weights.
void tangent_param_with_patchinput(TriMesh* mesh, const std::vector<std::vector<Geex::vec3>>& axes,
	const double scale, const int gm, const int gn, const bool use_patch_height,
	std::vector<int>& vtx_indices, std::vector<int>& axis_indices, std::vector<double>& baryweights,
	std::vector<double>& patch_input_feature);


// Demo interface 
class TangentParamVis {
public:
	TriMesh* mesh=NULL;
	std::vector<std::vector<TriMesh::Point>> axes;
	std::vector<TriMesh::Point> pointcloud;
	std::vector<TriMesh::Point> pc_normal;

	bool is_use_patch_height = true;
	bool is_reweighted = false;
	int marked_pt=0;
private:
	std::vector<int> indices, axis_indices;
	std::vector<double> baryweights;
	std::vector<double> output_features;
	int gm_, gn_;

public:
	void build_grid_samples(double scale, int gm, int gn);
	void build_grid_samples_from_pc(double scale, int gm, int gn);
	void write_weights(std::string path)
	{
		std::ofstream fout(path, std::ios::binary);
		/*
		fout << mesh->get_num_of_vertices() << " " << axes[0].size() << " " << gm_ << " " << gn_ << std::endl;
		for (int i = 0; i * 3 < baryweights.size(); i++)
		{
		for (int j = 0; j < 3; j++)
		{
		fout << baryweights[i * 3 + j] << " ";
		}
		fout << std::endl;
		}
		*/
		int vert_num = axes.size();
		int axis_num = axes[0].size();
		fout.write((const char*)&vert_num, sizeof(int));
		fout.write((const char*)&axis_num, sizeof(int));
		fout.write((const char*)&gm_, sizeof(int));
		fout.write((const char*)&gn_, sizeof(int));
		fout.write((const char*)&indices[0], indices.size() * sizeof(int));
		fout.write((const char*)&axis_indices[0], axis_indices.size() * sizeof(int));
		fout.write((const char*)&baryweights[0], baryweights.size() * sizeof(double));
		fout.close();
	}

	void write_input_feature(std::string path)
	{
		std::ofstream fout(path, std::ios::binary);
		int vert_num = axes.size();
		int axis_num = axes[0].size();
		fout.write((const char*)&vert_num, sizeof(int));
		fout.write((const char*)&axis_num, sizeof(int));
		fout.write((const char*)&gm_, sizeof(int));
		fout.write((const char*)&gn_, sizeof(int));
		int feature_channel = is_use_patch_height ? 4 : 3;
		fout.write((const char*)&feature_channel, sizeof(int));
		fout.write((const char*)&output_features[0], output_features.size() * sizeof(double));
		fout.close();
	}
};

#endif