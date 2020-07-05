#pragma once

#include <cmath>
#include <vector>
#include "TNT\tnt_matrix.h"
#include "TriMeshDef.hpp"


inline int get_axis_map_p2p(const TriMesh::Point& centN, const TriMesh::Point& nbN, int anum,
	const std::vector<TriMesh::Point>& centAxes, const std::vector<TriMesh::Point>& nbAxes) {
	// rotation mat from nbN to centN
	TNT::Matrix<double> rot_mat(3, 3, 0.0);
	{
		//set Identity rotation
		rot_mat[0][0] = rot_mat[1][1] = rot_mat[2][2] = 1.0;

		auto prdct = OpenMesh::cross(nbN, centN);
		double sin_theta = prdct.length();
		if (!is_valid(sin_theta)) {
			std::cout << "Cannot rotate and match two normal vectors" << std::endl << nbN << std::endl << centN;	//std::exit(0); 
		}
		else if (sin_theta == 0.0)
		{ // parallel, do nothing
		}
		else {
			auto rot_axis = (1.0 / sin_theta) * prdct;
			auto cos_theta = OpenMesh::dot(nbN, centN);
			TNT::Matrix<double> K(3, 3);
			K[0][1] = -rot_axis[2]; K[1][0] = -K[0][1];
			K[0][2] = rot_axis[1]; K[2][0] = -K[0][2];
			K[1][2] = -rot_axis[0]; K[2][1] = -K[1][2];
			auto K2 = K*K;
			rot_mat = rot_mat + TNT::mult(sin_theta, K) + TNT::mult(1 - cos_theta, K2);
		}
	}

	auto mult_tntmat_vec3 = [](TNT::Matrix<double>& mat, TriMesh::Point& v)->TriMesh::Point {
		TriMesh::Point ret;
		for (int i = 0; i < 3; i++)
		{
			ret[i] = mat[i][0] * v[0] + mat[i][1] * v[1] + mat[i][2] * v[2];
		}
		return ret;
	};

	// search permutation
	auto centAxis = centAxes[0];
	int cls_nb_axis = -1;
	double cls_max = -10;
	for (int nba_itr = 0; nba_itr < anum; nba_itr++) {
		auto nb_axis = nbAxes[nba_itr];
		double dpt = dot(centAxis, mult_tntmat_vec3(rot_mat, nb_axis));
		if (dpt > cls_max)
		{
			cls_max = dpt; cls_nb_axis = nba_itr;
		}
	}
	return cls_nb_axis;
}
/*
inline void get_axis_map(TriMesh* mesh, const std::vector<std::vector<TriMesh::Point>>& axes,
	std::map<std::pair<int, int>, int>& edge_permute) {
	for (auto vh : *mesh->get_vertices_list())
	{
		auto eh = vh->edge;
		do
		{
			auto nvh = eh->vert;
			if (vh->id < nvh->id)
			{
				// rotation mat from nbN to centN
				TNT::Matrix<double> rot_mat(3, 3, 0.0);
				{
					//set Identity rotation
					rot_mat[0][0] = rot_mat[1][1] = rot_mat[2][2] = 1.0;

					TriMesh::Point centN = vh->normal, nbN = nvh->normal;
					auto prdct = Geex::cross(nbN, centN);
					double sin_theta = prdct.length();
					if (!is_valid(sin_theta)) { std::cout << "Cannot rotate and match two normal vectors";	std::exit(0); }

					if (sin_theta == 0.0)
					{ // parallel, do nothing
					}
					else {
						auto rot_axis = (1.0 / sin_theta) * prdct;
						auto cos_theta = Geex::dot(nbN, centN);
						TNT::Matrix<double> K(3, 3);
						K[0][1] = -rot_axis[2]; K[1][0] = -K[0][1];
						K[0][2] = rot_axis[1]; K[2][0] = -K[0][2];
						K[1][2] = -rot_axis[0]; K[2][1] = -K[1][2];
						auto K2 = K*K;
						rot_mat = rot_mat + TNT::mult(sin_theta, K) + TNT::mult(1 - cos_theta, K2);
					}
				}

				auto mult_tntmat_vec3 = [](TNT::Matrix<double>& mat, TriMesh::Point& v)->TriMesh::Point {
					TriMesh::Point ret;
					for (int i = 0; i < 3; i++)
					{
						ret[i] = mat[i][0] * v[0] + mat[i][1] * v[1] + mat[i][2] * v[2];
					}
					return ret;
				};

				// search permutation
				{
					auto centAxis = axes[vh->id][0];
					int anum = axes[0].size();
					int cls_nb_axis = -1;
					double cls_max = -10;
					for (int nba_itr = 0; nba_itr < anum; nba_itr++) {
						auto nb_axis = axes[nvh->id][nba_itr];
						double dpt = dot(centAxis, mult_tntmat_vec3(rot_mat, nb_axis));
						if (dpt > cls_max)
						{
							cls_max = dpt; cls_nb_axis = nba_itr;
						}
					}
					edge_permute[std::pair<int, int>(vh->id, nvh->id)] = cls_nb_axis;
				}
			}
			eh = eh->pair->next;
		} while (eh != vh->edge);
	}
}
*/