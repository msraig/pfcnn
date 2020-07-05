#pragma once
#include "ANN/ANN.h"
#include <vector>
#include "TriMeshDef.hpp"

namespace surface_hierarchy{

	class ANN_Search {

	public:

	ANN_Search(const std::vector<TriMesh::Point>& pts) {
		data_pts = annAllocPts(pts.size(), 3);
		for (size_t i = 0; i < pts.size(); i++)
		{
			data_pts[i][0] = pts[i][0];
			data_pts[i][1] = pts[i][1];
			data_pts[i][2] = pts[i][2];
		}
		kd_tree_ = new ANNkd_tree(data_pts, pts.size(), 3); //bd_tree
	}
	~ANN_Search() {
		if (kd_tree_)
		{
			delete kd_tree_;
		}
		annDeallocPts(data_pts);
	}
	int project_pt(TriMesh::Point pt, TriMesh::Point& foot) {

		//std::cout << " Projecting pt\n";
		std::vector<int> nbs;
		std::vector<double> nb_dist;
		find_k_nbs(pt, 1, nbs, nb_dist);

		foot[0] = data_pts[nbs[0]][0];
		foot[1] = data_pts[nbs[0]][1];
		foot[2] = data_pts[nbs[0]][2];
		return nbs[0];
	}

	void find_k_nbs(TriMesh::Point pt, int K, 
		std::vector<int>& nb_ids, std::vector<double>& nb_dist) {
		ANNpoint		queryPt;				// query point
		std::vector<int> nnIdx(K);								// allocate near neighbor indices
		std::vector<double> dists(K);							// allocate near neighbor dists

		queryPt = annAllocPt(3);
		queryPt[0] = pt[0];
		queryPt[1] = pt[1];
		queryPt[2] = pt[2];

		//std::cout << " Searching tree" << std::endl;

		kd_tree_->annkSearch(					// search
			queryPt, 						// query point
			K, 								// number of near neighbors
			&nnIdx[0], 						// nearest neighbors (returned)
			&dists[0], 						// distance (returned)
			0.0);							// error bound

		annDeallocPt(queryPt);

		nb_ids = nnIdx;
		nb_dist = dists;
	}

	void find_pts_in_radius(TriMesh::Point pt, double radius, 
		std::vector<int>& nb_ids) {

		int K = 36;
		ANNpoint		queryPt;				// query point
		std::vector<int> nnIdx(K);								// allocate near neighbor indices
		std::vector<double> dists(K);							// allocate near neighbor dists

		queryPt = annAllocPt(3);
		queryPt[0] = pt[0];
		queryPt[1] = pt[1];
		queryPt[2] = pt[2];

		int vnum_in_r = kd_tree_->annkFRSearch(					// approx fixed-radius kNN search
			queryPt,				// the query point
			radius*radius,			// squared radius of query ball
			K,				// number of neighbors to return
			&nnIdx[0],	// nearest neighbor array (modified)
			&dists[0]		// dist to near neighbors (modified)
		);		// error bound

		nb_ids = std::vector<int>(nnIdx.begin(), nnIdx.begin() + std::min(K, vnum_in_r));
	}

	ANNkd_tree* kd_tree_ = NULL;
	ANNpointArray data_pts;
};
}
