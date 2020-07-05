#pragma once

#include "TriMeshDef.hpp"
#include "Wm5Matrix3.h"
#include "HierarchyBuilder.hpp"

namespace surface_hierarchy {

	class QEM_Decimator {

	public:
		QEM_Decimator(std::shared_ptr<TriMesh> mesh, int target_vnum,
			OpenMesh::VPropHandleT<std::vector<int>>* omp_covered_vts) {
			pmesh_ = mesh;
			target_vnum_ = target_vnum;
			covered_vts_vph_ = omp_covered_vts;
			record_vtx_cover_ = true;
		}
		QEM_Decimator(std::shared_ptr<TriMesh> mesh, int target_vnum) {
			pmesh_ = mesh;
			target_vnum_ = target_vnum;
			record_vtx_cover_ = false;
		}
		void decimate(const SurfaceHierarchyOptions& options);

	protected:

		class Mat4d {
		public:
			Mat4d() {}
			Mat4d(OpenMesh::Vec3d pt, OpenMesh::Vec3d normal) {
				normal.normalize();
				double plane_eq[4] = { normal[0],normal[1],normal[2], -OpenMesh::dot(normal, pt) };
				for (size_t i = 0; i < 4; i++)
				{
					for (size_t j = 0; j < 4; j++) {
						mat[i][j] = plane_eq[i] * plane_eq[j];
					}
				}

				// closeness to point position quadric
				double closeness_weight = 0.1;// 0.1;
				for (size_t i = 0; i < 3; i++)
				{
					mat[i][i] += closeness_weight;
					mat[i][3] += -1 * closeness_weight*pt[i];
					mat[3][i] += -1 * closeness_weight*pt[i];
				}
				mat[3][3] += closeness_weight*pt.sqrnorm();
			}
			Mat4d operator+(const Mat4d& other) const {
				Mat4d new_mat;
				for (size_t i = 0; i < 4; i++)
				{
					for (size_t j = 0; j < 4; j++) {
						new_mat.mat[i][j] = this->mat[i][j] + other.mat[i][j];
					}
				}
				return new_mat;
			}
			double mult(const OpenMesh::Vec3d& v) {
				double homo_v[4] = { v[0],v[1],v[2],1 };
				double quadric = 0;
				for (size_t i = 0; i < 4; i++)
				{
					double row = 0;
					for (size_t j = 0; j < 4; j++)
					{
						row += mat[i][j] * homo_v[j];
					}
					quadric += homo_v[i] * row;
				}
				return quadric;
			}
			OpenMesh::Vec3d optimal_pt() {
				Wm5::Matrix3d A;
				Wm5::Vector3d b(-mat[0][3],-mat[1][3],-mat[2][3]);
				for (size_t i = 0; i < 3; i++)
				{
					for (size_t j = 0; j < 3; j++)
					{
						A[i][j] = mat[i][j];
					}
				}
				auto opt = A.Inverse()*b;
				if (opt!=A.ZERO)
				{
					return OpenMesh::Vec3d(opt[0], opt[1], opt[2]);
				}
				else
				{
					std::cout << "Mat not invertible.\n";
					return OpenMesh::Vec3d(0, 0, 0);
				}
				
			}
		protected:
			double mat[4][4] = { 0. };
		};

		// properties for QEM computation
		OpenMesh::VPropHandleT<Mat4d> omp_qmat;
		

		double evaluate_edge_error(const TriMesh::EdgeHandle& eh, 
			 std::shared_ptr<const TriMesh> mesh);

	private:
		std::shared_ptr<TriMesh> pmesh_;
		int target_vnum_;
		OpenMesh::VPropHandleT<std::vector<int>>* covered_vts_vph_;
		bool record_vtx_cover_;
	};




}

