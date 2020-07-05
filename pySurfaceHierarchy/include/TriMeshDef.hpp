#pragma once

#include "OpenMesh\Core\IO\MeshIO.hh"
#include "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"

namespace surface_hierarchy {
	// Select mesh type (TriMesh) and kernel (ArrayKernel)
	// and define my personal mesh type (MyMesh)
	struct DPTraits : public OpenMesh::DefaultTraits
	{
		typedef OpenMesh::Vec3d Point; // use double-values points/normals
		typedef OpenMesh::Vec3d Normal;
	};
	typedef OpenMesh::TriMesh_ArrayKernelT<DPTraits>  TriMesh;

	inline bool is_valid(double val) {
		if (std::isnan(val) || std::isinf(val))
		{
			return false;
		}
		return true;
	}
	inline bool is_valid(TriMesh::Point pt) {
		return is_valid(pt[0]) && is_valid(pt[1]) && is_valid(pt[2]);
	}

	inline void normalize(TriMesh& mesh) {
		double bbox[6] = { 1e10,1e10,1e10,-1e10,-1e10,-1e10 };
		for (auto vitr = mesh.vertices_begin(); vitr != mesh.vertices_end(); vitr++)
		{
			auto pt = mesh.point(*vitr);
			bbox[0] = std::min(bbox[0], pt[0]);
			bbox[1] = std::min(bbox[1], pt[1]);
			bbox[2] = std::min(bbox[2], pt[2]);
			bbox[3] = std::max(bbox[3], pt[0]);
			bbox[4] = std::max(bbox[4], pt[1]);
			bbox[5] = std::max(bbox[5], pt[2]);
		}
		TriMesh::Point cent(0.5*(bbox[0] + bbox[3]), 0.5*(bbox[1] + bbox[4]), 0.5*(bbox[2] + bbox[5]));
		double radius = std::sqrt((bbox[3] - bbox[0])*(bbox[3] - bbox[0]) + (bbox[4] - bbox[1])*(bbox[4] - bbox[1])
			+ (bbox[5] - bbox[2])*(bbox[5] - bbox[2]));
		for (auto vitr = mesh.vertices_begin(); vitr != mesh.vertices_end(); vitr++)
		{
			auto& pt = mesh.point(*vitr);
			pt = (pt - cent) / radius;
		}
	}

	inline void normalize_given_r(TriMesh& mesh, double radius) {
		double bbox[6] = { 1e10,1e10,1e10,-1e10,-1e10,-1e10 };
		for (auto vitr = mesh.vertices_begin(); vitr != mesh.vertices_end(); vitr++)
		{
			auto pt = mesh.point(*vitr);
			bbox[0] = std::min(bbox[0], pt[0]);
			bbox[1] = std::min(bbox[1], pt[1]);
			bbox[2] = std::min(bbox[2], pt[2]);
			bbox[3] = std::max(bbox[3], pt[0]);
			bbox[4] = std::max(bbox[4], pt[1]);
			bbox[5] = std::max(bbox[5], pt[2]);
		}
		TriMesh::Point cent(0.5*(bbox[0] + bbox[3]), 0.5*(bbox[1] + bbox[4]), 0.5*(bbox[2] + bbox[5]));
		//double radius = std::sqrt((bbox[3] - bbox[0])*(bbox[3] - bbox[0]) + (bbox[4] - bbox[1])*(bbox[4] - bbox[1])
		//	+ (bbox[5] - bbox[2])*(bbox[5] - bbox[2]));
		for (auto vitr = mesh.vertices_begin(); vitr != mesh.vertices_end(); vitr++)
		{
			auto& pt = mesh.point(*vitr);
			pt = (pt - cent) / radius;
		}
	}


	inline double face_area(const TriMesh& mesh, TriMesh::FaceHandle face) {
		TriMesh::Point pt[3];
		auto vitr = mesh.cfv_begin(face);
		pt[0] = mesh.point(*vitr); vitr++;
		pt[1] = mesh.point(*vitr); vitr++;
		pt[2] = mesh.point(*vitr);
		return OpenMesh::cross(pt[1] - pt[0], pt[2] - pt[0]).norm() / 2.;
	}
}

