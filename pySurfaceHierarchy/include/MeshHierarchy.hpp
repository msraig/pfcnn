#pragma once

#include "HierarchyBuilder.hpp"
#include "TriMeshDef.hpp"
#include "SurfaceHierarchySerializer.hpp"

namespace surface_hierarchy {

	class MeshHierarchy : public SurfaceHierarchyBuilder{
		
		typedef SurfaceHierarchyBuilder Base;

	public:
		MeshHierarchy(const SurfaceHierarchyOptions& option) : Base(option){}

		void build(const char* filename);
		void write(const char* filename);
		template <typename T>
		void get_serializer(SurfaceHierarchySerializer<T>& srlz);

	private:
		std::vector<std::shared_ptr<TriMesh>> meshes;

		// openmesh properties
		std::vector<OpenMesh::VPropHandleT<std::vector<TriMesh::Point>>> omp_frames;
		std::vector<OpenMesh::VPropHandleT<std::vector<int>>> omp_covered_vts;
		std::vector<OpenMesh::VPropHandleT<double>> omp_vertex_areas;

	protected:
		void QEM_Decimation( std::shared_ptr<const TriMesh> fine_mesh,
			std::shared_ptr<TriMesh> simp_mesh, int target_vnum, 
			OpenMesh::VPropHandleT<std::vector<int>>& prop_handle);
		void PrincipalFrames(std::shared_ptr<TriMesh> mesh, 
			OpenMesh::VPropHandleT<std::vector<TriMesh::Point>>& prop_handle,
			std::vector<TriMesh::Point>& init_fx);
		void VertexWeights(std::shared_ptr<TriMesh> mesh,
			OpenMesh::VPropHandleT<double>& prop_handle);
	};

}