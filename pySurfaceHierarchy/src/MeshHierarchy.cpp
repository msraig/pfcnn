#include "MeshHierarchy.hpp"
#include "QEM_Decimator.hpp"
#include "DirectionFields.hpp"
#include "SurfaceHierarchySerializer.hpp"

namespace surface_hierarchy {

	void MeshHierarchy::build(const char* filename)
	{
		// read input mesh
		auto mesh = std::make_shared<TriMesh>();
		if (!OpenMesh::IO::read_mesh(*mesh, filename))
		{
			std::cerr << "Error: cannot read mesh file " << filename << std::endl;
			return;
		}
		if (options_.is_normalize)
		{
			if (options_.normalize_radius == -1) normalize(*mesh);
			else normalize_given_r(*mesh, options_.normalize_radius);
		}
		
		//faust template
		//normalize_given_r(*mesh, 2.0360705974315625);
		//faust scan
		//normalize_given_r(*mesh, 2.030551050811844);
		//scape
		//normalize_given_r(*mesh, 2.0340627364310744);
		int vnum = mesh->n_vertices();

		// determine the target vertex number for different levels
		std::vector<int> level_sizes;
		if (!Base::options_.get_level_sizes(vnum, level_sizes))
			return;
		//std::cout << level_sizes[0] << " " << level_sizes[1] << " " << level_sizes[2] << std::endl;
		// mesh simplification
		omp_covered_vts.resize(Base::options_.level_num);
		if (level_sizes[0] == vnum)
		{
			mesh->add_property(omp_covered_vts[0], "CoveredVertices");
			meshes.push_back(mesh);
		}
		else
		{
			auto mesh_simp = std::make_shared<TriMesh>();
			QEM_Decimation(mesh, mesh_simp, level_sizes[0], omp_covered_vts[0]);
			meshes.push_back(mesh_simp);
		}

		for (int i = 1; i < level_sizes.size(); i++)
		{
			auto mesh_simp = std::make_shared<TriMesh>();
			QEM_Decimation(meshes.back(), mesh_simp, level_sizes[i], omp_covered_vts[i]);
			meshes.push_back(mesh_simp);
		}
		if (Base::options_.not_build_frames)
		{
			return;
		}
		// build principal frames
		omp_frames.resize(Base::options_.level_num);
		std::vector<TriMesh::Point> init_FX;
		for (int litr = Base::options_.level_num-1; litr>-1; litr--)
		{
			if (litr == Base::options_.level_num-1)
			{
				init_FX.resize(meshes[litr]->n_vertices(),TriMesh::Point(0,0,0));
			}
			else
			{
				//copy from last mesh
				init_FX.resize(meshes[litr]->n_vertices());
				auto parent_mesh = meshes[litr + 1];
				auto& chld_prop = omp_covered_vts[litr + 1];
				auto& frame_prop = omp_frames[litr + 1];
				for (auto vitr = parent_mesh->vertices_begin();
					vitr!= parent_mesh->vertices_end(); vitr++)
				{
					auto& chld_list = parent_mesh->property(chld_prop, *vitr);
					for (auto chld_idx : chld_list)
					{
						init_FX[chld_idx] = parent_mesh->property(frame_prop, *vitr)[1]; //pd max
					}
				}
			}
			PrincipalFrames(meshes[litr], omp_frames[litr], init_FX);
		}

		// build vertex areas
		omp_vertex_areas.resize(Base::options_.level_num);
		for (int litr = 0; litr < meshes.size(); litr++)
		{
			VertexWeights(meshes[litr], omp_vertex_areas[litr]);
		}

		// Sanity checking
		for (int level=0; level< meshes.size(); level++)
		{
			auto pmesh = meshes[level];
			for (auto vitr=pmesh->vertices_begin(); vitr!=pmesh->vertices_end(); vitr++)
			{
				const auto& vframe = pmesh->property(omp_frames[level], *vitr);
				if (!is_valid(vframe[0]) ||
					!is_valid(vframe[1]) ||
					!is_valid(pmesh->point(*vitr))
					)
				{
					std::cout << " Invalid mesh data generated. " << std::endl;
					std::exit(0);
				}
			}
		}

		//std::cout << "Finished building surface hierarchy.\n";
	}

	void MeshHierarchy::write(const char* filename)
	{
		// for debug,
		// write each mesh into seperate files
		for (int i = 0; i < meshes.size(); i++)
		{
			std::string base_fn = std::string(filename)
				+ "_" + std::to_string(i);
			auto mesh = meshes[i];

			std::cout << "begin write mesh" << std::endl;
			OpenMesh::IO::write_mesh(*mesh, base_fn+".obj");
			if (!Base::options_.not_build_frames)
			{
				std::ofstream ofile(base_fn + ".frames");
				ofile << mesh->n_vertices() << " " << Base::options_.fields_option.sym_N << std::endl;
				for (auto vitr = mesh->vertices_begin(); vitr != mesh->vertices_end(); vitr++)
				{
					auto vh = *vitr;
					// two lines per vertex
					for (auto frame_axis : mesh->property(omp_frames[i], vh))
					{
						ofile << frame_axis[0] << " " << frame_axis[1] << " " << frame_axis[2] << std::endl;
					}
				}
				ofile.close();
			}
		}

		//// use hierarchy serializer to format output

		//SurfaceHierarchySerializer<double> sh_srlzr;
		//for (size_t i = 0; i < meshes.size(); i++)
		//{
		//	auto& mesh = meshes[i];
		//	SurfaceSerializer<double> sf_srlzr;
		//	for (size_t vitr = 0; vitr < mesh->n_vertices(); vitr++)
		//	{
		//		auto& vh = mesh->vertex_handle(vitr);
		//		// point position
		//		auto vpos = mesh->point(vh);
		//		for (size_t coord = 0; coord < 3; coord++)
		//		{
		//			sf_srlzr.pts.push_back(vpos[coord]);
		//		}
		//		// frame
		//		auto& vframe = mesh->property(omp_frames[i], vh);
		//		for (size_t frm_itr = 0; frm_itr < vframe.size(); frm_itr++)
		//		{
		//			sf_srlzr.frames.push_back(vframe[frm_itr][0]);
		//			sf_srlzr.frames.push_back(vframe[frm_itr][1]);
		//			sf_srlzr.frames.push_back(vframe[frm_itr][2]);
		//		}
		//		// covered vertex index
		//		auto& v_cvtx = mesh->property(omp_covered_vts[i], vh);
		//		sf_srlzr.pts_covered.push_back(v_cvtx);
		//	}
		//	// faces
		//	for (auto fitr = mesh->faces_begin(); fitr != mesh->faces_end(); fitr++)
		//	{
		//		auto& fh = *fitr;
		//		std::vector<int> fvs;
		//		for (auto fvitr = mesh->fv_begin(fh); fvitr != mesh->fv_end(fh); fvitr++)
		//		{
		//			fvs.push_back(fvitr->idx());
		//		}
		//		sf_srlzr.faces.push_back(fvs);
		//	}
		//
		//	sh_srlzr.surfaces.push_back(sf_srlzr);
		//}
		//
		//// write to file
		//std::ofstream ofile(filename, std::ios::binary);
		//std::vector<char> databuf(sh_srlzr.datasize(), 0);
		//sh_srlzr.serialize(databuf.data());
		//ofile.write(databuf.data(), databuf.size());
		//ofile.close();
		for (size_t i = 1; i < meshes.size(); i++)
		{
			std::ofstream ofile(std::string(filename) + "_" + std::to_string(i) + ".hrch");
			auto& mesh = meshes[i];
			ofile << mesh->n_vertices() << std::endl;
			for (size_t vitr = 0; vitr < mesh->n_vertices(); vitr++)
			{
				auto& vh = mesh->vertex_handle(vitr);
				// covered vertex index
				auto& v_cvtx = mesh->property(omp_covered_vts[i], vh);
				for (size_t v_index = 0; v_index < v_cvtx.size(); v_index++)
				{
					ofile << v_cvtx[v_index] << " ";
				}
				ofile << std::endl;
			}
			ofile.close();
		}
	}

	template<typename T>
	void MeshHierarchy::get_serializer(SurfaceHierarchySerializer<T>& srlz)
	{
		srlz.surfaces.clear();
		for (size_t i = 0; i < meshes.size(); i++)
		{
			auto& mesh = meshes[i];
			SurfaceSerializer<T> sf_srlzr;
			sf_srlzr.frame_sym_N = Base::options_.fields_option.sym_N;
			for (size_t vitr = 0; vitr < mesh->n_vertices(); vitr++)
			{
				auto& vh = mesh->vertex_handle(vitr);
				// point position
				auto vpos = mesh->point(vh);
				for (size_t coord = 0; coord < 3; coord++)
				{
					sf_srlzr.pts.push_back((T)vpos[coord]);
				}
				// frame
				auto& vframe = mesh->property(omp_frames[i], vh);
				for (size_t frm_itr = 0; frm_itr < vframe.size(); frm_itr++)
				{
					sf_srlzr.frames.push_back((T)vframe[frm_itr][0]);
					sf_srlzr.frames.push_back((T)vframe[frm_itr][1]);
					sf_srlzr.frames.push_back((T)vframe[frm_itr][2]);
				}
				// weight
				sf_srlzr.weights.push_back((T)mesh->property(omp_vertex_areas[i], vh));
				// covered vertex index
				auto& v_cvtx = mesh->property(omp_covered_vts[i], vh);
				sf_srlzr.pts_covered.push_back(v_cvtx);

				// neighbor vertices
				std::vector<int> nbs;
				for (auto vv_itr = mesh->vv_begin(vh); vv_itr != mesh->vv_end(vh); vv_itr++)
				{
					nbs.push_back(vv_itr->idx());
				}
				sf_srlzr.nb_vts.push_back(nbs);
			}

			//// faces
			//for (auto fitr=mesh->faces_begin();fitr!=mesh->faces_end();fitr++)
			//{
			//	auto& fh = *fitr;
			//	std::vector<int> fvs;
			//	for (auto fvitr = mesh->fv_begin(fh); fvitr != mesh->fv_end(fh); fvitr++)
			//	{
			//		fvs.push_back(fvitr->idx());
			//	}
			//	sf_srlzr.faces.push_back(fvs);
			//}

			srlz.surfaces.push_back(sf_srlzr);
		}
	}


	///////////////////////
	// The QEM algorithm to simplify mesh to a target vertex number
	void MeshHierarchy::QEM_Decimation(std::shared_ptr<const TriMesh> fine_mesh,
		std::shared_ptr<TriMesh> simp_mesh, int target_vnum, 
		OpenMesh::VPropHandleT<std::vector<int>>& prop_handle) {
			
		for (auto vitr = fine_mesh->vertices_begin(); vitr != fine_mesh->vertices_end(); vitr++)
			simp_mesh->add_vertex(fine_mesh->point(*vitr));

		for (auto fitr = fine_mesh->faces_begin(); fitr != fine_mesh->faces_end(); fitr++)
		{
			auto fh = *fitr;
			std::vector<TriMesh::VertexHandle> face_vts;
			for (auto fvitr=fine_mesh->cfv_begin(fh); fvitr!=fine_mesh->cfv_end(fh); fvitr++)
			{
				face_vts.push_back(simp_mesh->vertex_handle(fvitr->idx()));
			}
			simp_mesh->add_face(face_vts);
		}

		QEM_Decimator decimator(simp_mesh, target_vnum, &prop_handle);
		decimator.decimate(Base::options_);
	}

	////////////////////////
	// Build frames for each vertex of the mesh.
	// Could utilize the mutli-scale surface hierarchy to accelerate the frame computation.
	// For now, we just compute the frames for each mesh seperately.
	void MeshHierarchy::PrincipalFrames(std::shared_ptr<TriMesh> mesh,
		OpenMesh::VPropHandleT<std::vector<TriMesh::Point>>& prop_handle,
		std::vector<TriMesh::Point>& init_fx) {

		mesh->request_face_normals();
		mesh->request_vertex_normals();
		mesh->update_normals();
		
		int vnum = mesh->n_vertices();
		FieldsOnSurface frames(vnum);

		// fill V,N,FX,Adj.
		for (size_t i = 0; i < vnum; i++)
		{
			auto vh = mesh->vertex_handle(i);
			frames.V[i] = mesh->point(vh);
			frames.N[i] = mesh->normal(vh);
			if (frames.N[i].norm()<1e-6)
			{
				frames.N[i] = TriMesh::Point(1, 0, 0);
			}
			frames.FX[i] = init_fx[i];

			for (auto nbitr=mesh->vv_begin(vh); nbitr!=mesh->vv_end(vh); nbitr++)
			{
				frames.Adj[i].push_back(nbitr->idx());
			}
		}
		// solve field
		FieldsOptions field_option = Base::options_.fields_option;
		//field_option.field_computation_method = FieldsOptions::POLYVECTOR; // FieldsOptions::RAW_CURVATURE_TENSOR;
		if (options_.IsSmooth)
		{
			field_option.field_computation_method = FieldsOptions::POLYVECTOR;
			if (options_.Smooth_only)
			{
				field_option.smooth_only = true;
			}
			else
			{
				field_option.smooth_only = false;
				field_option.lambda = options_.poly_lambda;
			}
		}
		else
		{
			field_option.field_computation_method = FieldsOptions::RAW_CURVATURE_TENSOR;
		}
		frames.solve(field_option);
		/*FieldsOnMesh frames(mesh, field_option, &prop_handle);
		frames.build();*/

		// copy to mesh
		mesh->add_property(prop_handle, "Frames");
		for (size_t i = 0; i < vnum; i++)
		{
			auto vh = mesh->vertex_handle(i);
			auto& vframe = mesh->property(prop_handle, vh);
			vframe.push_back(frames.N[i]);
			vframe.push_back(frames.FX[i]);
		}
	}

	void MeshHierarchy::VertexWeights(std::shared_ptr<TriMesh> mesh,
		OpenMesh::VPropHandleT<double>& prop_handle) {

		mesh->add_property(prop_handle, "VertexArea");

		for (auto vitr = mesh->vertices_begin(); vitr != mesh->vertices_end(); vitr++)
		{
			mesh->property(prop_handle, *vitr) = 0.0;
		}

		for (auto fitr=mesh->faces_begin(); fitr!=mesh->faces_end(); fitr++)
		{
			auto fh = *fitr;
			// assume triangle mesh
			TriMesh::Point pt[3];
			auto vitr = mesh->fv_begin(fh);
			pt[0] = mesh->point(*vitr); vitr++;
			pt[1] = mesh->point(*vitr); vitr++;
			pt[2] = mesh->point(*vitr);
			double area = OpenMesh::cross(pt[1] - pt[0], pt[2] - pt[0]).norm()/6;

			vitr = mesh->fv_begin(fh);
			mesh->property(prop_handle, *vitr) += area; vitr++;
			mesh->property(prop_handle, *vitr) += area; vitr++;
			mesh->property(prop_handle, *vitr) += area; 
		}

		for (auto vitr = mesh->vertices_begin(); vitr != mesh->vertices_end(); vitr++)
		{
			if (mesh->property(prop_handle, *vitr) < 1e-7) { // hanging vertex
				mesh->property(prop_handle, *vitr) = 1.0;
			}
		}
	}

	template void MeshHierarchy::get_serializer(SurfaceHierarchySerializer<float>& srlz);
}

