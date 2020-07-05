
#include "DirectionFields.hpp"
#include "Wm5MeshCurvature.h"
//#include "Eigen/CholmodSupport"

namespace surface_hierarchy {
	void FieldsOnMesh::build()
	{
		pmesh_->add_property(*frame_vph_, "Frames");
		pmesh_->request_face_normals();
		pmesh_->request_vertex_normals();
		pmesh_->update_normals();

		pmesh_->add_property(omp_crv_tensor_, "CurvatureTensor");
		

		if (option_.field_computation_method == FieldsOptions::RAW_CURVATURE_TENSOR)
		{
			build_raw_curvature_tensor();
		}
		else if (option_.field_computation_method == FieldsOptions::POLYVECTOR)
		{
			build_polyvector();
		}
		else
		{
			std::cerr << "Error: field computation method " 
				<< option_.field_computation_method << " not supported.";
		}

		pmesh_->remove_property(omp_crv_tensor_);
	}


	void FieldsOnMesh::build_raw_curvature_tensor()
	{
		compute_curvature_tensor();
		for (auto vitr=pmesh_->vertices_begin(); vitr!=pmesh_->vertices_end(); vitr++)
		{
			auto& vframe = pmesh_->property(*frame_vph_, *vitr);
			auto& vcrvt_tensor = pmesh_->property(omp_crv_tensor_, *vitr);
			vframe.push_back(pmesh_->normal(*vitr));
			vframe.push_back(vcrvt_tensor.pdmax);
			vframe.push_back(vcrvt_tensor.pdmin);

			// check right-handness of the frame.
			if (dot(OpenMesh::cross(vframe[0], vframe[1]), vframe[2]) < 0)
				vframe[2] *= -1;
		}
	}

	void FieldsOnMesh::build_polyvector()
	{

		compute_curvature_tensor();

		int vnum = pmesh_->n_vertices();

		//complex number based smooth direction field computation

		Eigen::SparseMatrix<double, Eigen::ColMajor> cmplx_smth_normal_mat(vnum * 2, vnum * 2);
		std::vector<double> rhs(vnum*2,0.0);
		typedef Eigen::Triplet<double> MatEntry;
		std::vector<MatEntry> mat_entries;

		//std::cout << "Smoothing direction field... " << std::endl;
		const double crv_dir_weight = 0.01;
		for (int vitr = 0; vitr < vnum; vitr++)
		{
			auto vh = pmesh_->vertex_handle(vitr);
			auto& crv_tensor = pmesh_->property(omp_crv_tensor_, vh);

			double abs_crv = std::tanh(std::abs(crv_tensor.kmin - crv_tensor.kmax));

			std::complex<double> vdir = complex_from_vec(vh, *pmesh_, crv_tensor.pdmax);
			if (std::isnan(vdir.real())||std::isnan(vdir.imag()))
			{
				std::cout << "Raw curvature direction is NaN!\n";
				continue;
			}
			vdir = std::pow(vdir, option_.sym_N);

			mat_entries.push_back(MatEntry(vitr * 2, vitr * 2, abs_crv*crv_dir_weight));
			rhs[vitr * 2] = abs_crv*crv_dir_weight*vdir.real();

			mat_entries.push_back(MatEntry(vitr * 2 + 1, vitr * 2 + 1, abs_crv*crv_dir_weight));
			rhs[vitr * 2 + 1] = abs_crv*crv_dir_weight*vdir.imag();
		}

		build_complex_based_laplacian_mat(mat_entries);
		cmplx_smth_normal_mat.setFromTriplets(mat_entries.begin(), mat_entries.end());
		//std::cout << "Finished building the smoothness matrix" << std::endl;

		Eigen::VectorXd sol;
		//{
		//	//Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		//	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;

		//	//Eigen::LDLT<Eigen::SparseMatrix<double>> solver;
		//	//Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>> solver;

		//	solver.compute(cmplx_smth_normal_mat);
		//	if (solver.info() != Eigen::Success)
		//	{
		//		std::cerr << "Error: Eigen LDLT factorization failed.";
		//		//return;
		//		std::exit(0);
		//	}
		//	Eigen::Map<Eigen::VectorXd> vrhs(rhs.data(), rhs.size());
		//	sol = solver.solve(vrhs);
		//	if (solver.info() != Eigen::Success)
		//	{
		//		std::cerr << "Error: Eigen solve failed.";
		//		//return;
		//		std::exit(0);
		//	}
		//}
		{
			Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg_solver;
			cg_solver.setMaxIterations(10000);
			std::cout << "\tCG solver begin...\n";
			cg_solver.compute(cmplx_smth_normal_mat);
			Eigen::Map<Eigen::VectorXd> vrhs(rhs.data(), rhs.size());
			sol = cg_solver.solve(vrhs);
			if (cg_solver.info() != Eigen::Success)
			{
				std::cerr << "Error: CG solve not converged.";
				//return;
				//std::exit(0);
			}
			std::cout << "\tCG #iterations:     " << cg_solver.iterations() << std::endl;
			std::cout << "\t   estimated error: " << cg_solver.error() << std::endl;
		}
		//{
		//	Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>, 
		//		Eigen::COLAMDOrdering<Eigen::SparseMatrix<double, Eigen::ColMajor>::StorageIndex>> solver;

		//	solver.analyzePattern(cmplx_smth_normal_mat);
		//	solver.factorize(cmplx_smth_normal_mat);
		//	if (solver.info() != Eigen::Success)
		//	{
		//		std::cerr << "Error: EigenLU factorization failed.";
		//		//return;
		//		std::exit(0);
		//	}
		//	Eigen::Map<Eigen::VectorXd> vrhs(rhs.data(), rhs.size());
		//	sol = solver.solve(vrhs);
		//	if (solver.info() != Eigen::Success)
		//	{
		//		std::cerr << "Error: EigenLU solve failed.";
		//		//return;
		//		std::exit(0);
		//	}
		//}


		//std::cout << "Finished solving the matrix" << std::endl;

		//normalize solution and scale with domain area
		//(used for generating scaled frames that guide conv-kernel sizes)
		double domain_area = 0.;
		{
			for (auto fitr=pmesh_->faces_begin(); fitr!=pmesh_->faces_end(); ++fitr)
			{
				std::vector<TriMesh::Point> pts;
				for (auto fvitr = pmesh_->fv_begin(*fitr); fvitr != pmesh_->fv_end(*fitr); ++fvitr)
					pts.push_back(pmesh_->point(*fvitr));
				TriMesh::Point cpt = (pts[1] - pts[0])%(pts[2] - pts[0]);
				domain_area += cpt.norm() * 0.5;
			}
		}
		double total_pow_norm = 0.;
		std::vector<std::complex<double>> sols(vnum);
		for (int vitr = 0; vitr < vnum; vitr++)
		{
			std::complex<double> u_pow(sol[vitr * 2], sol[vitr * 2 + 1]); //u_pow = std::pow(u_pow, 1.0 / option_.sym_N);
			total_pow_norm += std::norm(u_pow); 
			sols[vitr] = u_pow;
		}
		double scale_factor = domain_area / total_pow_norm;
		//read solutions
		for (int vitr = 0; vitr < vnum; vitr++) {
			auto u = sols[vitr];
			double scale = scale_factor*std::norm(u); scale = sqrt(scale);

			u = std::pow(u, 1.0 / option_.sym_N);

			if (std::norm(u) < 1e-7)
			{
				auto vh = pmesh_->vertex_handle(vitr);
				auto& frame = pmesh_->property(*frame_vph_, vh);
				TriMesh::Point xaxis, yaxis;
				get_ref_frame(pmesh_->normal(vh), xaxis, yaxis);
				frame.push_back(pmesh_->normal(vh));
				frame.push_back(scale*xaxis);
				frame.push_back(scale*yaxis);
			}
			else {
				u = u / std::sqrt(std::norm(u));
				// check validity
				if (std::isnan(u.real()) || std::isnan(u.imag()))
				{
					std::cout << "Solved polyvector is Nan! " << u.real() << " " << u.imag() << std::endl;

					std::cout << "Dumping solution...\n";
					for (int i = 0; i < 2 * vnum; i++)
					{
						std::cout << sol[i] << " ";
					}

					std::exit(0);
				}

				// store smoothed principle curvature
				auto vh = pmesh_->vertex_handle(vitr);
				auto p_dir1 = scale*vec_from_complex(vh, *pmesh_, u);
				auto p_dir2 = OpenMesh::cross(pmesh_->normal(vh), p_dir1);
				auto& frame = pmesh_->property(*frame_vph_, vh);
				frame.push_back(pmesh_->normal(vh));
				frame.push_back(p_dir1);
				frame.push_back(p_dir2);
			}
		}

		////read solutions
		//for (int vitr = 0; vitr < vnum; vitr++)
		//{
		//	std::complex<double> u_pow(sol[vitr * 2], sol[vitr * 2 + 1]);
		//	//std::cout << u_pow << std::endl;

		//	auto Nvec = pmesh_->normal(pmesh_->vertex_handle(vitr));
		//	if (std::isnan(Nvec[0]) || std::isnan(Nvec[1]) || std::isnan(Nvec[2]))
		//	{
		//		std::cout << " NaN for normal vector!\n";
		//		std::exit(0);
		//	}

		//	//if (std::norm(u_pow)<1e-7)
		//	//{
		//	//	auto vh = pmesh_->vertex_handle(vitr);
		//	//	auto& frame = pmesh_->property(*frame_vph_, vh);
		//	//	TriMesh::Point xaxis, yaxis;
		//	//	get_ref_frame(pmesh_->normal(vh), xaxis, yaxis);
		//	//	frame.push_back(pmesh_->normal(vh));
		//	//	frame.push_back(xaxis);
		//	//	frame.push_back(yaxis);
		//	//	continue;
		//	//}
		//	//else
		//	{
		//		u_pow /= std::sqrt(std::norm(u_pow));
		//		/*u_pow = std::sqrt(u_pow);
		//		u_pow = std::sqrt(u_pow);*/
		//		u_pow = std::pow(u_pow, 1.0 / option_.sym_N);

		//		// check validity
		//		if (std::isnan(u_pow.real()) || std::isnan(u_pow.imag()))
		//		{
		//			std::cout << "Solved polyvector is Nan! " << sol[vitr * 2] << " " << sol[vitr * 2 + 1] << std::endl;

		//			std::cout << "Dumping solution...\n";
		//			for (int i = 0; i < 2 * vnum; i++)
		//			{
		//				std::cout << sol[i] << " ";
		//			}

		//			std::exit(0);
		//		}

		//		// store smoothed principle curvature

		//		auto vh = pmesh_->vertex_handle(vitr);

		//		auto p_dir1 = vec_from_complex(vh, *pmesh_, u_pow);
		//		auto p_dir2 = OpenMesh::cross(pmesh_->normal(vh), p_dir1);
		//		//p_dir2.normalize();

		//		auto& frame = pmesh_->property(*frame_vph_, vh);
		//		frame.push_back(pmesh_->normal(vh));
		//		frame.push_back(p_dir1);
		//		frame.push_back(p_dir2);
		//	}
		//}

		//std::cout << "Finished solving the complex-based "
		//	"field smoothness problem." << std::endl;

	}


	void FieldsOnMesh::compute_curvature_tensor() {
		
		if (!pmesh_->is_trimesh())
		{
			std::cerr << "Error: curvature estimation not supported for"
				"meshes not triangular.\n";
			return;
		}

		//call GeometricTools for curvature tensor computation.
		int numVertices = pmesh_->n_vertices();
		int numTri = pmesh_->n_faces();
		std::vector<Wm5::Vector3d> vertices(numVertices);
		std::vector<int> indices(numTri*3);
		for (auto vitr=pmesh_->vertices_begin();vitr!=pmesh_->vertices_end();
			vitr++)
		{
			auto pt = pmesh_->point(*vitr);
			vertices[vitr->idx()] = Wm5::Vector3d(pt[0],pt[1],pt[2]);
		}
			
		int vcounter = 0;
		for (auto fitr=pmesh_->faces_begin();fitr!=pmesh_->faces_end();
			fitr++)
		{
			auto fh = *fitr;
			
			for (auto fvitr = pmesh_->fv_begin(fh); 
				fvitr != pmesh_->fv_end(fh); fvitr++)
			{
				indices[vcounter++] = fvitr->idx();
			}
		}
		Wm5::MeshCurvatured curvEstimator (numVertices, &vertices[0], numTri, &indices[0]);

		for (int i = 0; i < numVertices; i++)
		{
			auto vh = pmesh_->vertex_handle(i);
			auto& vp = pmesh_->property(omp_crv_tensor_, vh);

			auto pdmin = curvEstimator.GetMinDirections()[i],
				pdmax = curvEstimator.GetMaxDirections()[i];

			vp.pdmax[0] = pdmax[0];	vp.pdmax[1] = pdmax[1];	vp.pdmax[2] = pdmax[2];
			vp.pdmin[0] = pdmin[0];	vp.pdmin[1] = pdmin[1];	vp.pdmin[2] = pdmin[2];

			vp.kmax = curvEstimator.GetMaxCurvatures()[i];
			vp.kmin = curvEstimator.GetMinCurvatures()[i];

			// check validity
			if (std::isnan(pdmax[0])||std::isnan(pdmax[1])|| std::isnan(pdmax[2])
				|| std::isnan(pdmin[0]) || std::isnan(pdmin[1]) || std::isnan(pdmin[2]))
			{
				std::cout << " Nan frames computed!" << std::endl;
				std::exit(0);
			}
		}
	}

	void FieldsOnMesh::build_complex_based_laplacian_mat(
			std::vector<Eigen::Triplet<double>>& mat_entries) 
	{

		for (auto eitr=pmesh_->edges_begin(); eitr!=pmesh_->edges_end();
			eitr++)
		{
			auto eh = *eitr;
			auto heh = pmesh_->halfedge_handle(eh, 0);
			auto from_vh = pmesh_->from_vertex_handle(heh),
				to_vh = pmesh_->to_vertex_handle(heh);
			
			//setup direction difference measure.

			auto edge_dir = pmesh_->point(to_vh) - pmesh_->point(from_vh);
			if (edge_dir.length()<1e-6)
			{
				edge_dir = TriMesh::Point(1, 0, 0);
			}
			else
			{
				edge_dir.normalize();
			}
			

			std::complex<double> ef = complex_from_vec(from_vh, *pmesh_, edge_dir);
			std::complex<double> et = complex_from_vec(to_vh, *pmesh_, edge_dir);
			ef = std::conj(ef);
			et = std::conj(et);
			//for (size_t pw_itr = 0; pw_itr < 2; pw_itr++)
			//{
			//	ef *= ef;
			//	et *= et;
			//}
			ef = std::pow(ef, option_.sym_N);
			et = std::pow(et, option_.sym_N);

			int var_id[4] = { from_vh.idx() * 2, from_vh.idx() * 2 + 1,	
				to_vh.idx() * 2, to_vh.idx() * 2 + 1 };
			double diff[2][4] = { { ef.real(), -ef.imag(), -et.real(), et.imag() },
				{ ef.imag(), ef.real(), -et.imag(), -et.real() } };

			for (size_t i = 0; i < 2; i++)
			{
				for (size_t var_itr = 0; var_itr < 4; var_itr++) {
					for (size_t var_itr_1 = 0; var_itr_1 < 4; var_itr_1++)
					{
						//if (var_id[var_itr] >= var_id[var_itr_1])
						{
							mat_entries.push_back(
								Eigen::Triplet<double>(var_id[var_itr], var_id[var_itr_1], diff[i][var_itr] * diff[i][var_itr_1])
							);
						}
					}
				}
			}
		}

	}


	////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////


	void FieldsOnSurface::solve(const FieldsOptions& option) {
		m_options = option;
		// compute curvature tensor
		std::vector<CurvatureTensor> crv_tensors;
		compute_curvature_tensor(crv_tensors);

		if (option.field_computation_method == FieldsOptions::RAW_CURVATURE_TENSOR)
		{
			std::cout << "RAW_CURVATURE" << std::endl;
			FX.resize(V.size());
			for (size_t i = 0; i < V.size(); i++)
			{
				FX[i] = crv_tensors[i].pdmax;
			}
		}
		else if (option.field_computation_method == FieldsOptions::POLYVECTOR)
		{
			if (option.smooth_only)
			{
				std::cout << "smooth_only" << std::endl;
				solve_polyvector_smoothonly(crv_tensors);
			}
			else
			{
				std::cout << "S&C lambda:" << option.lambda << std::endl;
				solve_polyvector(crv_tensors, option.lambda);
			}
		}
		else
		{
			printf("Not implemented! Line %d, %s.\n",__LINE__, __FILE__);
			std::exit(0);
		}
	}

	void FieldsOnSurface::solve_polyvector(const std::vector<CurvatureTensor>& crv_tensors, double lambda) {
		
		const int vnum = V.size();
		const int sym_N = m_options.sym_N;

		//build a tangent frame for each point
		std::vector<TriMesh::Point> RefU(vnum), RefV(vnum);
		for (size_t i = 0; i < vnum; i++)
		{
			complement_axis(N[i], RefU[i], RefV[i]);
		}

		//complex number based smooth direction field computation
		Eigen::SparseMatrix<double, Eigen::ColMajor> cmplx_smth_normal_mat(vnum * 2, vnum * 2);
		std::vector<double> rhs(vnum * 2, 0.0);
		typedef Eigen::Triplet<double> MatEntry;
		std::vector<MatEntry> mat_entries;

		//std::cout << "Smoothing direction field... " << std::endl;
		const double crv_dir_weight = lambda; //default is 0.01
		for (int vitr = 0; vitr < vnum; vitr++)
		{
			auto& crv_tensor = crv_tensors[vitr];

			double abs_crv = std::tanh(std::abs(crv_tensor.kmin - crv_tensor.kmax));

			std::complex<double> vdir = complex_from_vec(crv_tensor.pdmax,RefU[vitr],RefV[vitr]);
			vdir = std::pow(vdir, sym_N);

			if (is_valid(vdir.real()) && is_valid(vdir.imag()) && is_valid(abs_crv))
			{
				mat_entries.push_back(MatEntry(vitr * 2, vitr * 2, abs_crv*crv_dir_weight));
				rhs[vitr * 2] = abs_crv*crv_dir_weight*vdir.real();

				mat_entries.push_back(MatEntry(vitr * 2 + 1, vitr * 2 + 1, abs_crv*crv_dir_weight));
				rhs[vitr * 2 + 1] = abs_crv*crv_dir_weight*vdir.imag();

			}
			else
			{
				std::cout << "Polyvector for raw curvature direction is NaN! "
					<< crv_tensor.pdmax << ", " << RefU[vitr] << ", " << RefV[vitr]
					<< "    " << abs_crv << std::endl;
			}

		}

		build_complex_based_laplacian_mat(mat_entries, RefU, RefV);

		// debug, add diagonal entry to make PD
		for (size_t vitr = 0; vitr < vnum; vitr++)
		{
			const double stablizer = 1e-6;
			//if (Adj[vitr].size()<1)
			{
				mat_entries.push_back(MatEntry(vitr * 2, vitr * 2, stablizer));
				mat_entries.push_back(MatEntry(vitr * 2 + 1, vitr * 2 + 1, stablizer));
			}
		}

		cmplx_smth_normal_mat.setFromTriplets(mat_entries.begin(), mat_entries.end());
		//std::cout << "Finished building the smoothness matrix" << std::endl;

		//// debug, check symmetry
		//{
		//	for (size_t i = 0; i < 2*vnum; i++)
		//	{
		//		for (size_t j = 0; j < 2*vnum; j++)
		//		{
		//			if (cmplx_smth_normal_mat.coeff(i, j) != cmplx_smth_normal_mat.coeff(j, i))
		//			{
		//				std::cout << "Matrix not symmetric!\n";
		//				printf("(%d,%d) %f, (%d,%d) %f \n",
		//					i, j, cmplx_smth_normal_mat.coeff(i, j),
		//					j, i, cmplx_smth_normal_mat.coeff(j, i));

		//				std::exit(0);
		//			}
		//		}
		//		
		//	}
		//}

		Eigen::VectorXd sol, initial_sol(vnum*2);

		// set initial value
		for (size_t vitr = 0; vitr < vnum; vitr++)
		{
			initial_sol[2 * vitr] = initial_sol[2 * vitr + 1] = 0.0;
			
			if (FX[vitr].norm()>1e-7)
			{
				std::complex<double> vdir = complex_from_vec(FX[vitr], RefU[vitr], RefV[vitr]);
				if (is_valid(vdir.real()) && is_valid(vdir.imag()))
				{
					vdir = std::pow(vdir, sym_N);
					initial_sol[2 * vitr] = vdir.real();
					initial_sol[2 * vitr + 1] = vdir.imag();
				}
				else
				{
					std::cout << "Polyvector for initial direction is NaN!"
						<< FX[vitr] << " " << RefU[vitr] << " " << RefV[vitr] << std::endl;
				}
			}
		}

		{
			Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,Eigen::Lower|Eigen::Upper> cg_solver;
			int maxItr = std::max(500 - vnum/1000.0*20.0, 200.0);
			cg_solver.setMaxIterations(maxItr);
			std::cout << "\tCG solver begin... vnum: "<<vnum<<std::endl;
			cg_solver.compute(cmplx_smth_normal_mat);
			Eigen::Map<Eigen::VectorXd> vrhs(rhs.data(), rhs.size());
			//sol = cg_solver.solve(vrhs);
			sol = cg_solver.solveWithGuess(vrhs, initial_sol);
			if (cg_solver.info() != Eigen::Success)
			{
				std::cerr << "\tWarning: CG solve not converged.\n";
				//return;
				//std::exit(0);
			}
			std::cout << "\tCG #iterations:     " << cg_solver.iterations() << std::endl;
			std::cout << "\t   estimated error: " << cg_solver.error() << std::endl;
		}

		/*{
			Sparse_Matrix sparse_mat(vnum * 2, vnum * 2, SYM_LOWER);
			for (size_t i = 0; i < mat_entries.size(); i++)
			{
				auto& entry = mat_entries[i];
				sparse_mat.fill_entry(entry.row(), entry.col(), entry.value());
			}
			sparse_mat.get_rhs() = rhs;
			sparse_mat.get_solution().resize(vnum * 2);
			for (size_t i = 0; i < vnum*2; i++)
			{
				sparse_mat.get_solution()[i] = initial_sol[i];
			} 

			solve_by_CG_with_jacobian_pred(&sparse_mat, vnum*2);
			sol.resize(vnum * 2);
			for (size_t i = 0; i < vnum*2; i++)
			{
				sol[i] = sparse_mat.get_solution()[i];
			}
		}*/

		//std::cout << "Finished solving the matrix" << std::endl;

		//read solutions
		for (int vitr = 0; vitr < vnum; vitr++)
		{
			std::complex<double> u_pow(sol[vitr * 2], sol[vitr * 2 + 1]);
			//std::cout << u_pow << std::endl;
			double norm = std::norm(u_pow);
			if (!is_valid(sol[vitr*2]) || !is_valid(sol[vitr*2+1]))
			{
				std::cout << "Solution has invalid number. Aborting.\n";
				std::ofstream dump_file("DUMP.txt");
				dump_file << cmplx_smth_normal_mat << std::endl << std::endl;
				for (size_t i = 0; i < vnum; i++)
				{
					dump_file << sol[i * 2] << "\n" << sol[i * 2 + 1] << std::endl;
				}
				dump_file << std::endl << std::endl;
				for (size_t i = 0; i < vnum; i++)
				{
					dump_file << rhs[i * 2] << "\n" << rhs[i * 2 + 1] << std::endl;
				}
				dump_file.close();

				for (size_t i = 0; i < vnum; i++)
				{
					if (Adj[i].size()<1)
					{
						std::cout << "Hanging vertex here." << std::endl;
					}
				}

				std::exit(0);
			}

			if (norm < 1e-7)
			{
				FX[vitr] = crv_tensors[vitr].pdmax;
				continue;
			}
			else
			{
				u_pow /= std::sqrt(norm); 
				//u_pow = std::sqrt(u_pow);
				//u_pow = std::sqrt(u_pow);
				u_pow = std::pow(u_pow, 1.0 / sym_N);
				
				// check validity
				if (!is_valid(u_pow.real()) || !is_valid(u_pow.imag()))
				{
					std::cout << "Solved polyvector is invalid! " << sol[vitr * 2] << " " << sol[vitr * 2 + 1] << std::endl;

					std::cout << "Dumping solution...\n";
					for (int i = 0; i < 2 * vnum; i++)
					{
						std::cout << sol[i] << " ";
					}

					std::exit(0);
				}

				// store smoothed principle curvature
				FX[vitr] = vec_from_complex(u_pow, RefU[vitr], RefV[vitr]).normalize();
				//FY[vitr] = OpenMesh::cross(N[vitr], FX[vitr]).normalize();
			}
		}
	}

	void FieldsOnSurface::solve_polyvector_smoothonly(const std::vector<CurvatureTensor>& crv_tensors) {
		int vnum = V.size();

		//build a tangent frame for each point
		std::vector<TriMesh::Point> RefU(vnum), RefV(vnum);
		for (size_t i = 0; i < vnum; i++)
		{
			complement_axis(N[i], RefU[i], RefV[i]);
		}

		//complex number based smooth direction field computation
		Eigen::SparseMatrix<double, Eigen::ColMajor> cmplx_smth_normal_mat(vnum * 2, vnum * 2);
		std::vector<double> rhs(vnum * 2, 0.0);
		typedef Eigen::Triplet<double> MatEntry;
		std::vector<MatEntry> mat_entries;

		//std::cout << "Smoothing direction field... " << std::endl;

		build_complex_based_laplacian_mat(mat_entries, RefU, RefV);

		// debug, add diagonal entry to make PD
		for (size_t vitr = 0; vitr < vnum; vitr++)
		{
			const double stablizer = 1e-6;
			//if (Adj[vitr].size()<1)
			{
				mat_entries.push_back(MatEntry(vitr * 2, vitr * 2, stablizer));
				mat_entries.push_back(MatEntry(vitr * 2 + 1, vitr * 2 + 1, stablizer));
			}
		}

		cmplx_smth_normal_mat.setFromTriplets(mat_entries.begin(), mat_entries.end());
		//std::cout << "Finished building the smoothness matrix" << std::endl;


		Eigen::VectorXd sol, initial_sol(vnum * 2);

		// set initial value
		for (size_t vitr = 0; vitr < vnum; vitr++)
		{
			initial_sol[2 * vitr] = initial_sol[2 * vitr + 1] = 0.0;
			//if (FX[vitr].norm()>1e-7)
			//{
			//	std::complex<double> vdir = complex_from_vec(FX[vitr], RefU[vitr], RefV[vitr]);
			//	if (is_valid(vdir.real()) && is_valid(vdir.imag()))
			//	{
			//		vdir = std::pow(vdir, 4);
			//		initial_sol[2 * vitr] = vdir.real();
			//		initial_sol[2 * vitr + 1] = vdir.imag();
			//	}
			//	else
			//	{
			//		std::cout << "Polyvector for initial direction is NaN!"
			//			<< FX[vitr] << " " << RefU[vitr] << " " << RefV[vitr] << std::endl;
			//	}
			//}
			//else
			{
				double rn = static_cast <double> (rand()) / static_cast <double> (RAND_MAX); //[0,1] random float
				rn = 2 * rn - 1;
				initial_sol[2 * vitr] = rn;
				initial_sol[2 * vitr + 1] = std::sqrt(std::max(0., 1 - rn*rn));
			}
		}

		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_solver;
		//std::cout << "\tLDLT solver begin... vnum: " << vnum << std::endl;
		ldlt_solver.compute(cmplx_smth_normal_mat);
		const int max_pwitr = 500;
		for (int pwitr = 0; pwitr<max_pwitr; pwitr++)
		{
			sol = ldlt_solver.solve(initial_sol);
			if (ldlt_solver.info() != Eigen::Success)
			{
				std::cerr << "\tWarning: LDLT solver failed.\n";
				//return;
				//std::exit(0);
			}
			//normalize sol
			sol.normalize();
			double diff = (sol - initial_sol).norm()/sol.size();
			std::cout <<"\t"<< pwitr <<" Power Itr. diff norm: " << diff << std::endl;

			if (diff<1e-6)
			{
				break;
			}
			else
			{
				//copy for next iteration
				initial_sol = sol;
			}
		}

		//std::cout << "Finished solving the matrix" << std::endl;

		//read solutions
		for (int vitr = 0; vitr < vnum; vitr++)
		{
			std::complex<double> u_pow(sol[vitr * 2], sol[vitr * 2 + 1]);
			//std::cout << u_pow << std::endl;
			double norm = std::norm(u_pow);
			if (!is_valid(sol[vitr * 2]) || !is_valid(sol[vitr * 2 + 1]))
			{
				std::cout << "Solution has invalid number. Aborting.\n";
				std::ofstream dump_file("DUMP.txt");
				dump_file << cmplx_smth_normal_mat << std::endl << std::endl;
				for (size_t i = 0; i < vnum; i++)
				{
					dump_file << sol[i * 2] << "\n" << sol[i * 2 + 1] << std::endl;
				}
				dump_file << std::endl << std::endl;
				for (size_t i = 0; i < vnum; i++)
				{
					dump_file << rhs[i * 2] << "\n" << rhs[i * 2 + 1] << std::endl;
				}
				dump_file.close();

				for (size_t i = 0; i < vnum; i++)
				{
					if (Adj[i].size()<1)
					{
						std::cout << "Hanging vertex here." << std::endl;
					}
				}

				std::exit(0);
			}

			if (norm < 1e-7)
			{
				FX[vitr] = crv_tensors[vitr].pdmax;
				continue;
			}
			else
			{
				u_pow /= std::sqrt(norm);
				//u_pow = std::sqrt(u_pow);
				//u_pow = std::sqrt(u_pow);
				u_pow = std::pow(u_pow, 1.0 / m_options.sym_N);

				// check validity
				if (!is_valid(u_pow.real()) || !is_valid(u_pow.imag()))
				{
					std::cout << "Solved polyvector is invalid! " << sol[vitr * 2] << " " << sol[vitr * 2 + 1] << std::endl;

					std::cout << "Dumping solution...\n";
					for (int i = 0; i < 2 * vnum; i++)
					{
						std::cout << sol[i] << " ";
					}

					std::exit(0);
				}

				// store smoothed principle curvature
				FX[vitr] = vec_from_complex(u_pow, RefU[vitr], RefV[vitr]).normalize();
			}
		}
	}

	void FieldsOnSurface::compute_curvature_tensor(std::vector<CurvatureTensor>& crv_tensors) {

		// Adapted from WildMagic5, GeometricTools.

		const int vnum = V.size();
		
		// Compute the matrix of normal derivatives.
		Wm5::Matrix3d* DNormal = new1<Wm5::Matrix3d>(vnum);
		Wm5::Matrix3d* WWTrn = new1<Wm5::Matrix3d>(vnum);
		Wm5::Matrix3d* DWTrn = new1<Wm5::Matrix3d>(vnum);
		bool* DWTrnZero = new1<bool>(vnum);
		memset(WWTrn, 0, vnum * sizeof(Wm5::Matrix3d));
		memset(DWTrn, 0, vnum * sizeof(Wm5::Matrix3d));
		memset(DWTrnZero, 0, vnum * sizeof(bool));

		int row, col;
		for (int vitr = 0; vitr < vnum; ++vitr)
		{
			for (int nbitr = 0; nbitr < Adj[vitr].size(); nbitr++)
			{
				int v0 = vitr;
				int v1 = Adj[vitr][nbitr];
				
				// Compute edge from V0 to V1, project to tangent plane of vertex,
				// and compute difference of adjacent normals.
				Wm5::Vector3d E = PointToVec(V[v1]) - PointToVec(V[v0]);
				Wm5::Vector3d W = E - (E.Dot(PointToVec(N[v0]))) * PointToVec(N[v0]);
				Wm5::Vector3d D = PointToVec(N[v1] - N[v0]);  // TODO: can tune this normal difference to be robust to normal flipping
				for (row = 0; row < 3; ++row)
				{
					for (col = 0; col < 3; ++col)
					{
						WWTrn[v0][row][col] += W[row] * W[col];
						DWTrn[v0][row][col] += D[row] * W[col];
					}
				}

			}
		}

		// Add in N*N^T to W*W^T for numerical stability.  In theory 0*0^T gets
		// added to D*W^T, but of course no update is needed in the
		// implementation.  Compute the matrix of normal derivatives.
		for (int vitr = 0; vitr < vnum; ++vitr)
		{

			for (row = 0; row < 3; ++row)
			{
				for (col = 0; col < 3; ++col)
				{
					WWTrn[vitr][row][col] = (0.5)*WWTrn[vitr][row][col] +
						N[vitr][row] * N[vitr][col];
					DWTrn[vitr][row][col] *= 0.5;
				}
			}

			// Compute the max-abs entry of D*W^T.  If this entry is (nearly)
			// zero, flag the DNormal matrix as singular.
			double maxAbs = 0.;
			for (row = 0; row < 3; ++row)
			{
				for (col = 0; col < 3; ++col)
				{
					double absEntry = std::abs(DWTrn[vitr][row][col]);
					if (absEntry > maxAbs)
					{
						maxAbs = absEntry;
					}
				}
			}
			if (maxAbs < 1e-07)
			{
				DWTrnZero[vitr] = true;
			}

			DNormal[vitr] = DWTrn[vitr] * WWTrn[vitr].Inverse();
		}

		//delete1(WWTrn);
		//delete1(DWTrn);

		// If N is a unit-length normal at a vertex, let U and V be unit-length
		// tangents so that {U, V, N} is an orthonormal set.  Define the matrix
		// J = [U | V], a 3-by-2 matrix whose columns are U and V.  Define J^T
		// to be the transpose of J, a 2-by-3 matrix.  Let dN/dX denote the
		// matrix of first-order derivatives of the normal vector field.  The
		// shape matrix is
		//   S = (J^T * J)^{-1} * J^T * dN/dX * J = J^T * dN/dX * J
		// where the superscript of -1 denotes the inverse.  (The formula allows
		// for J built from non-perpendicular vectors.) The matrix S is 2-by-2.
		// The principal curvatures are the eigenvalues of S.  If k is a principal
		// curvature and W is the 2-by-1 eigenvector corresponding to it, then
		// S*W = k*W (by definition).  The corresponding 3-by-1 tangent vector at
		// the vertex is called the principal direction for k, and is J*W.
		crv_tensors.resize(vnum);
		for (int vitr = 0; vitr < vnum; ++vitr)
		{
			// Compute U and V given N.
			Wm5::Vector3d U, V;
			Wm5::Vector3d::GenerateComplementBasis(U, V, PointToVec(N[vitr]));

			if (DWTrnZero[vitr])
			{
				// At a locally planar point.
				crv_tensors[vitr].kmin = 0;
				crv_tensors[vitr].kmax = 0;
				crv_tensors[vitr].pdmin = VecToPoint(U);
				crv_tensors[vitr].pdmax = VecToPoint(V);
				continue;
			}

			//if (Adj[vitr].size() < 2) // Less than two points cannot make robust estimation.
			//{
			//	crv_tensors[vitr].kmin = 0;
			//	crv_tensors[vitr].kmax = 0;
			//	crv_tensors[vitr].pdmin = VecToPoint(U);
			//	crv_tensors[vitr].pdmax = VecToPoint(V);
			//	continue;
			//}

			// Compute S = J^T * dN/dX * J.  In theory S is symmetric, but
			// because we have estimated dN/dX, we must slightly adjust our
			// calculations to make sure S is symmetric.
			double s01 = U.Dot(DNormal[vitr] * V);
			double s10 = V.Dot(DNormal[vitr] * U);
			double sAvr = 0.5*(s01 + s10);
			Wm5::Matrix2d S
			(
				U.Dot(DNormal[vitr] * U), sAvr,
				sAvr, V.Dot(DNormal[vitr] * V)
			);

			// Compute the eigenvalues of S (min and max curvatures).
			double trace = S[0][0] + S[1][1];
			double det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
			double discr = trace*trace - 4.0*det;
			double rootDiscr = std::sqrt(std::abs(discr));
			crv_tensors[vitr].kmin = 0.5*(trace - rootDiscr);
			crv_tensors[vitr].kmax = 0.5*(trace + rootDiscr);

			// Compute the eigenvectors of S.
			Wm5::Vector2d W0(S[0][1], crv_tensors[vitr].kmin - S[0][0]);
			Wm5::Vector2d W1(crv_tensors[vitr].kmin - S[1][1], S[1][0]);
			if (W0.SquaredLength() >= W1.SquaredLength())
			{
				W0.Normalize();
				crv_tensors[vitr].pdmin = VecToPoint(W0.X()*U + W0.Y()*V);
			}
			else
			{
				W1.Normalize();
				crv_tensors[vitr].pdmin = VecToPoint(W1.X()*U + W1.Y()*V);
			}

			W0 = Wm5::Vector2d(S[0][1], crv_tensors[vitr].kmax - S[0][0]);
			W1 = Wm5::Vector2d(crv_tensors[vitr].kmax - S[1][1], S[1][0]);
			if (W0.SquaredLength() >= W1.SquaredLength())
			{
				W0.Normalize();
				crv_tensors[vitr].pdmax = VecToPoint(W0.X()*U + W0.Y()*V);
			}
			else
			{
				W1.Normalize();
				crv_tensors[vitr].pdmax = VecToPoint(W1.X()*U + W1.Y()*V);
			}

			// Final checking on curvatures
			if (!is_valid(crv_tensors[vitr].kmax)||
				!is_valid(crv_tensors[vitr].kmin)||
				!is_valid(crv_tensors[vitr].pdmax)||
				!is_valid(crv_tensors[vitr].pdmin)||
				crv_tensors[vitr].pdmax.norm()<1e-6||
				crv_tensors[vitr].pdmin.norm()<1e-6
				)
			{
				crv_tensors[vitr].kmin = 0;
				crv_tensors[vitr].kmax = 0;
				crv_tensors[vitr].pdmin = VecToPoint(U);
				crv_tensors[vitr].pdmax = VecToPoint(V);
			}
		}

		

		////debug, check computed curvature tensor validity
		//{
		//	auto isNan = [](TriMesh::Point p)->bool {
		//		if (std::isnan(p[0])||std::isnan(p[1])||std::isnan(p[2]))
		//		{
		//			return true;
		//		}
		//		return false;
		//	};
		//	auto print_tensor = [](CurvatureTensor& t)->std::string {
		//		char s[1024];
		//		sprintf(s, "kmin %f, kmax %f, pdmin: %f,%f,%f, pdmax: %f,%f,%f",
		//			t.kmin, t.kmax, t.pdmin[0], t.pdmin[1], t.pdmin[2],
		//			t.pdmax[0], t.pdmax[1], t.pdmax[2]);
		//		return std::string(s);
		//	};
		//	auto print_mat = [](Wm5::Matrix3d& mat)->std::string {
		//		char s[1024];
		//		sprintf(s, "%f %f %f\n%f %f %f\n%f %f %f", mat[0][0], mat[0][1], mat[0][2],
		//			mat[1][0], mat[1][1], mat[1][2],
		//			mat[2][0], mat[2][1], mat[2][2]);
		//		return std::string(s);
		//	};
		//	for (size_t i = 0; i < crv_tensors.size(); i++)
		//	{
		//		auto t = crv_tensors[i];
		//		if (std::isnan(t.kmin)||std::isnan(t.kmax)||isNan(t.pdmax)||isNan(t.pdmin))
		//		{
		//			std::cout << "Nan occured " << print_tensor(t) << std::endl;
		//			int junk;
		//			std::cin >> junk;
		//		}
		//		if (t.pdmax.norm()<1e-7 || t.pdmin.norm()<1e-7)
		//		{
		//			std::cout << "Zero length PDir occured " << print_tensor(t) << std::endl;
		//			std::cout << " Predicted planarity is " << (DWTrnZero[i] ? "True" : "False") << std::endl;
		//			std::cout << " Nb num: " << Adj[i].size() << std::endl;
		//			std::cout << "Nb N: " << N[Adj[i][0]]<<std::endl;
		//			
		//			Wm5::Vector3d U, V;
		//			Wm5::Vector3d::GenerateComplementBasis(U, V, PointToVec(N[i]));
		//			std::cout << " N, U, V: " << N[i]<<"   " << U << "  " << V << std::endl;
		//			//std::cout << "DNormal: " << DNormal[i] << std::endl;
		//			std::cout << "DWTrn: " << print_mat(DWTrn[i]) << std::endl;
		//			std::cout << "WWTrn: " << print_mat(WWTrn[i]) << std::endl;
		//			std::cout << "WWTrn Inverse: " << print_mat(WWTrn[i].Inverse()) << std::endl;
		//			std::cout << "DNormal: " << print_mat(DNormal[i]) << std::endl;

		//			std::cout << " DNormal*U, DNormal*V " << DNormal[i]*U
		//				<< "   " << DNormal[i]*V << std::endl;

		//			int junk;
		//			std::cin >> junk;
		//		}
		//	}
		//}


		delete1(WWTrn);
		delete1(DWTrn);
		delete1(DWTrnZero);
		delete1(DNormal);

	}

	void FieldsOnSurface::build_complex_based_laplacian_mat(std::vector<Eigen::Triplet<double>>& mat_entries,
		const std::vector<TriMesh::Point>& RefU, const std::vector<TriMesh::Point>& RefV) {
		for (int vitr = 0; vitr < V.size(); vitr++)
		{
			for (int nb_itr = 0; nb_itr < Adj[vitr].size(); nb_itr++)
			{
				int from_vidx = vitr, to_vidx = Adj[vitr][nb_itr];
				//setup direction difference measure.

				auto edge_dir = V[to_vidx] - V[from_vidx];
				edge_dir.normalize();
				
				std::complex<double> ef = complex_from_vec(edge_dir, RefU[from_vidx], RefV[from_vidx]);
				std::complex<double> et = complex_from_vec(edge_dir, RefU[to_vidx], RefV[to_vidx]);

				ef = std::conj(ef);
				et = std::conj(et);
				/*for (size_t pw_itr = 0; pw_itr < 2; pw_itr++)
				{
					ef *= ef;
					et *= et;
				}*/
				ef = std::pow(ef, m_options.sym_N);
				et = std::pow(et, m_options.sym_N);

				if (!is_valid(ef.real()) ||
					!is_valid(ef.imag()) ||
					!is_valid(et.real()) ||
					!is_valid(et.imag()))
				{
					continue;
				}

				int var_id[4] = { from_vidx * 2, from_vidx * 2 + 1,
					to_vidx * 2, to_vidx * 2 + 1 };
				double diff[2][4] = { { ef.real(), -ef.imag(), -et.real(), et.imag() },
				{ ef.imag(), ef.real(), -et.imag(), -et.real() } };

				for (size_t i = 0; i < 2; i++)
				{
					for (size_t var_itr = 0; var_itr < 4; var_itr++) {
						for (size_t var_itr_1 = 0; var_itr_1 < 4; var_itr_1++)
						{
							//if (var_id[var_itr] >= var_id[var_itr_1])
							{
								mat_entries.push_back(
									Eigen::Triplet<double>(var_id[var_itr], var_id[var_itr_1], 
										0.5*diff[i][var_itr] * diff[i][var_itr_1]) // 0.5 because the pair is visited twice
								);
							}
						}
					}
				}
			}
		}
	}
}