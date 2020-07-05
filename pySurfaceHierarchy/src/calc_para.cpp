#include "calc_para.h"

int get_axis_map_p2p(TriMesh::Point centN, TriMesh::Point nbN, int anum, const std::vector<TriMesh::Point>& centAxes,
	const std::vector<TriMesh::Point>& nbAxes) {
	// rotation mat from nbN to centN
	TNT::Matrix<double> rot_mat(3, 3, 0.0);
	{
		//set Identity rotation
		rot_mat[0][0] = rot_mat[1][1] = rot_mat[2][2] = 1.0;

		auto prdct = OpenMesh::cross(nbN, centN);
		double sin_theta = prdct.length();
		if (!is_valid(sin_theta)) {
			std::cout << "Cannot rotate and match two normal vectors";	//std::exit(0); 
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
			auto K2 = K * K;
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
		double dpt = OpenMesh::dot(centAxis, mult_tntmat_vec3(rot_mat, nb_axis));
		if (dpt > cls_max)
		{
			cls_max = dpt; cls_nb_axis = nba_itr;
		}
	}
	return cls_nb_axis;
}

int load_frame(std::string frame_path, TriMesh *new_mesh, std::vector<VertexFrame> &new_frame_field)//didn't push normal into axis but give normal to mesh
{
	int nv, nsym = 0;
	if (new_mesh)
	{
		std::ifstream infile(frame_path);
		infile >> nv >> nsym;

		if (nv != new_mesh->n_vertices())
		{
			std::cout << "Error: vertex number does not match!" << std::endl;
			std::cout << "Num verts: " << new_mesh->n_vertices() << " Num frames: " << nv << std::endl;
			return nsym;
		}
		new_frame_field.clear();
		new_frame_field.resize(nv);

		new_mesh->request_face_normals();
		new_mesh->request_vertex_normals();
		new_mesh->update_normals();

		for (int i = 0; i < nv; i++)
		{
			TriMesh::Point nml, xaxis;
			infile >> nml >> xaxis;

			new_mesh->set_normal(new_mesh->vertex_handle(i), nml);

			TriMesh::Point yaxis = OpenMesh::cross(nml, xaxis);
			double delta = 2 * M_PI / nsym, theta = 0.;

			auto& frame = new_frame_field[i];
			//frame.axis.push_back(nml);
			for (int rot = 0; rot < nsym; rot++)
			{
				TriMesh::Point rot_axis = std::cos(theta)*xaxis + std::sin(theta)*yaxis;
				theta += delta;
				frame.axis.push_back(rot_axis);
			}
			//std::cout << frame.axis.size() << std::endl;
			frame.pos = new_mesh->point(new_mesh->vertex_handle(i));
			frame.vid = i;
		}
		infile.close();
	}
	else
	{
		std::cout << "[Error]No such mesh" << std::endl;
	}
	return nsym;
}

int load_frame(std::ifstream &infile, std::vector<TriMesh::Point> &point_cloud, std::vector<VertexFrame> &new_frame_field)//didn't push normal into axis but give normal to mesh
{
	int nv, nsym = 0;
	infile >> nv >> nsym;
	//nsym = 1;
	if (nv != point_cloud.size())
	{
		std::cout << "Error: vertex number does not match!" << std::endl;
		std::cout << "Num verts: " << point_cloud.size() << " Num frames: " << nv << std::endl;
		return nsym;
	}
	new_frame_field.clear();
	new_frame_field.resize(nv);

	for (int i = 0; i < nv; i++)
	{
		TriMesh::Point nml, xaxis;
		infile >> nml >> xaxis;

		TriMesh::Point yaxis = OpenMesh::cross(nml, xaxis);
		double delta = 2 * M_PI / nsym, theta = 0.;

		auto& frame = new_frame_field[i];
		//frame.axis.push_back(nml);
		for (int rot = 0; rot < nsym; rot++)
		{
			TriMesh::Point rot_axis = std::cos(theta)*xaxis + std::sin(theta)*yaxis;
			theta += delta;
			frame.axis.push_back(rot_axis);
		}
		//std::cout << frame.axis.size() << std::endl;
		frame.pos = point_cloud[i];
		frame.vid = i;
	}
	return nsym;
}

void load_pc_nml(std::ifstream &infile, std::vector<TriMesh::Point> &point_cloud, std::vector<TriMesh::Point> &pc_normal, int level_i)
{
	int vnum;
	infile >> vnum;
	//debug_msg(std::to_string(vnum));
	point_cloud.resize(vnum);
	pc_normal.resize(vnum);
	for (size_t vitr = 0; vitr < vnum; vitr++)
	{
		infile >> point_cloud[vitr][0] >> point_cloud[vitr][1] >> point_cloud[vitr][2];
		infile >> pc_normal[vitr][0] >> pc_normal[vitr][1] >> pc_normal[vitr][2];
		if (pc_normal[vitr].length() < 1e-6)
		{
			pc_normal[vitr][0] = 1;
			pc_normal[vitr][1] = 0;
			pc_normal[vitr][2] = 0;
		}
		else pc_normal[vitr].normalize();

		double r, g, b, label;
		infile >> r >> g >> b;
		if (level_i == 0) infile >> label;
	}
}

void read_hrch(std::string hrch_path, std::vector<std::vector<int>> &cvts)
{
	std::ifstream fin(hrch_path);
	int v_num;
	fin >> v_num;
	std::string str;
	std::getline(fin, str);//first line
	for (int i = 0; i < v_num; i++)
	{
		std::getline(fin, str);
		std::istringstream is(str);
		int tmp;
		while (is >> tmp)
		{
			cvts[i].push_back(tmp);
		}
	}
	fin.close();
	return;
}

void tan_para_singlefile(std::string mesh_dir_path, std::string filename, int start_level, int level_num, double *grid_length, int grid_size)
{
	std::cout << filename << std::endl;
	for (int j = start_level; j < start_level + level_num; j++)
	{
		///*
		std::string ck_para_path = mesh_dir_path + filename + "_" + std::to_string(j) + "_pad.para";
		if (_access(ck_para_path.c_str(), 0) != -1)//exists
		{
			std::cout << ck_para_path << " exists" << std::endl;
			continue;
		}
		//*/
		std::string mesh_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".obj";
		std::cout << mesh_path << std::endl;
		//load mesh
		TriMesh *new_mesh = new TriMesh;
		if (!OpenMesh::IO::read_mesh(*new_mesh, mesh_path))
		{
			std::cerr << "Error: cannot read mesh file " << mesh_path << std::endl;
			return;
		}
		std::cout << "[Success]Read mesh" << std::endl;
		//load frames
		std::string frame_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".frames";
		std::vector<VertexFrame> new_frame_field;
		load_frame(frame_path, new_mesh, new_frame_field);
		//calc para
		std::cout << "[Success]load frames" << std::endl;
		TangentParamVis new_tangent_map_;
		new_tangent_map_.mesh = new_mesh;

		new_tangent_map_.axes.resize(new_frame_field.size());
		for (int i = 0; i < new_mesh->n_vertices(); i++)
		{
			new_tangent_map_.axes[i].insert(
				new_tangent_map_.axes[i].end(),
				new_frame_field[i].axis.begin(), new_frame_field[i].axis.end());
		}

		new_tangent_map_.is_reweighted = true;
		new_tangent_map_.is_use_patch_height = true;
		new_tangent_map_.build_grid_samples(grid_length[j], grid_size, grid_size);
		//store para

		std::string para_path = mesh_dir_path + filename + "_" + std::to_string(j) + "_pad.para";
		new_tangent_map_.write_weights(para_path);


		std::string input_feature_path = mesh_dir_path + filename + "_" + std::to_string(j) + (new_tangent_map_.is_reweighted ? "_reweighted" : "") + ".input";
		new_tangent_map_.write_input_feature(input_feature_path);
		std::cout << "Finished writing" << input_feature_path << std::endl;
	}
}

void tan_para(std::string mesh_dir_path, std::string ext, int start_level, int level_num, double *grid_length)
{
	long long handle;
	struct _finddata_t fileinfo;
	std::string in_data_fn, out_fn;
	handle = _findfirst((mesh_dir_path + ext).c_str(), &fileinfo);
	do
	{
		std::string filename = fileinfo.name;
		std::cout << filename << std::endl;
		filename = filename.substr(0, filename.length() - 6);
		std::cout << filename << std::endl;
		for (int j = start_level; j < start_level + level_num; j++)
		{
			///*
			std::string ck_para_path = mesh_dir_path + filename + "_" + std::to_string(j) + "_pad.para";
			if (_access(ck_para_path.c_str(), 0) != -1)//exists
			{
				std::cout << ck_para_path << " exists" << std::endl;
				continue;
			}
			//*/
			std::string mesh_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".obj";
			std::cout << mesh_path << std::endl;
			//load mesh
			TriMesh *new_mesh = new TriMesh;
			if (!OpenMesh::IO::read_mesh(*new_mesh, mesh_path))
			{
				std::cerr << "Error: cannot read mesh file " << mesh_path << std::endl;
				return;
			}
			std::cout << "[Success]Read mesh" << std::endl;
			//load frames
			std::string frame_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".frames";
			std::vector<VertexFrame> new_frame_field;
			load_frame(frame_path, new_mesh, new_frame_field);
			//calc para
			std::cout << "[Success]load frames" << std::endl;
			TangentParamVis new_tangent_map_;
			new_tangent_map_.mesh = new_mesh;

			new_tangent_map_.axes.resize(new_frame_field.size());
			for (int i = 0; i < new_mesh->n_vertices(); i++)
			{
				new_tangent_map_.axes[i].insert(
					new_tangent_map_.axes[i].end(),
					new_frame_field[i].axis.begin(), new_frame_field[i].axis.end());
			}
			/*
			double avg_elen = 0.0;
			for (auto eh : *new_mesh->get_edges_list())
			{
			avg_elen += (eh->vert->pos - eh->pair->vert->pos).length();
			}
			avg_elen /= (double)new_mesh->get_num_of_edges();
			new_tangent_map_.build_grid_samples(avg_elen, 5, 5);
			*/
			new_tangent_map_.is_reweighted = true;
			new_tangent_map_.is_use_patch_height = true;
			new_tangent_map_.build_grid_samples(grid_length[j], 3, 3);
			//store para

			//skip
			std::string para_path = mesh_dir_path + filename + "_" + std::to_string(j) + "_pad.para";
			new_tangent_map_.write_weights(para_path);


			std::string input_feature_path = mesh_dir_path + filename + "_" + std::to_string(j) + (new_tangent_map_.is_reweighted ? "_reweighted" : "") + ".input";
			new_tangent_map_.write_input_feature(input_feature_path);
			std::cout << "Finished writing" << input_feature_path << std::endl;
		}
	} while (!_findnext(handle, &fileinfo));
}



void tan_para_from_pc(std::string data_path, std::string ext, int start_level, int level_num, double *grid_length)
{
	long long handle;
	struct _finddata_t fileinfo;
	std::string in_data_fn, out_fn;
	handle = _findfirst((data_path + ext).c_str(), &fileinfo);
	do
	{
		std::string filename = fileinfo.name;
		std::cout << filename << std::endl;
		filename = filename.substr(0, filename.length() - 4);
		std::cout << filename << std::endl;

		std::string frame_name = data_path + filename + ".frames";
		std::ifstream frame_file(frame_name);

		std::string data_name = data_path + filename + ".txt";
		std::ifstream data_file(data_name);

		for (int j = start_level; j < start_level + level_num; j++)
		{
			/*
			std::string ck_para_path = data_path + filename + "_" + std::to_string(j) + "_pad.para";
			if (_access(ck_para_path.c_str(), 0) != -1)//exists
			{
				std::cout << ck_para_path << " exists" << std::endl;
				continue;
			}
			*/
			TangentParamVis new_tangent_map_;
			load_pc_nml(data_file, new_tangent_map_.pointcloud, new_tangent_map_.pc_normal, j);

			std::vector<VertexFrame> new_frame_field;
			load_frame(frame_file, new_tangent_map_.pointcloud, new_frame_field);

			new_tangent_map_.axes.resize(new_frame_field.size());
			for (int i = 0; i < new_tangent_map_.pointcloud.size(); i++)
			{
				new_tangent_map_.axes[i].insert(
					new_tangent_map_.axes[i].end(),
					new_frame_field[i].axis.begin(), new_frame_field[i].axis.end());
			}

			new_tangent_map_.is_reweighted = true;
			new_tangent_map_.is_use_patch_height = true;
			new_tangent_map_.build_grid_samples_from_pc(grid_length[j], 3, 3);
			//new_tangent_map_.build_grid_samples_from_pc(grid_length[j], 5, 5);
			//store para

			//skip
			std::string para_path = data_path + filename + "_" + std::to_string(j) + "_pad.para";
			new_tangent_map_.write_weights(para_path);


			std::string input_feature_path = data_path + filename + "_" + std::to_string(j) + (new_tangent_map_.is_reweighted ? "_reweighted" : "") + ".input";
			new_tangent_map_.write_input_feature(input_feature_path);
			std::cout << "Finished writing" << input_feature_path << std::endl;
		}
		frame_file.close();
		data_file.close();
	} while (!_findnext(handle, &fileinfo));
}

void hrch_axis_corr_singlefile(std::string mesh_dir_path, std::string filename, int start_level, int level_num)
{
	std::cout << "calc axis hrch: " << filename << std::endl;
	for (int j = start_level + 1; j < level_num; j++)
	{
		std::string check_pooling_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".pool";
		if (_access(check_pooling_path.c_str(), 0) != -1)//exists
		{
			std::cout << check_pooling_path << " exists" << std::endl;
			continue;
		}

		std::string child_mesh_path = mesh_dir_path + filename + "_" + std::to_string(j - 1) + ".obj";
		//load child mesh
		TriMesh *child_mesh = new TriMesh;
		if (!OpenMesh::IO::read_mesh(*child_mesh, child_mesh_path))
		{
			std::cerr << "Error: cannot read mesh file " << child_mesh_path << std::endl;
			return;
		}

		//load child frames and and give normals to mesh
		std::string child_frame_path = mesh_dir_path + filename + "_" + std::to_string(j - 1) + ".frames";
		std::vector<VertexFrame> child_frame_field;
		int axis_num = load_frame(child_frame_path, child_mesh, child_frame_field);

		//load parent mesh
		std::string parent_mesh_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".obj";
		TriMesh *parent_mesh = new TriMesh;
		if (!OpenMesh::IO::read_mesh(*parent_mesh, parent_mesh_path))
		{
			std::cerr << "Error: cannot read mesh file " << parent_mesh_path << std::endl;
			return;
		}
		//load parent frames and give normals to mesh
		std::string parent_frame_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".frames";
		std::vector<VertexFrame> parent_frame_field;
		load_frame(parent_frame_path, parent_mesh, parent_frame_field);
		//load coverd vertex
		std::string hrch_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".hrch";
		std::vector<std::vector<int>> covered_vts(parent_mesh->n_vertices(), std::vector<int>());

		read_hrch(hrch_path, covered_vts);

		std::string pooling_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".pool";
		std::ofstream fout(pooling_path);

		for (int k = 0; k < parent_mesh->n_vertices(); k++)
		{
			for (int l = 0; l < covered_vts[k].size(); l++)
			{
				int axis_idx = get_axis_map_p2p(parent_mesh->normal(parent_mesh->vertex_handle(k)), child_mesh->normal(child_mesh->vertex_handle(covered_vts[k][l])), axis_num, parent_frame_field[k].axis, child_frame_field[covered_vts[k][l]].axis);
				fout << axis_idx << " ";
			}
			fout << std::endl;
		}
		fout.close();
	}
}


void hrch_axis_corr(std::string mesh_dir_path, std::string ext, int start_level, int level_num)
{
	int axis_num = 4;
	long long handle;
	struct _finddata_t fileinfo;
	std::string in_data_fn, out_fn;
	handle = _findfirst((mesh_dir_path + ext).c_str(), &fileinfo);
	do
	{
		std::string filename = fileinfo.name;
		std::cout << "calc axis hrch: " << filename << std::endl;
		filename = filename.substr(0, filename.length() - 6);
		for (int j = start_level + 1; j < level_num; j++)
		{
			std::string check_pooling_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".pool";
			if (_access(check_pooling_path.c_str(), 0) != -1)//exists
			{
				std::cout << check_pooling_path << " exists" << std::endl;
				continue;
			}

			std::string child_mesh_path = mesh_dir_path + filename + "_" + std::to_string(j - 1) + ".obj";
			//load child mesh
			TriMesh *child_mesh = new TriMesh;
			if (!OpenMesh::IO::read_mesh(*child_mesh, child_mesh_path))
			{
				std::cerr << "Error: cannot read mesh file " << child_mesh_path << std::endl;
				return;
			}


			//load child frames and and give normals to mesh
			std::string child_frame_path = mesh_dir_path + filename + "_" + std::to_string(j - 1) + ".frames";
			std::vector<VertexFrame> child_frame_field;
			load_frame(child_frame_path, child_mesh, child_frame_field);

			//load parent mesh
			std::string parent_mesh_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".obj";
			TriMesh *parent_mesh = new TriMesh;
			if (!OpenMesh::IO::read_mesh(*parent_mesh, parent_mesh_path))
			{
				std::cerr << "Error: cannot read mesh file " << parent_mesh_path << std::endl;
				return;
			}
			//load parent frames and give normals to mesh
			std::string parent_frame_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".frames";
			std::vector<VertexFrame> parent_frame_field;
			load_frame(parent_frame_path, parent_mesh, parent_frame_field);
			//load coverd vertex
			std::string hrch_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".hrch";
			std::vector<std::vector<int>> covered_vts(parent_mesh->n_vertices(), std::vector<int>());

			read_hrch(hrch_path, covered_vts);

			std::string pooling_path = mesh_dir_path + filename + "_" + std::to_string(j) + ".pool";
			std::ofstream fout(pooling_path);

			for (int k = 0; k < parent_mesh->n_vertices(); k++)
			{
				/*std::cout << "vertex " << k << std::endl;
				std::cout << parent_mesh->get_vertex(k)->pos << std::endl;
				std::cout << "normal: " << parent_mesh->get_vertex(k)->normal << std::endl;
				for (int x = 0; x < parent_frame_field[k].axis.size(); x++)
				{
				std::cout << parent_frame_field[k].axis[x] << " | ";
				}
				std::cout << std::endl;*/
				for (int l = 0; l < covered_vts[k].size(); l++)
				{
					/*std::cout << "child " << l << std::endl;
					std::cout << child_mesh->get_vertex(covered_vts[k][l])->pos << std::endl;
					std::cout << "normal: " << child_mesh->get_vertex(k)->normal << std::endl;
					for (int x = 0; x < child_frame_field[covered_vts[k][l]].axis.size(); x++)
					{
					std::cout << child_frame_field[covered_vts[k][l]].axis[x] << " | ";
					}
					std::cout << std::endl;*/
					//axis do not include normal
					int axis_idx = get_axis_map_p2p(parent_mesh->normal(parent_mesh->vertex_handle(k)), child_mesh->normal(child_mesh->vertex_handle(covered_vts[k][l])), axis_num, parent_frame_field[k].axis, child_frame_field[covered_vts[k][l]].axis);
					fout << axis_idx << " ";
				}
				//std::cout << std::endl;
				fout << std::endl;
			}
			fout.close();
		}
	} while (!_findnext(handle, &fileinfo));
}

void store_tan_para_from_pc(std::string dataset_path, std::string ext, int strat_level, int level_num, double *grid_length)
{
	tan_para_from_pc(dataset_path, ext, strat_level, level_num, grid_length);
}

void store_tan_para(std::string dataset_path, std::string ext, int start_level, int level_num, double *grid_length)
{
	//double grid_length[3] = { 0.01, 0.015, 0.025 };
	//std::string dataset_path = "D:\\Yuqi\\data\\MPI-FAUST_training\\cleaned_dataset\\";
	//std::string ext = "*_0.obj";
	tan_para(dataset_path, ext, start_level, level_num, grid_length);
	hrch_axis_corr(dataset_path, ext, start_level, level_num);
	/*
	double grid_length[3] = { 0.01, 0.02, 0.04 };
	std::string train_path = "D:\\Yuqi\\data\\sig17_seg\\v_based_3level\\train\\";
	std::string test_path = "D:\\Yuqi\\data\\sig17_seg\\v_based_3level\\test\\";
	tan_para(train_path, "*_0.obj", 0, 3, grid_length);
	tan_para(test_path, "*_0.obj", 0, 3, grid_length);
	*/
}

void store_tan_para_singlefile(std::string mesh_dir_path, std::string filename, int start_level, int level_num, double *grid_length, int grid_size)
{
	tan_para_singlefile(mesh_dir_path, filename, start_level, level_num, grid_length, grid_size);
	hrch_axis_corr_singlefile(mesh_dir_path, filename, start_level, level_num);
}