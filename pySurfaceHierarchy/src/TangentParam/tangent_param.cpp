#include "tangent_param.h"
#include <unordered_set>
#include <queue>
#include "DT2D\delaunay2d.h"
#include <MathSuite/SPARSE/Sparse_Matrix.h>
#include "param_util.h"
#include "ANNSearch.hpp"

bool is_nan(float x) {
#ifdef WIN32
	return (_isnan(x) != 0) || (_finite(x) == 0);
#else
	return isnan(x) || !finite(x);
#endif
}

bool is_nan(double x) {
#ifdef WIN32
	return (_isnan(x) != 0) || (_finite(x) == 0);
#else
	return isnan(x) || !finite(x);
#endif
}

inline double robust_acos(double cos_val) {

	if (is_nan(cos_val))
	{
		return 0;
	}

	double acos_val = std::acos(cos_val);
	while (is_nan(acos_val))
	{
		if (cos_val > 1.0)
		{
			cos_val -= 1.0e-10;
		}
		else if (cos_val < -1.0)
		{
			cos_val += 1.0e-10;
		}
		acos_val = std::acos(cos_val);
	}

	return acos_val;
}

double get_arc_length(double chord, double cos_theta) {
	double theta = robust_acos(cos_theta);
	if (theta<1e-3) return chord;
	else return chord*0.5 / std::sin(theta*0.5) * theta;
}

//#define _USE_SCALED_FRAMES_

void tangent_param_with_patchinput(TriMesh* mesh, const std::vector<std::vector<TriMesh::Point>>& axes,
	const double scale, const int gm, const int gn, const bool use_patch_height,
	std::vector<int>& vtx_indices, std::vector<int>& axis_indices, std::vector<double>& baryweights,
	std::vector<double>& patch_input_feature) {

	const int vnum = mesh->n_vertices();
	const int anum = axes[0].size();
	vtx_indices.resize(vnum*anum*gm*gn * 3, -1);
	axis_indices.resize(vtx_indices.size(), -1);
	baryweights.resize(vtx_indices.size(), 0.);
	patch_input_feature.resize(vnum*anum*gm*gn*(use_patch_height ? 4 : 3), 0.); //patch input feature contains patch-wise normal, and optional height
	const double crop_radius = scale*0.708 * 2;

	size_t total_masked_ptnum = 0;

#pragma omp parallel for
	for (int i = 0; i < vnum; i++)
	{
		//different axes have rotational symmetry, so we compute the 0-th axis parameterization only,
		// and find the sampling data for all axes by symmetry.

		struct pt_frame {
			int vidx;
			OpenMesh::Vec2d pos_2d;
			int frame_idx;
			pt_frame(int vtx, OpenMesh::Vec2d& pt2d, int fidx) {
				vidx = vtx; pos_2d = pt2d; frame_idx = fidx;
			}
		};

		TriMesh::Point vpos = mesh->point(mesh->vertex_handle(i));
		TriMesh::Point vframe[2] = { axes[i][0], OpenMesh::cross(mesh->normal(mesh->vertex_handle(i)), axes[i][0]) };
		TriMesh::Point vNormal = mesh->normal(mesh->vertex_handle(i));

		std::vector<pt_frame> cropped_vts;
		std::unordered_set<int> visited_vts; visited_vts.insert(i);
		std::queue<pt_frame> Q;
		Q.push(pt_frame(i, OpenMesh::Vec2d(0, 0), 0));
		while (!Q.empty())
		{
			auto cur_pt = Q.front(); Q.pop();
			cropped_vts.push_back(cur_pt);
			auto cur_vh = mesh->vertex_handle(cur_pt.vidx);

			if ((mesh->point(cur_vh) - vpos).length() > crop_radius || cur_pt.pos_2d.length() > crop_radius)
			{
				continue;
			}

			for (TriMesh::VertexVertexIter vv_it = mesh->vv_iter(cur_vh); vv_it.is_valid(); vv_it++)
			{
				//printf("visiting nb vtx %d\n", vv_it->idx());
				if (visited_vts.find(vv_it->idx()) == visited_vts.end())
				{
					//printf("nb vtx %d is not visited before\n", vv_it->idx());
					visited_vts.insert(vv_it->idx());

					//get nb frame axis
					int fdelta = get_axis_map_p2p(mesh->normal(cur_vh), mesh->normal(vv_it), anum, axes[cur_vh.idx()], axes[vv_it->idx()]);
					int nb_fidx = (cur_pt.frame_idx + fdelta) % anum;

					//printf("nb vtx %d has dist %f, in range\n", vv_it->idx(), dist);
					TriMesh::Point edge3d = mesh->point(vv_it) - mesh->point(cur_vh);
					OpenMesh::Vec2d edge2d_pq(OpenMesh::dot(edge3d, axes[cur_vh.idx()][cur_pt.frame_idx]),
						OpenMesh::dot(edge3d, cross(mesh->normal(cur_vh), axes[cur_vh.idx()][cur_pt.frame_idx]))),
						edge2d_qp(OpenMesh::dot(edge3d, axes[vv_it->idx()][nb_fidx]),
							OpenMesh::dot(edge3d, cross(mesh->normal(vv_it), axes[vv_it->idx()][nb_fidx])));
					auto edge2d = 0.5*(edge2d_pq + edge2d_qp) + cur_pt.pos_2d;

					if (!is_valid(edge2d[0]) || !is_valid(edge2d[1]))
					{
						std::cout << "Invalid 2D projection.\n"; //std::exit(0);
					}
					else
					{
						Q.push(pt_frame(vv_it->idx(), edge2d, nb_fidx)); //printf("queued nb vtx %d\n", vv_it->idx());
					}

				}
				//std::cout<<"checking next edge in 1-ring..."<<( he->pair->next?"valid edge":"pair->next is null" );
			}
		}

		//std::cout << "Found " << cropped_vts.size() << " vts." << std::endl;

		//sample the grids for different axes
		if (cropped_vts.size()<3)
		{
			printf("Less than 3 nb verts.\n");
			for (int axis = 0; axis < anum; axis++)
			{
				for (int row = 0; row < gm; row++)
				{
					for (int col = 0; col < gn; col++) {
						const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
						for (int d = 0; d < 3; d++) {
							vtx_indices[offset3 + d] = i;
							axis_indices[offset3 + d] = 0;
							baryweights[offset3 + d] = 1. / 3.0;
						}
						if (use_patch_height) {
							const int offset4 = (((i*anum + axis)*gm + row)*gn + col) * 4;
							patch_input_feature[offset4] = 1;
							patch_input_feature[offset4 + 1] = 0;
							patch_input_feature[offset4 + 2] = 0;
							patch_input_feature[offset4 + 3] = 0;
						}
						else
						{
							patch_input_feature[offset3] = 1;
							patch_input_feature[offset3 + 1] = 0;
							patch_input_feature[offset3 + 2] = 0;
						}
					}
				}
			}
		}
		else
		{
			//triangulate the projected pts
			std::vector<OpenMesh::Vec2d> proj_pts(cropped_vts.size());
			for (size_t ptitr = 0; ptitr < cropped_vts.size(); ptitr++)
			{
				proj_pts[ptitr] = cropped_vts[ptitr].pos_2d;
			}
			std::vector<std::vector<int>> proj_dt;
			Delaunay2D dt2d;
			if (!dt2d.Triangulate(proj_pts, proj_dt))
			{
				/*for (int kkk = 0; kkk < cropped_vts.size(); kkk++)
				{
					std::cout << cropped_vts[kkk].vidx << std::endl;
					std::cout << mesh->point(mesh->vertex_handle(cropped_vts[kkk].vidx)) << std::endl;
					system("pause");
				}*/
				for (int axis = 0; axis < anum; axis++)
				{
					for (int row = 0; row < gm; row++)
					{
						for (int col = 0; col < gn; col++) {
							const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
							for (int d = 0; d < 3; d++) {
								vtx_indices[offset3 + d] = i;
								axis_indices[offset3 + d] = 0;
								baryweights[offset3 + d] = 1. / 3.0;
							}
							if (use_patch_height) {
								const int offset4 = (((i*anum + axis)*gm + row)*gn + col) * 4;
								patch_input_feature[offset4] = 1;
								patch_input_feature[offset4 + 1] = 0;
								patch_input_feature[offset4 + 2] = 0;
								patch_input_feature[offset4 + 3] = 0;
							}
							else
							{
								patch_input_feature[offset3] = 1;
								patch_input_feature[offset3 + 1] = 0;
								patch_input_feature[offset3 + 2] = 0;
							}
						}
					}
				}
				continue;
			}
			const double row_stride = scale / (gm - 1), col_stride = scale / (gn - 1);
			const OpenMesh::Vec2d ll_corner(-col_stride*(gn - 1) / 2.0, -row_stride*(gm - 1) / 2.0);

			struct mat2x2 {
				double data[2][2] = { 0. };
				mat2x2(double* d) {
					data[0][0] = d[0]; data[0][1] = d[1]; data[1][0] = d[2]; data[1][1] = d[3];
					//std::cout << "mat2x2:" << std::endl;
					//std::cout << data[0][0] << " " << data[0][1] << " " << data[1][0] << " " << data[1][1] << std::endl;
				}
				OpenMesh::Vec2d& operator()(OpenMesh::Vec2d& a) const {
					return OpenMesh::Vec2d(data[0][0] * a[0] + data[0][1] * a[1], data[1][0] * a[0] + data[1][1] * a[1]);
				}
			};
			auto map_between_axis = [](const TriMesh::Point& b, const TriMesh::Point& a, const TriMesh::Point& normal)->mat2x2 {
				TriMesh::Point b_perp = OpenMesh::cross(normal, b), a_perp = OpenMesh::cross(normal, a);
				double mat[4] = { OpenMesh::dot(b,a), OpenMesh::dot(b_perp,a), OpenMesh::dot(b,a_perp), OpenMesh::dot(b_perp,a_perp) };
				//std::cout << "mat4" << std::endl;
				//std::cout << mat[0] << " " << mat[1] << " " << mat[2] << " " << mat[3] << std::endl;
				return mat2x2(mat);
			};

			for (int axis = 0; axis < anum; axis++)
			{
				const auto map_0th = map_between_axis(axes[i][axis], axes[i][0], vNormal);
				//std::cout << "map_0th.data" << std::endl;
				//std::cout << map_0th.data[0][0] << " " << map_0th.data[0][1] << " " << map_0th.data[1][0] << " " << map_0th.data[1][1] << std::endl;
				int masked_ptnum = 0;
				for (int row = 0; row < gm; row++)
				{
					for (int col = 0; col < gn; col++)
					{
						OpenMesh::Vec2d smp_pt = ll_corner;
						smp_pt[0] += col*col_stride;
						smp_pt[1] += (gm - 1 - row)*row_stride;
						smp_pt = map_0th(smp_pt);

						auto triarea2d = [](OpenMesh::Vec2d& v0, OpenMesh::Vec2d& v1, OpenMesh::Vec2d& v2) -> double {
							auto v01 = v1 - v0;
							auto v02 = v2 - v0;
							return v01[0] * v02[1] - v01[1] * v02[0];
						};
						//find the triangle containing it
						int cntn_tri = -1;
						double cntn_tri_ws[3] = { 0 };
						for (size_t tri_itr = 0; tri_itr < proj_dt.size(); tri_itr++)
						{
							double tri_ws[4] = {
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][1]], proj_pts[proj_dt[tri_itr][2]]), //1,2
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][2]], proj_pts[proj_dt[tri_itr][0]]), //2,0
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][0]], proj_pts[proj_dt[tri_itr][1]]), //0,1
								triarea2d(proj_pts[proj_dt[tri_itr][0]], proj_pts[proj_dt[tri_itr][1]], proj_pts[proj_dt[tri_itr][2]]) //0,1,2
							};
							//std::cout << tri_ws[0] << " " << tri_ws[1] << " " << tri_ws[2] << " " << tri_ws[3] << std::endl;
							tri_ws[0] /= tri_ws[3]; tri_ws[1] /= tri_ws[3]; tri_ws[2] /= tri_ws[3];
							if (tri_ws[0] >= 0. && tri_ws[1] >= 0. && tri_ws[2] >= 0.)
							{
								cntn_tri = tri_itr;
								std::copy(tri_ws, tri_ws + 3, cntn_tri_ws);
								break;
							}
						}

						const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
						const int offset4 = (((i*anum + axis)*gm + row)*gn + col) * 4;
						if (cntn_tri>-1)
						{
							for (int d = 0; d < 3; d++) {
								vtx_indices[offset3 + d] = cropped_vts[proj_dt[cntn_tri][d]].vidx;
								axis_indices[offset3 + d] = (cropped_vts[proj_dt[cntn_tri][d]].frame_idx + axis) % anum;
								baryweights[offset3 + d] = cntn_tri_ws[d];
							}

							double smpl_h = 0;
							TriMesh::Point smpl_nml(0, 0, 0);
							for (int tri_vtx = 0; tri_vtx < 3; tri_vtx++)
							{
								auto vh = mesh->vertex_handle(cropped_vts[proj_dt[cntn_tri][tri_vtx]].vidx);
								smpl_h += cntn_tri_ws[tri_vtx] * OpenMesh::dot(mesh->point(vh) - vpos, vNormal) / scale;
								smpl_nml += cntn_tri_ws[tri_vtx] * TriMesh::Point(dot(mesh->normal(vh), vNormal),
									OpenMesh::dot(mesh->normal(vh), axes[i][axis]),
									OpenMesh::dot(mesh->normal(vh), cross(vNormal, axes[i][axis])));
							}
							if (use_patch_height) {
								patch_input_feature[offset4] = smpl_nml[0];
								patch_input_feature[offset4 + 1] = smpl_nml[1];
								patch_input_feature[offset4 + 2] = smpl_nml[2];
								patch_input_feature[offset4 + 3] = smpl_h;
							}
							else
							{
								patch_input_feature[offset3] = smpl_nml[0];
								patch_input_feature[offset3 + 1] = smpl_nml[1];
								patch_input_feature[offset3 + 2] = smpl_nml[2];
							}
						}
						else
						{
							/*std::ofstream dump("V" + std::to_string(i) + "_projdump.obj");
							for (size_t pt_itr = 0; pt_itr < proj_pts.size(); pt_itr++)
							{
							dump << "v " << proj_pts[pt_itr][0] << " " << proj_pts[pt_itr][1] << " " << "0\n";
							}
							dump << "v " << smp_pt[0] << " " << smp_pt[1] << " " << "0\n";
							for (size_t tri_itr = 0; tri_itr < proj_dt.size(); tri_itr++)
							{
							dump << "f " << proj_dt[tri_itr][0] + 1 << " " << proj_dt[tri_itr][1] + 1 << " " << proj_dt[tri_itr][2] + 1 << std::endl;
							}
							dump.close();*/
#pragma omp atomic
							total_masked_ptnum++;

							masked_ptnum++;
							//printf("Failed to find containing nondegenerate triangle! Vtx, axis, row, col: %d, %d, %d, %d\n", i, axis, row, col);
							for (int d = 0; d < 3; d++) {
								vtx_indices[offset3 + d] = i;
								axis_indices[offset3 + d] = 0;
								//baryweights[offset3 + d] = 0.0; initialized to 0.
							}
						}
					}
				}

				////compensate masked grid points energy
				//const double comp_scale = double(gm*gn) / double(gm*gn - masked_ptnum);
				//for (int row = 0; row < gm; row++)
				//{
				//	for (int col = 0; col < gn; col++)
				//	{
				//		const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
				//		for (int d = 0; d < 3; d++)
				//			baryweights[offset3 + d] *= comp_scale;
				//	}
				//}
			}
		}
	}

	std::cout << "Masked grid point num: " << total_masked_ptnum << " out of " << (vnum*anum*gm*gn) << std::endl;
}

void tangent_param_with_patchinput_NG(TriMesh* mesh, const std::vector<std::vector<TriMesh::Point>>& axes,
	const double scale, const int gm, const int gn, const bool use_patch_height,
	std::vector<int>& vtx_indices, std::vector<int>& axis_indices, std::vector<double>& baryweights,
	std::vector<double>& patch_input_feature) {

	const int vnum = mesh->n_vertices();
	const int anum = axes[0].size();
	vtx_indices.resize(vnum*anum*gm*gn * 3, -1);
	axis_indices.resize(vtx_indices.size(), -1);
	baryweights.resize(vtx_indices.size(), 0.);
	patch_input_feature.resize(vnum*anum*gm*gn*(use_patch_height ? 4 : 3), 0.); //patch input feature contains patch-wise normal, and optional height
	const double crop_radius = scale * 0.708 * 2;

	size_t total_masked_ptnum = 0;

#pragma omp parallel for
	for (int i = 0; i < vnum; i++)
	{
		//different axes have rotational symmetry, so we compute the 0-th axis parameterization only,
		// and find the sampling data for all axes by symmetry.

		struct pt_frame {
			int vidx;
			OpenMesh::Vec2d pos_2d;
			int frame_idx;
			pt_frame(int vtx, OpenMesh::Vec2d& pt2d, int fidx) {
				vidx = vtx; pos_2d = pt2d; frame_idx = fidx;
			}
		};

		TriMesh::Point vpos = mesh->point(mesh->vertex_handle(i));
		TriMesh::Point vframe[2] = { axes[i][0], OpenMesh::cross(mesh->normal(mesh->vertex_handle(i)), axes[i][0]) };
		TriMesh::Point vNormal = mesh->normal(mesh->vertex_handle(i));

		std::vector<pt_frame> cropped_vts;
		std::unordered_set<int> visited_vts; visited_vts.insert(i);
		std::queue<pt_frame> Q;
		Q.push(pt_frame(i, OpenMesh::Vec2d(0, 0), 0));
		while (!Q.empty())
		{
			auto cur_pt = Q.front(); Q.pop();
			cropped_vts.push_back(cur_pt);
			auto cur_vh = mesh->vertex_handle(cur_pt.vidx);

			if ((mesh->point(cur_vh) - vpos).length() > crop_radius || cur_pt.pos_2d.length() > crop_radius)
			{
				continue;
			}

			for (TriMesh::VertexVertexIter vv_it = mesh->vv_iter(cur_vh); vv_it.is_valid(); vv_it++)
			{
				//printf("visiting nb vtx %d\n", vv_it->idx());
				if (visited_vts.find(vv_it->idx()) == visited_vts.end())
				{
					//printf("nb vtx %d is not visited before\n", vv_it->idx());
					visited_vts.insert(vv_it->idx());

					//get nb frame axis
					int fdelta = get_axis_map_p2p(mesh->normal(cur_vh), mesh->normal(vv_it), anum, axes[cur_vh.idx()], axes[vv_it->idx()]);
					int nb_fidx = (cur_pt.frame_idx + fdelta) % anum;

					//printf("nb vtx %d has dist %f, in range\n", vv_it->idx(), dist);
					TriMesh::Point edge3d = mesh->point(vv_it) - mesh->point(cur_vh);
					OpenMesh::Vec2d edge2d_pq(OpenMesh::dot(edge3d, axes[cur_vh.idx()][cur_pt.frame_idx]),
						OpenMesh::dot(edge3d, cross(mesh->normal(cur_vh), axes[cur_vh.idx()][cur_pt.frame_idx])));
					auto edge2d = edge2d_pq + cur_pt.pos_2d;

					if (!is_valid(edge2d[0]) || !is_valid(edge2d[1]))
					{
						std::cout << "Invalid 2D projection.\n"; //std::exit(0);
					}
					else
					{
						Q.push(pt_frame(vv_it->idx(), edge2d, nb_fidx)); //printf("queued nb vtx %d\n", vv_it->idx());
					}

				}
				//std::cout<<"checking next edge in 1-ring..."<<( he->pair->next?"valid edge":"pair->next is null" );
			}
		}

		//std::cout << "Found " << cropped_vts.size() << " vts." << std::endl;

		//sample the grids for different axes
		if (cropped_vts.size() < 3)
		{
			printf("Less than 3 nb verts.\n");
			for (int axis = 0; axis < anum; axis++)
			{
				for (int row = 0; row < gm; row++)
				{
					for (int col = 0; col < gn; col++) {
						const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
						for (int d = 0; d < 3; d++) {
							vtx_indices[offset3 + d] = i;
							axis_indices[offset3 + d] = 0;
							baryweights[offset3 + d] = 1. / 3.0;
						}
						if (use_patch_height) {
							const int offset4 = (((i*anum + axis)*gm + row)*gn + col) * 4;
							patch_input_feature[offset4] = 1;
							patch_input_feature[offset4 + 1] = 0;
							patch_input_feature[offset4 + 2] = 0;
							patch_input_feature[offset4 + 3] = 0;
						}
						else
						{
							patch_input_feature[offset3] = 1;
							patch_input_feature[offset3 + 1] = 0;
							patch_input_feature[offset3 + 2] = 0;
						}
					}
				}
			}
		}
		else
		{
			//triangulate the projected pts
			std::vector<OpenMesh::Vec2d> proj_pts(cropped_vts.size());
			for (size_t ptitr = 0; ptitr < cropped_vts.size(); ptitr++)
				proj_pts[ptitr] = cropped_vts[ptitr].pos_2d;
			std::vector<std::vector<int>> proj_dt;
			Delaunay2D dt2d(proj_pts, proj_dt);

			const double row_stride = scale / (gm - 1), col_stride = scale / (gn - 1);
			const OpenMesh::Vec2d ll_corner(-col_stride * (gn - 1) / 2.0, -row_stride * (gm - 1) / 2.0);

			struct mat2x2 {
				double data[2][2] = { 0. };
				mat2x2(double* d) {
					data[0][0] = d[0]; data[0][1] = d[1]; data[1][0] = d[2]; data[1][1] = d[3];
					//std::cout << "mat2x2:" << std::endl;
					//std::cout << data[0][0] << " " << data[0][1] << " " << data[1][0] << " " << data[1][1] << std::endl;
				}
				OpenMesh::Vec2d& operator()(OpenMesh::Vec2d& a) const {
					return OpenMesh::Vec2d(data[0][0] * a[0] + data[0][1] * a[1], data[1][0] * a[0] + data[1][1] * a[1]);
				}
			};
			auto map_between_axis = [](const TriMesh::Point& b, const TriMesh::Point& a, const TriMesh::Point& normal)->mat2x2 {
				TriMesh::Point b_perp = OpenMesh::cross(normal, b), a_perp = OpenMesh::cross(normal, a);
				double mat[4] = { OpenMesh::dot(b,a), OpenMesh::dot(b_perp,a), OpenMesh::dot(b,a_perp), OpenMesh::dot(b_perp,a_perp) };
				//std::cout << "mat4" << std::endl;
				//std::cout << mat[0] << " " << mat[1] << " " << mat[2] << " " << mat[3] << std::endl;
				return mat2x2(mat);
			};

			for (int axis = 0; axis < anum; axis++)
			{
				const auto map_0th = map_between_axis(axes[i][axis], axes[i][0], vNormal);
				//std::cout << "map_0th.data" << std::endl;
				//std::cout << map_0th.data[0][0] << " " << map_0th.data[0][1] << " " << map_0th.data[1][0] << " " << map_0th.data[1][1] << std::endl;
				int masked_ptnum = 0;
				for (int row = 0; row < gm; row++)
				{
					for (int col = 0; col < gn; col++)
					{
						OpenMesh::Vec2d smp_pt = ll_corner;
						smp_pt[0] += col * col_stride;
						smp_pt[1] += (gm - 1 - row)*row_stride;
						smp_pt = map_0th(smp_pt);

						auto triarea2d = [](OpenMesh::Vec2d& v0, OpenMesh::Vec2d& v1, OpenMesh::Vec2d& v2) -> double {
							auto v01 = v1 - v0;
							auto v02 = v2 - v0;
							return v01[0] * v02[1] - v01[1] * v02[0];
						};
						//find the triangle containing it
						int cntn_tri = -1;
						double cntn_tri_ws[3] = { 0 };
						for (size_t tri_itr = 0; tri_itr < proj_dt.size(); tri_itr++)
						{
							double tri_ws[4] = {
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][1]], proj_pts[proj_dt[tri_itr][2]]), //1,2
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][2]], proj_pts[proj_dt[tri_itr][0]]), //2,0
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][0]], proj_pts[proj_dt[tri_itr][1]]), //0,1
								triarea2d(proj_pts[proj_dt[tri_itr][0]], proj_pts[proj_dt[tri_itr][1]], proj_pts[proj_dt[tri_itr][2]]) //0,1,2
							};
							//std::cout << tri_ws[0] << " " << tri_ws[1] << " " << tri_ws[2] << " " << tri_ws[3] << std::endl;
							tri_ws[0] /= tri_ws[3]; tri_ws[1] /= tri_ws[3]; tri_ws[2] /= tri_ws[3];
							if (tri_ws[0] >= 0. && tri_ws[1] >= 0. && tri_ws[2] >= 0.)
							{
								cntn_tri = tri_itr;
								std::copy(tri_ws, tri_ws + 3, cntn_tri_ws);
								break;
							}
						}

						const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
						const int offset4 = (((i*anum + axis)*gm + row)*gn + col) * 4;
						if (cntn_tri > -1)
						{
							for (int d = 0; d < 3; d++) {
								vtx_indices[offset3 + d] = cropped_vts[proj_dt[cntn_tri][d]].vidx;
								axis_indices[offset3 + d] = (cropped_vts[proj_dt[cntn_tri][d]].frame_idx + axis) % anum;
								baryweights[offset3 + d] = cntn_tri_ws[d];
							}

							double smpl_h = 0;
							TriMesh::Point smpl_nml(0, 0, 0);
							for (int tri_vtx = 0; tri_vtx < 3; tri_vtx++)
							{
								auto vh = mesh->vertex_handle(cropped_vts[proj_dt[cntn_tri][tri_vtx]].vidx);
								smpl_h += cntn_tri_ws[tri_vtx] * OpenMesh::dot(mesh->point(vh) - vpos, vNormal) / scale;
								smpl_nml += cntn_tri_ws[tri_vtx] * TriMesh::Point(dot(mesh->normal(vh), vNormal),
									OpenMesh::dot(mesh->normal(vh), axes[i][axis]),
									OpenMesh::dot(mesh->normal(vh), cross(vNormal, axes[i][axis])));
							}
							if (use_patch_height) {
								patch_input_feature[offset4] = smpl_nml[0];
								patch_input_feature[offset4 + 1] = smpl_nml[1];
								patch_input_feature[offset4 + 2] = smpl_nml[2];
								patch_input_feature[offset4 + 3] = smpl_h;
							}
							else
							{
								patch_input_feature[offset3] = smpl_nml[0];
								patch_input_feature[offset3 + 1] = smpl_nml[1];
								patch_input_feature[offset3 + 2] = smpl_nml[2];
							}
						}
						else
						{
							/*std::ofstream dump("V" + std::to_string(i) + "_projdump.obj");
							for (size_t pt_itr = 0; pt_itr < proj_pts.size(); pt_itr++)
							{
							dump << "v " << proj_pts[pt_itr][0] << " " << proj_pts[pt_itr][1] << " " << "0\n";
							}
							dump << "v " << smp_pt[0] << " " << smp_pt[1] << " " << "0\n";
							for (size_t tri_itr = 0; tri_itr < proj_dt.size(); tri_itr++)
							{
							dump << "f " << proj_dt[tri_itr][0] + 1 << " " << proj_dt[tri_itr][1] + 1 << " " << proj_dt[tri_itr][2] + 1 << std::endl;
							}
							dump.close();*/
#pragma omp atomic
							total_masked_ptnum++;

							masked_ptnum++;
							//printf("Failed to find containing nondegenerate triangle! Vtx, axis, row, col: %d, %d, %d, %d\n", i, axis, row, col);
							for (int d = 0; d < 3; d++) {
								vtx_indices[offset3 + d] = i;
								axis_indices[offset3 + d] = 0;
								//baryweights[offset3 + d] = 0.0; initialized to 0.
							}
						}
					}
				}

				////compensate masked grid points energy
				//const double comp_scale = double(gm*gn) / double(gm*gn - masked_ptnum);
				//for (int row = 0; row < gm; row++)
				//{
				//	for (int col = 0; col < gn; col++)
				//	{
				//		const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
				//		for (int d = 0; d < 3; d++)
				//			baryweights[offset3 + d] *= comp_scale;
				//	}
				//}
			}
		}
	}

	std::cout << "Masked grid point num: " << total_masked_ptnum << " out of " << (vnum*anum*gm*gn) << std::endl;
}



void tangent_param_with_patchinput_pointcloud(std::vector<TriMesh::Point> &pointcloud, std::vector<TriMesh::Point> &pc_normal, const std::vector<std::vector<TriMesh::Point>>& axes,
	const double scale, const int gm, const int gn, const bool use_patch_height,
	std::vector<int>& vtx_indices, std::vector<int>& axis_indices, std::vector<double>& baryweights,
	std::vector<double>& patch_input_feature) {

	const int vnum = pointcloud.size();
	const int anum = axes[0].size();
	vtx_indices.resize(vnum*anum*gm*gn * 3, -1);
	axis_indices.resize(vtx_indices.size(), -1);
	baryweights.resize(vtx_indices.size(), 0.);
	patch_input_feature.resize(vnum*anum*gm*gn*(use_patch_height ? 4 : 3), 0.); //patch input feature contains patch-wise normal, and optional height
	const double crop_radius = scale * 0.708 * 2;
	const int crop_k = 6;
	size_t total_masked_ptnum = 0;

	for (int i = 0; i < pc_normal.size(); i++)
		pc_normal[i].normalize();

	surface_hierarchy::ANN_Search ann_search(pointcloud);

//#pragma omp parallel for schedule(static,1) num_threads(6)
	for (int i = 0; i < vnum; i++)
	{
		//different axes have rotational symmetry, so we compute the 0-th axis parameterization only,
		// and find the sampling data for all axes by symmetry.

		struct pt_frame {
			int vidx;
			OpenMesh::Vec2d pos_2d;
			int frame_idx;
			pt_frame(int vtx, OpenMesh::Vec2d& pt2d, int fidx) {
				vidx = vtx; pos_2d = pt2d; frame_idx = fidx;
			}
		};

		TriMesh::Point vpos = pointcloud[i];
		TriMesh::Point vframe[2] = { axes[i][0], OpenMesh::cross(pc_normal[i], axes[i][0]) };
		TriMesh::Point vNormal = pc_normal[i];

		std::vector<pt_frame> cropped_vts;

		std::vector<int> new_nbs;
		ann_search.find_pts_in_radius(pointcloud[i], crop_radius * 5, new_nbs);

		/*
		// filter by normal
		std::vector<int> filtered_nbs;
		filtered_nbs.clear();
		for (auto nb : new_nbs)
		{
			if (OpenMesh::dot(pc_normal[nb], pc_normal[i]) > -0.5)
			{
				filtered_nbs.push_back(nb);
			}
		}
		new_nbs.assign(filtered_nbs.begin(), filtered_nbs.end());

		*/
		if(new_nbs.size() < 3)
		{
			new_nbs.clear();
			std::vector<double> new_nb_dists;
			ann_search.find_k_nbs(pointcloud[i], crop_k, new_nbs, new_nb_dists);
		}

		new_nbs.erase(std::remove(new_nbs.begin(), new_nbs.end(), i), new_nbs.end()); //remove itself

		auto cur_pt = pt_frame(i, OpenMesh::Vec2d(0, 0), 0);
		cropped_vts.push_back(cur_pt);

		for (int j = 0; j < new_nbs.size(); j++)
		{
			//get nb frame axis
			int fdelta = get_axis_map_p2p(vNormal, pc_normal[new_nbs[j]], anum, axes[i], axes[new_nbs[j]]);
			int nb_fidx = (cur_pt.frame_idx + fdelta) % anum;

			TriMesh::Point edge3d = pointcloud[new_nbs[j]] - vpos;
			OpenMesh::Vec2d edge2d_pq(OpenMesh::dot(edge3d, axes[i][cur_pt.frame_idx]), OpenMesh::dot(edge3d, cross(vNormal, axes[i][cur_pt.frame_idx]))),
				edge2d_qp(OpenMesh::dot(edge3d, axes[new_nbs[j]][nb_fidx]), OpenMesh::dot(edge3d, cross(pc_normal[new_nbs[j]], axes[new_nbs[j]][nb_fidx])));
			auto edge2d = 0.5*(edge2d_pq + edge2d_qp) + cur_pt.pos_2d;

			if (!is_valid(edge2d[0]) || !is_valid(edge2d[1]))
			{
				std::cout << "Invalid 2D projection.\n"; //std::exit(0);
			}
			else
			{
				cropped_vts.push_back(pt_frame(new_nbs[j], edge2d, nb_fidx)); //printf("queued nb vtx %d\n", vv_it->idx());
			}
		}
		//std::cout << "Found " << cropped_vts.size() << " vts." << std::endl;

		//sample the grids for different axes
		{
			//triangulate the projected pts
			std::vector<OpenMesh::Vec2d> proj_pts(cropped_vts.size());
			for (size_t ptitr = 0; ptitr < cropped_vts.size(); ptitr++)
				proj_pts[ptitr] = cropped_vts[ptitr].pos_2d;
			std::vector<std::vector<int>> proj_dt;
			Delaunay2D dt2d(proj_pts, proj_dt);

			const double row_stride = scale / (gm - 1), col_stride = scale / (gn - 1);
			const OpenMesh::Vec2d ll_corner(-col_stride * (gn - 1) / 2.0, -row_stride * (gm - 1) / 2.0);

			struct mat2x2 {
				double data[2][2] = { 0. };
				mat2x2(double* d) {
					data[0][0] = d[0]; data[0][1] = d[1]; data[1][0] = d[2]; data[1][1] = d[3];
					//std::cout << "mat2x2:" << std::endl;
					//std::cout << data[0][0] << " " << data[0][1] << " " << data[1][0] << " " << data[1][1] << std::endl;
				}
				OpenMesh::Vec2d& operator()(OpenMesh::Vec2d& a) const {
					return OpenMesh::Vec2d(data[0][0] * a[0] + data[0][1] * a[1], data[1][0] * a[0] + data[1][1] * a[1]);
				}
			};
			auto map_between_axis = [](const TriMesh::Point& b, const TriMesh::Point& a, const TriMesh::Point& normal)->mat2x2 {
				TriMesh::Point b_perp = OpenMesh::cross(normal, b), a_perp = OpenMesh::cross(normal, a);
				double mat[4] = { OpenMesh::dot(b,a), OpenMesh::dot(b_perp,a), OpenMesh::dot(b,a_perp), OpenMesh::dot(b_perp,a_perp) };
				//std::cout << "mat4" << std::endl;
				//std::cout << mat[0] << " " << mat[1] << " " << mat[2] << " " << mat[3] << std::endl;
				return mat2x2(mat);
			};

			for (int axis = 0; axis < anum; axis++)
			{
				const auto map_0th = map_between_axis(axes[i][axis], axes[i][0], vNormal);
				//std::cout << "map_0th.data" << std::endl;
				//std::cout << map_0th.data[0][0] << " " << map_0th.data[0][1] << " " << map_0th.data[1][0] << " " << map_0th.data[1][1] << std::endl;
				int masked_ptnum = 0;
				for (int row = 0; row < gm; row++)
				{
					for (int col = 0; col < gn; col++)
					{
						OpenMesh::Vec2d smp_pt = ll_corner;
						smp_pt[0] += col * col_stride;
						smp_pt[1] += (gm - 1 - row)*row_stride;
						smp_pt = map_0th(smp_pt);

						auto triarea2d = [](OpenMesh::Vec2d& v0, OpenMesh::Vec2d& v1, OpenMesh::Vec2d& v2) -> double {
							auto v01 = v1 - v0;
							auto v02 = v2 - v0;
							return v01[0] * v02[1] - v01[1] * v02[0];
						};
						//find the triangle containing it
						int cntn_tri = -1;
						double cntn_tri_ws[3] = { 0 };
						for (size_t tri_itr = 0; tri_itr < proj_dt.size(); tri_itr++)
						{
							double tri_ws[4] = {
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][1]], proj_pts[proj_dt[tri_itr][2]]), //1,2
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][2]], proj_pts[proj_dt[tri_itr][0]]), //2,0
								triarea2d(smp_pt, proj_pts[proj_dt[tri_itr][0]], proj_pts[proj_dt[tri_itr][1]]), //0,1
								triarea2d(proj_pts[proj_dt[tri_itr][0]], proj_pts[proj_dt[tri_itr][1]], proj_pts[proj_dt[tri_itr][2]]) //0,1,2
							};
							//std::cout << tri_ws[0] << " " << tri_ws[1] << " " << tri_ws[2] << " " << tri_ws[3] << std::endl;
							tri_ws[0] /= tri_ws[3]; tri_ws[1] /= tri_ws[3]; tri_ws[2] /= tri_ws[3];
							if (tri_ws[0] >= 0. && tri_ws[1] >= 0. && tri_ws[2] >= 0.)
							{
								cntn_tri = tri_itr;
								std::copy(tri_ws, tri_ws + 3, cntn_tri_ws);
								break;
							}
						}

						const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
						const int offset4 = (((i*anum + axis)*gm + row)*gn + col) * 4;
						if (cntn_tri > -1)
						{
							for (int d = 0; d < 3; d++) {
								vtx_indices[offset3 + d] = cropped_vts[proj_dt[cntn_tri][d]].vidx;
								axis_indices[offset3 + d] = (cropped_vts[proj_dt[cntn_tri][d]].frame_idx + axis) % anum;
								baryweights[offset3 + d] = cntn_tri_ws[d];
							}

							double smpl_h = 0;
							TriMesh::Point smpl_nml(0, 0, 0);
							for (int tri_vtx = 0; tri_vtx < 3; tri_vtx++)
							{
								int vidx = cropped_vts[proj_dt[cntn_tri][tri_vtx]].vidx;
								smpl_h += cntn_tri_ws[tri_vtx] * OpenMesh::dot(pointcloud[vidx] - vpos, vNormal) / scale;
								smpl_nml += cntn_tri_ws[tri_vtx] * TriMesh::Point(dot(pc_normal[vidx], vNormal),
									OpenMesh::dot(pc_normal[vidx], axes[i][axis]),
									OpenMesh::dot(pc_normal[vidx], cross(vNormal, axes[i][axis])));
							}
							if (use_patch_height) {
								patch_input_feature[offset4] = smpl_nml[0];
								patch_input_feature[offset4 + 1] = smpl_nml[1];
								patch_input_feature[offset4 + 2] = smpl_nml[2];
								patch_input_feature[offset4 + 3] = smpl_h;
							}
							else
							{
								patch_input_feature[offset3] = smpl_nml[0];
								patch_input_feature[offset3 + 1] = smpl_nml[1];
								patch_input_feature[offset3 + 2] = smpl_nml[2];
							}
						}
						else
						{
							/*std::ofstream dump("V" + std::to_string(i) + "_projdump.obj");
							for (size_t pt_itr = 0; pt_itr < proj_pts.size(); pt_itr++)
							{
							dump << "v " << proj_pts[pt_itr][0] << " " << proj_pts[pt_itr][1] << " " << "0\n";
							}
							dump << "v " << smp_pt[0] << " " << smp_pt[1] << " " << "0\n";
							for (size_t tri_itr = 0; tri_itr < proj_dt.size(); tri_itr++)
							{
							dump << "f " << proj_dt[tri_itr][0] + 1 << " " << proj_dt[tri_itr][1] + 1 << " " << proj_dt[tri_itr][2] + 1 << std::endl;
							}
							dump.close();*/
//#pragma omp atomic
							total_masked_ptnum++;

							masked_ptnum++;
							//printf("Failed to find containing nondegenerate triangle! Vtx, axis, row, col: %d, %d, %d, %d\n", i, axis, row, col);
							for (int d = 0; d < 3; d++) {
								vtx_indices[offset3 + d] = i;
								axis_indices[offset3 + d] = 0;
								//baryweights[offset3 + d] = 0.0; initialized to 0.
							}
						}
					}
				}

				////compensate masked grid points energy
				//const double comp_scale = double(gm*gn) / double(gm*gn - masked_ptnum);
				//for (int row = 0; row < gm; row++)
				//{
				//	for (int col = 0; col < gn; col++)
				//	{
				//		const int offset3 = (((i*anum + axis)*gm + row)*gn + col) * 3;
				//		for (int d = 0; d < 3; d++)
				//			baryweights[offset3 + d] *= comp_scale;
				//	}
				//}
			}
		}
	}

	std::cout << "Masked grid point num: " << total_masked_ptnum << " out of " << (vnum*anum*gm*gn) << std::endl;
}



void TangentParamVis::build_grid_samples(double scale, int gm, int gn) {
	gm_ = gm; gn_ = gn;
	//tangent_param(mesh, axes, scale, gm, gn, indices, axis_indices, baryweights);
	//std::vector<double> output_features;
	tangent_param_with_patchinput(mesh, axes, scale, gm, gn, true, indices, axis_indices, baryweights, output_features);
	//tangent_param_with_patchinput_LQP(mesh, axes, scale, gm, gn, true, indices, axis_indices, baryweights, output_features);
}

void TangentParamVis::build_grid_samples_from_pc(double scale, int gm, int gn) {
	gm_ = gm; gn_ = gn;
	//tangent_param(mesh, axes, scale, gm, gn, indices, axis_indices, baryweights);
	//std::vector<double> output_features;
	tangent_param_with_patchinput_pointcloud(pointcloud, pc_normal, axes, scale, gm, gn, true, indices, axis_indices, baryweights, output_features);
	//tangent_param_with_patchinput_LQP(mesh, axes, scale, gm, gn, true, indices, axis_indices, baryweights, output_features);
}
