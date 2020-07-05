#include "QEM_Decimator.hpp"
//#include "boost\heap\binomial_heap.hpp"
#include <queue>
#include <unordered_map>

namespace surface_hierarchy {

	void QEM_Decimator::decimate(const SurfaceHierarchyOptions& options) {

		// initialize vertex properties

		pmesh_->request_vertex_normals();
		pmesh_->request_face_normals();
		pmesh_->update_normals();

		if(record_vtx_cover_) pmesh_->add_property(*covered_vts_vph_, "CoveredVertices");
		
		pmesh_->add_property(omp_qmat, "QuadricMat");

		for (auto vitr= pmesh_->vertices_begin(); vitr!= pmesh_->vertices_end(); vitr++)
		{
			auto vh = *vitr;
			if (record_vtx_cover_) pmesh_->property(*covered_vts_vph_, vh).push_back(vh.idx());
			
			Mat4d plane_mat(pmesh_->point(vh), pmesh_->normal(vh));
			pmesh_->property(omp_qmat, vh) = plane_mat;
		}

		pmesh_->request_face_status();
		pmesh_->request_edge_status();
		pmesh_->request_vertex_status();

		// initialize edge score

		struct q_edge {
			OpenMesh::EdgeHandle eh;
			double score;
			q_edge(OpenMesh::EdgeHandle handle, double s) {
				eh = handle; score = s;
			}
			q_edge() { score = 0.; }
		};
		auto edge_compare = [](const q_edge& e0, const q_edge& e1) -> bool {
			return e0.score > e1.score;
		};

		// up-to-date scores
		std::unordered_map<OpenMesh::EdgeHandle, double> edge_scores;

		// setup min heap
		std::priority_queue<q_edge, std::vector<q_edge>, decltype(edge_compare)> 
			edge_heap(edge_compare);

		for (auto eitr=pmesh_->edges_begin(); eitr!= pmesh_->edges_end(); eitr++)
		{
			double score = evaluate_edge_error(*eitr, pmesh_);
			edge_scores[*eitr] = score;
			edge_heap.push(q_edge(*eitr, score));
		}

		// loop to collapse edges, until target_vnum

		int remaining_vnum = pmesh_->n_vertices();
		while (!edge_heap.empty() && remaining_vnum > target_vnum_) {
			q_edge se;

			if (!options.randomized_simplfy)
			{
				se = edge_heap.top();
				edge_heap.pop();
			}
			else
			{
				const int topK = options.random_top_k;
				std::vector<q_edge> top_edges;
				for (size_t itr = 0; itr < topK && !edge_heap.empty(); itr++)
				{
					top_edges.push_back(edge_heap.top());
					edge_heap.pop();
				}
				//std::random_shuffle(top_edges.begin(), top_edges.end());
				int selected_idx = rand() % top_edges.size();
				se = top_edges[selected_idx];
				std::swap(top_edges[0], top_edges[selected_idx]);
				for (size_t itr = 1; itr < top_edges.size(); itr++)
				{
					edge_heap.push(top_edges[itr]);
				}
			}

			if (pmesh_->status(se.eh).deleted())
			{
				continue;
			}
			if (se.score < edge_scores[se.eh])
			{
				continue;
			}
			// collapse edge
			auto heh = pmesh_->halfedge_handle(se.eh, 0);
			if (!pmesh_->is_collapse_ok(heh))
			{
				continue;
			}
			auto to_vh = pmesh_->to_vertex_handle(heh),
				from_vh = pmesh_->from_vertex_handle(heh);
			
			// use mid point
			pmesh_->point(to_vh) = 0.5*(pmesh_->point(to_vh) + pmesh_->point(from_vh));
			// use optimal point
			//pmesh_->point(to_vh) = (pmesh_->property(omp_qmat, to_vh) + pmesh_->property(omp_qmat, from_vh)).optimal_pt();

			pmesh_->property(omp_qmat, to_vh) = 
				pmesh_->property(omp_qmat, to_vh) +	pmesh_->property(omp_qmat, from_vh);
			if (record_vtx_cover_)
				pmesh_->property(*covered_vts_vph_, to_vh).insert(
					pmesh_->property(*covered_vts_vph_, to_vh).end(),
					pmesh_->property(*covered_vts_vph_, from_vh).begin(),
					pmesh_->property(*covered_vts_vph_, from_vh).end()
				);

			pmesh_->collapse(heh);

			remaining_vnum--;

			// update edge score
			for (auto ve_itr = pmesh_->cve_begin(to_vh); 
				ve_itr != pmesh_->cve_end(to_vh); ve_itr++)
			{
				auto nb_eh = *ve_itr;
				auto score = evaluate_edge_error(nb_eh, pmesh_);
				edge_scores[nb_eh] = score;
				edge_heap.push(q_edge(nb_eh, score));
			}
		}

		// clear deleted v/e/f
		pmesh_->garbage_collection();
		//// debug
		//for (auto vitr=pmesh_->vertices_begin(); vitr!=pmesh_->vertices_end();
		//	vitr++)
		//{
		//	if (!pmesh_->status(*vitr).deleted())
		//	{
		//		auto& cv = pmesh_->property(*covered_vts_vph_, *vitr);
		//		std::cout << "Vertex " << vitr->idx() << ": ";
		//		for (auto cvitr : cv)
		//		{
		//			std::cout << cvitr << " ";
		//		}
		//		std::cout << std::endl;
		//	}
		//}


		pmesh_->remove_property(omp_qmat);
	}

	double QEM_Decimator::evaluate_edge_error(const TriMesh::EdgeHandle& eh,
		std::shared_ptr<const TriMesh> mesh) {
		auto heh = mesh->halfedge_handle(eh, 0);
		auto vh0 = mesh->to_vertex_handle(heh),
			vh1 = mesh->from_vertex_handle(heh);
		TriMesh::Point mid_pt = (mesh->point(vh0) + mesh->point(vh1))*0.5;
		Mat4d sum_q = mesh->property(omp_qmat, vh0) + mesh->property(omp_qmat, vh1);

		//auto opt_pt = sum_q.optimal_pt();
		//if (opt_pt[0] == 0 && opt_pt[1] == 0 && opt_pt[2] == 0)
		//{
			return sum_q.mult(mid_pt);
		//}
		//else
		//{
			//return sum_q.mult(opt_pt);
		//}
	}

}