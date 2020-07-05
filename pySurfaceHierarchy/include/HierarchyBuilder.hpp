#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "DirectionFields.hpp"

namespace surface_hierarchy {
	class SurfaceHierarchyOptions {
	public:
		// surface representation, default = TRIMESH.
		enum  {TRIMESH, POINTCLOUD} surface_rep = TRIMESH;
		bool not_build_frames = false;
		// hierarchy levels and sizes
		bool is_normalize = true;
		double normalize_radius = -1;//-1 for individual normalize
		bool IsSmooth = true;
		bool Smooth_only = false;
		double poly_lambda = 0.01;

		int top_level_size = 1024; // coarsest level resolution, default = 1024.
		bool force_top_level_size = true; // force top level size to be specified value.
		bool force_level_size_factor = false;
		int level_num = 5; // number of levels, default = 5.
		// lower bound of size multiple factor 
		//  between neighboring levels, default = 1.5.
		double level_size_factor = 1.5; 
		
		bool randomized_simplfy = false; // use randomization when doing simplfication or not.
		int random_top_k = 100; // the number of top candidates to consider when using randomized simplification

		//// local frame computation
		FieldsOptions fields_option;

		//// raw frame estimation method, default = PCA.
		//enum { PCA, GEOMETRIC_TOOLS, INSTANT_FIELD } frame_estimation = PCA;

		//// weight of smoothing the principal directions for frame computation, 
		// // 0.0 means using raw frame estimations, default = 0.0.
		//double frame_smoothing = 0.0; 

		bool get_level_sizes(int vnum, std::vector<int>& level_sizes) {
			// determine the target vertex number for 
			// different levels
			level_sizes.resize(level_num, 0);

			if (vnum < top_level_size)
			{
				std::cerr << "Error: mesh size smaller than coarsest level.\n";
				return false;
			}
			//std::cout << top_level_size << std::endl;
			double finest_level_ptnum = top_level_size * std::pow(level_size_factor, level_num - 1);
			
			if (finest_level_ptnum  >(double)vnum)
			{
				std::cerr << "Error: mesh size not big enough for "\
					"specified number of levels and level size factor.\n";
				return false;
			}

			if (level_num == 1)
			{
				if (force_top_level_size)
				{
					level_sizes[0] = top_level_size;
				}
				else
				{
					level_sizes[0] = vnum;
				}
			}
			else
			{
				if (force_level_size_factor)
				{
					double ratio = level_size_factor;
					level_sizes[0] = top_level_size;
					for (int i = 0; i < level_num; i++)
					{
						level_sizes[i + 1] = level_sizes[i] * ratio;
					}
				}
				else
				{
					double ratio = std::pow((double)vnum / (double)top_level_size,
						1.0 / (level_num - 1));
					level_sizes[0] = top_level_size;
					for (int i = 0; i < level_num - 1; i++)
					{
						level_sizes[i + 1] = level_sizes[i] * ratio;
					}
					level_sizes.back() = vnum;
				}
			}
			std::reverse(level_sizes.begin(), level_sizes.end());
			return true;
		}
	};

	class SurfaceHierarchyBuilder {
	protected:
		SurfaceHierarchyOptions options_;
	public:
		SurfaceHierarchyBuilder(const SurfaceHierarchyOptions& option)
		{
			options_ = option;
		}
		virtual void build(const char* filename) = 0;
		virtual void write(const char* filename) = 0;
	};
}