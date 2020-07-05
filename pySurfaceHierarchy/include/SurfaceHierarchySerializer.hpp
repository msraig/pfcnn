#pragma once

///////////////////////////////////////////////////////////
/* This file defines the format of serialization/deserialization 
   of surface hierarchy data, together with training/label data, 
   and batched database setup.
   IMPORTANT: This file is shared by both the training/evaluation 
   data generator, and caffe extension for surface CNNs. Keep the 
   different copies up-to-date.
   Note: the same thing could have been done through XML/protobuf.
 */

#include <vector>

namespace surface_hierarchy {

#define DATA_TO_BYTE_ARRAY(var, des) \
	std::copy(static_cast<const char*>(static_cast<const void*>(&var)), \
		static_cast<const char*>(static_cast<const void*>(&var)) + sizeof var, \
		des); des += sizeof(var)
#define BYTE_ARRAY_TO_DATA(src, var, T) \
	var = *(T*) src; src += sizeof(var)

	/////////////////////////////////////////////////////
	// Surface structure serialization.
	template <typename T>
	class SurfaceSerializer {
	public:
		std::vector<T> pts;
		std::vector<T> frames;
		std::vector<T> weights;
		std::vector<std::vector<int>> pts_covered;
		std::vector<std::vector<int>> nb_vts;
		int frame_sym_N=4; // frame field symmetry size, default is 4 (cross field).

		int datasize() {
			int byte_num =
				sizeof(int) // frame_sym_N
				+ sizeof(int) // pts size
				+ pts.size() * sizeof(T) // pts data
				+ sizeof(int) // frames size
				+ frames.size() * sizeof(T) // frames data
				+ sizeof(int) // weights size
				+ weights.size() * sizeof(T); // weights data
			
			byte_num += sizeof(int);
			for (size_t i = 0; i < pts_covered.size(); i++)
			{
				byte_num += sizeof(int)*(pts_covered[i].size() + 1);
			}
			byte_num += sizeof(int);
			for (size_t i = 0; i < nb_vts.size(); i++)
			{
				byte_num += sizeof(int)*(nb_vts[i].size() + 1);
			}
			return byte_num;
		}

		char* serialize(char* des) {
			DATA_TO_BYTE_ARRAY(frame_sym_N, des);

			int pt_num = pts.size();
			DATA_TO_BYTE_ARRAY(pt_num, des); 
			for (size_t i = 0; i < pts.size(); i++)
			{
				DATA_TO_BYTE_ARRAY(pts[i], des); 
			}

			int frame_num = frames.size();
			DATA_TO_BYTE_ARRAY(frame_num, des);	
			for (size_t i = 0; i < frames.size(); i++)
			{
				DATA_TO_BYTE_ARRAY(frames[i], des); 
			}

			int weight_num = weights.size();
			DATA_TO_BYTE_ARRAY(weight_num, des);
			for (size_t i = 0; i < weights.size(); i++)
			{
				DATA_TO_BYTE_ARRAY(weights[i], des);
			}

			int pts_cover_num = pts_covered.size();
			DATA_TO_BYTE_ARRAY(pts_cover_num, des);	
			for (size_t i = 0; i < pts_covered.size(); i++)
			{
				int cover_num = pts_covered[i].size();
				DATA_TO_BYTE_ARRAY(cover_num, des);	
				for (size_t j = 0; j < cover_num; j++)
				{
					DATA_TO_BYTE_ARRAY(pts_covered[i][j], des);
				}
			}

			int patch_num = nb_vts.size();
			DATA_TO_BYTE_ARRAY(patch_num, des);
			for (size_t i = 0; i < nb_vts.size(); i++)
			{
				int nb_vnum = nb_vts[i].size();
				DATA_TO_BYTE_ARRAY(nb_vnum, des);
				for (size_t j = 0; j < nb_vnum; j++)
				{
					DATA_TO_BYTE_ARRAY(nb_vts[i][j], des);
				}
			}
			return des;
		}
		const char* deserialize(const char* src) {
			BYTE_ARRAY_TO_DATA(src, frame_sym_N, int);

			int pt_num;
			BYTE_ARRAY_TO_DATA(src, pt_num, int);
			pts.resize(pt_num);
			for (size_t i = 0; i < pt_num; i++)
			{
				BYTE_ARRAY_TO_DATA(src, pts[i], T);
			}

			int frame_num;
			BYTE_ARRAY_TO_DATA(src, frame_num, int);
			frames.resize(frame_num);
			for (size_t i = 0; i < frame_num; i++)
			{
				BYTE_ARRAY_TO_DATA(src, frames[i], T);
			}

			int weight_num;
			BYTE_ARRAY_TO_DATA(src, weight_num, int);
			weights.resize(weight_num);
			for (size_t i = 0; i < weight_num; i++)
			{
				BYTE_ARRAY_TO_DATA(src, weights[i], T);
			}

			int pts_cover_num;
			BYTE_ARRAY_TO_DATA(src, pts_cover_num, int);
			pts_covered.resize(pts_cover_num);
			for (size_t i = 0; i < pts_cover_num; i++)
			{
				int cover_num;
				BYTE_ARRAY_TO_DATA(src, cover_num, int);
				pts_covered[i].resize(cover_num);
				for (size_t j = 0; j < cover_num; j++)
				{
					BYTE_ARRAY_TO_DATA(src, pts_covered[i][j], int);
				}
			}

			int patch_num;
			BYTE_ARRAY_TO_DATA(src, patch_num, int);
			nb_vts.resize(patch_num);
			for (size_t i = 0; i < patch_num; i++)
			{
				int nb_vnum;
				BYTE_ARRAY_TO_DATA(src, nb_vnum, int);
				nb_vts[i].resize(nb_vnum);
				for (size_t j = 0; j < nb_vnum; j++)
				{
					BYTE_ARRAY_TO_DATA(src, nb_vts[i][j], int);
				}
			}

			return src;
		}
	};

	template <typename T>
	class SurfaceHierarchySerializer {
	public:
		std::vector<SurfaceSerializer<T>> surfaces;
		int datasize() {
			int bytenum = 0;
			for (auto& ss : surfaces)
			{
				bytenum += ss.datasize();
			}
			return bytenum;
		}
		char* serialize(char* des) {
			int surface_num = surfaces.size();
			DATA_TO_BYTE_ARRAY(surface_num, des);
			for (auto& ss : surfaces)
			{
				des = ss.serialize(des);
			}
			return des;
		}
		const char* deserialize(const char* src) {
			int surface_num;
			BYTE_ARRAY_TO_DATA(src, surface_num, int);
			surfaces.resize(surface_num);
			for (auto& ss:surfaces)
			{
				src = ss.deserialize(src);
			}
			return src;
		}
	};

	///////////////////////////////////////////////////
	// Surface data/label serialization, for training/evalutation.
	template <typename T_in, typename T_out>
	class SurfaceDataSerializer {
	public:
		/*int n_point;
		int n_in_channel;
		int n_label;
		int n_out_channel;*/

		// Data, of shape n_in_channel x n_point
		std::vector<std::vector<T_in>> point_data;
		// Label, of shape n_out_channel x n_label
		std::vector<std::vector<T_out>> label;

		int datasize() {
			if (point_data.size()<1)
			{
				std::cerr << "Error: no point data." << std::endl;
				return -1;
			}
			if (label.size()<1)
			{
				std::cerr << "Error: no label data." << std::endl;
				return -1;
			}
			int bytenum = 0;
			bytenum += 2*sizeof(int);
			bytenum += point_data.size()*point_data[0].size() * sizeof(T_in);

			bytenum += 2 * sizeof(int);
			bytenum += label.size()*label[0].size() * sizeof(T_out);
			return bytenum;
		}

		char* serialize(char* des) {
			int channel_in = point_data.size();
			if (channel_in<1)
			{
				std::cerr << "Error: no point data." << std::endl;
				return des;
			}
			int n_point = point_data[0].size();
			DATA_TO_BYTE_ARRAY(channel_in, des);
			DATA_TO_BYTE_ARRAY(n_point, des);
			for (size_t i = 0; i < channel_in; i++)
			{
				for (size_t j = 0; j < n_point; j++) {
					DATA_TO_BYTE_ARRAY(point_data[i][j], des);
				}
			}

			int channel_out = label.size();
			if (channel_out<1)
			{
				std::cerr << "Error: no label data." << std::endl;
				return des;
			}
			int n_label = label[0].size();
			DATA_TO_BYTE_ARRAY(channel_out, des);
			DATA_TO_BYTE_ARRAY(n_label, des);
			for (size_t i = 0; i < channel_out; i++)
			{
				for (size_t j = 0; j < n_label; j++) {
					DATA_TO_BYTE_ARRAY(label[i][j], des);
				}
			}

			return des;
		}

		const char* deserialize(const char* src) {
			int n_point, n_in_channel, n_label, n_out_channel;
			BYTE_ARRAY_TO_DATA(src, n_in_channel, int);
			BYTE_ARRAY_TO_DATA(src, n_point, int);
			point_data.resize(n_in_channel, std::vector<T_in>(n_point));
			for (size_t i = 0; i < n_in_channel; i++)
			{
				for (size_t j = 0; j < n_point; j++) {
					BYTE_ARRAY_TO_DATA(src, point_data[i][j], T_in);
				}
			}

			BYTE_ARRAY_TO_DATA(src, n_out_channel, int);
			BYTE_ARRAY_TO_DATA(src, n_label, int);
			label.resize(n_out_channel, std::vector<T_out>(n_label));
			for (size_t i = 0; i < n_out_channel; i++)
			{
				for (size_t j = 0; j < n_label; j++) {
					BYTE_ARRAY_TO_DATA(src, label[i][j], T_out);
				}
			}

			return src;
		}
	};

	//////////////////////////////////////////////////////////////
	// Surface CNN one sample data serialization.
	template <typename T_s, typename T_in, typename T_out>
	class SurfaceCNNSampleSerializer {
	public:
		SurfaceHierarchySerializer<T_s> shs;
		SurfaceDataSerializer<T_in, T_out> sds;

		int datasize() {
			return shs.datasize() + sds.datasize();
		}
		char* serialize(char* des) {
			des = sds.serialize(des);
			des = shs.serialize(des);
			return des;
		}
		const char* deserialize(const char* src) {
			src = sds.deserialize(src);
			src = shs.deserialize(src);
			return src;
		}
	};

	//////////////////////////////////////////////////////
	// Surface CNN batch sample data serialization.
	template <typename T_s, typename T_in, typename T_out>
	class SurfaceCNNBatchSerializer {
	public:
		std::vector<SurfaceCNNSampleSerializer<T_s, T_in, T_out>> samples;

		int datasize() {
			int bytenum = sizeof(int);
			for (auto& ss : samples)
			{
				bytenum += ss.datasize();
			}
			return bytenum;
		}
		char* serialize(char* des) {
			int n_sample = samples.size();
			DATA_TO_BYTE_ARRAY(n_sample, des);
			for (auto& ss : samples)
			{
				des = ss.serialize(des);
			}
			return des;
		}
		const char* deserialize(const char* src) {
			int n_sample;
			BYTE_ARRAY_TO_DATA(src, n_sample, int);
			samples.resize(n_sample);
			for (auto& ss:samples)
			{
				src = ss.deserialize(src);
			}
			return src;
		}
	};

	template class SurfaceSerializer<double>;
	template class SurfaceSerializer<float>;
	template class SurfaceHierarchySerializer<double>;
	template class SurfaceHierarchySerializer<float>;
	template class SurfaceDataSerializer<double, int>;
	template class SurfaceDataSerializer<float, int>;
	template class SurfaceDataSerializer<double, double>;
	template class SurfaceCNNSampleSerializer<double,double, int>;
	template class SurfaceCNNSampleSerializer<float, float, int>;
	template class SurfaceCNNSampleSerializer<double,double, double>;
	template class SurfaceCNNBatchSerializer<double, double, int>;
	template class SurfaceCNNBatchSerializer<double, double, double>;
}