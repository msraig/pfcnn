#pragma once

#include "Mesh3D\Mesh3D.h"
#include <vector>
#include "Geex\graphics\opengl.h"

template<typename T>
struct COOEntry {
	size_t index[2] = { 0 };
	T val_ = 0;
	COOEntry() {}
	COOEntry(int r, int c, T v) {
		index[0] = r; index[1] = c; val_ = v;
	}
	size_t& row() { return index[0]; }
	size_t& col() { return index[1]; }
	T& val() { return val_; }
};

// Generates the sparse and dense tensors for computing polynomial convolution 
//  using degree-4(6 point) Gaussian quadrature.
// Input: 
//	mesh - surface mesh
//	axes - axis for each vertex
//	use_patch_height - whether to use localized height as patch-wise input signal
//	ref_scale - scale the distances so that the trainable polynomial coefficients only need to vary in a small range, e.g. average edge length
// Output:
//	S_fv - a sparse matrix of shape (F*3*A*6,V*A), for each face quadrature point giving the barycentric weights for interpolating the vertex signals
//	S_vf - a sparse matrix of shape (V*A, F*3*A), for each vertex giving the summation coefficients of convolved quantities of its 1-ring faces
//	D_fw - a dense matrix of shape (F*3*A*6, 10), for each face and quadrature point giving the cubic polynomial terms of the conv kernel
//	D_patchinput - a dense matrix of shape (F*3*A*6, C_in), similar to S_fv*F_v, giving the patch-wise localized input signals
void polynomial_conv(Geex::Mesh3D* mesh, const std::vector<std::vector<Geex::vec3>>& axes, bool use_patch_height,//inputs
	std::vector<COOEntry<float>>& S_fv, std::vector<COOEntry<float>>& S_vf,  //outputs
	std::vector<std::vector<float>>& D_fw, std::vector<std::vector<float>>& D_patchinput,
	std::vector<int> &S_fv_index, std::vector<float> &S_fv_value, std::vector<int> &S_vf_index, std::vector<float> &S_vf_value);


// Demo interface 
class PolynomialCovVis {
public:
	Geex::Mesh3D* mesh = NULL;
	std::vector<std::vector<Geex::vec3>> axes;
	int marked_pt = 0;
private:
	std::vector<COOEntry<float>> S_fv;
	std::vector<COOEntry<float>> S_vf;
	std::vector<std::vector<float>> D_fw;
	std::vector<std::vector<float>> D_patchinput;

	std::vector<int> S_fv_index;
	std::vector<float> S_fv_value;

	std::vector<int> S_vf_index;
	std::vector<float> S_vf_value;

public:
	void draw(int axis);
	void build_polynomialconv(double ref_scale);
	void write_input(std::string output_path)
	{
		std::ofstream fout(output_path);
		for (int i = 0; i < D_patchinput.size(); i++)
		{
			for (int j = 0; j < D_patchinput[0].size(); j++)
			{
				fout << D_patchinput[i][j] << " ";
			}
			fout << std::endl;
		}
		fout.close();
	}

	void write_para(std::string output_path)
	{
		std::ofstream fout(output_path, std::ios::binary);
		const int face_num = mesh->get_num_of_faces();
		const int v_num = mesh->get_num_of_vertices();
		const int axis_num = axes[0].size();
		const int fvaq_num = face_num * 3 * axis_num * 6;
		const int S_vf_len = S_vf.size();



		fout.write((const char*)&v_num, sizeof(int));
		fout.write((const char*)&face_num, sizeof(int));
		fout.write((const char*)&axis_num, sizeof(int));
		fout.write((const char*)&S_vf_len, sizeof(int));


		fout.write((const char*)&S_fv_index[0], S_fv_index.size() * sizeof(int));
		fout.write((const char*)&S_fv_value[0], S_fv_value.size() * sizeof(float));

		fout.write((const char*)&S_vf_index[0], S_vf_index.size() * sizeof(int));
		fout.write((const char*)&S_vf_value[0], S_vf_value.size() * sizeof(float));



		for (int i = 0; i < D_fw.size(); i++)
		{
			fout.write((const char*)&D_fw[i][0], D_fw[i].size() * sizeof(float));
		}
		/*
		std::cout << v_num << face_num << axis_num << std::endl;
		std::cout << S_fv_index.size() << std::endl;
		std::cout << S_fv_value.size() << std::endl;
		std::cout << S_vf_index.size() << std::endl;
		std::cout << S_vf_value.size() << std::endl;

		std::cout << D_fw.size() << std::endl;
		std::cout << D_fw[0].size() << std::endl;

		std::cout << D_patchinput.size() << std::endl;
		std::cout << D_patchinput[0].size() << std::endl;
		*/
		for (int i = 0; i < D_patchinput.size(); i++)
		{
			fout.write((const char*)&D_patchinput[i][0], D_patchinput[i].size() * sizeof(float));
		}

		fout.close();
	}
};
