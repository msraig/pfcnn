#pragma once

/*********************************************
 This file contains different ways of direction/frame fields computation, 
 for mesh surfaces and point clouds.
 For mesh surface, the data structure is OpenMesh.
 We only compute 4-direction/frame fields.
*********************************************/

#include "TriMeshDef.hpp"
#include "Eigen/Sparse"
#include "Wm5Matrix3.h"

namespace surface_hierarchy {
	class FieldsOptions {
	public:
		enum FieldComputation {
			RAW_PCA, RAW_CURVATURE_TENSOR, // raw principal directions
			MIXED_INTEGER, COSINE_FUNCTION, // angle based methods
			POLYVECTOR, EXTRINSIC_VECTOR // vector based methods
		} field_computation_method = POLYVECTOR;
		bool smooth_only = false;
		double lambda = 0.01;
		int sym_N=4; // N-symmetry direction field, default is 4 (cross field).
	};

	class FieldsOnMesh {
	public:
		// mesh must have been garbage collected, so no deleted vertices.
		// mesh must be triangular.
		FieldsOnMesh(std::shared_ptr<TriMesh> mesh, const FieldsOptions& option,
			OpenMesh::VPropHandleT<std::vector<TriMesh::Point>>* omp_frames) {
			pmesh_ = mesh;
			option_ = option;
			frame_vph_ = omp_frames;
		}

		void build();
	protected:
		void build_raw_curvature_tensor();
		void build_polyvector();

		void compute_curvature_tensor();
		void build_complex_based_laplacian_mat(std::vector<Eigen::Triplet<double>>& mat_entries);

	private:
		std::shared_ptr<TriMesh> pmesh_;
		FieldsOptions option_;
		OpenMesh::VPropHandleT<std::vector<TriMesh::Point>>* frame_vph_;


		struct CurvatureTensor {
			double kmax, kmin;
			TriMesh::Point pdmax, pdmin;
			CurvatureTensor(){}
			CurvatureTensor(double kmax, double kmin, TriMesh::Point pdmax, 
				TriMesh::Point pdmin) {
				this->kmax = kmax;
				this->kmin = kmin;
				this->pdmax = pdmax,
					this->pdmin = pdmin;
			}
		};

		OpenMesh::VPropHandleT<CurvatureTensor> omp_crv_tensor_;

		// functions for converting between vector/complex number
		// use normal + one edge projected in tangent as reference frames.

		std::complex<double> complex_from_vec(TriMesh::VertexHandle& vh, TriMesh& mesh, TriMesh::Point vec)
		{
			TriMesh::Point xaxis, yaxis;
			get_ref_frame(mesh.normal(vh), xaxis, yaxis);
			std::complex<double> cn(
				OpenMesh::dot(vec, xaxis),
				OpenMesh::dot(vec, yaxis)
			);
			cn /= std::sqrt(std::norm(cn));
			return cn;
		};

		TriMesh::Point vec_from_complex(TriMesh::VertexHandle& vh, TriMesh& mesh, std::complex<double> cn)
		{
			TriMesh::Point xaxis, yaxis;
			get_ref_frame(mesh.normal(vh), xaxis, yaxis);
			return cn.real()*xaxis + cn.imag()*yaxis;
		};

		inline void get_ref_frame(const TriMesh::Point& normal, TriMesh::Point& xaxis, TriMesh::Point& yaxis) {
			
			if (std::abs(normal[0])<1e-7)
			{
				xaxis[0] = 1.0; xaxis[1] = 0.0; xaxis[2] = 0.0;
			}
			else
			{
				xaxis[0] = -normal[0]; xaxis[1] = 0.0; xaxis[2] = 0.0;
			}
			xaxis = xaxis - dot(xaxis, normal)*normal;
			xaxis.normalize();
			yaxis = OpenMesh::cross(normal, xaxis);
			
			/*auto nvh = *mesh.vv_begin(vh);
			auto edir = (mesh.point(nvh) - mesh.point(vh)).normalize();
			xaxis = (edir - OpenMesh::dot(edir,normal)*normal).normalize();
			yaxis = OpenMesh::cross(normal, xaxis);*/
		}
	};

	// General frame field computation on surfaces represented by (Vertices,Normals,Adjacency,InitialFrames)
	class FieldsOnSurface {

		struct CurvatureTensor {
			double kmax, kmin;
			TriMesh::Point pdmax, pdmin;
			CurvatureTensor() {}
			CurvatureTensor(double kmax, double kmin, TriMesh::Point pdmax,
				TriMesh::Point pdmin) {
				this->kmax = kmax;
				this->kmin = kmin;
				this->pdmax = pdmax,
					this->pdmin = pdmin;
			}
		};

	public:

		FieldsOnSurface(int vnum) {
			V.resize(vnum);
			N.resize(vnum);
			Adj.resize(vnum);
			FX.resize(vnum);
		}

		std::vector<TriMesh::Point> V;
		std::vector<TriMesh::Point> N;
		std::vector<std::vector<int>> Adj;
		std::vector<TriMesh::Point> FX; //X axis of initial frame, result x axis is also stored here

		void solve(const FieldsOptions& option);

	private:

		FieldsOptions m_options;

		void compute_curvature_tensor(std::vector<CurvatureTensor>& crv_tensors);
		inline Wm5::Vector3d PointToVec(const TriMesh::Point& p) {
			return Wm5::Vector3d(p[0], p[1], p[2]);
		}
		inline TriMesh::Point VecToPoint(const Wm5::Vector3d& v) {
			return TriMesh::Point(v[0], v[1], v[2]);
		}
		inline void complement_axis(const TriMesh::Point& normal, TriMesh::Point& xaxis, TriMesh::Point& yaxis) {
			if (std::abs(normal[0]) >= std::abs(normal[1]))
			{
				// W.x or W.z is the largest magnitude component, swap them
				xaxis[0] = -normal[2];
				xaxis[1] = 0;
				xaxis[2] = normal[0];
			}
			else
			{
				// W.y or W.z is the largest magnitude component, swap them
				xaxis[0] = 0;
				xaxis[1] = normal[2];
				xaxis[2] = -normal[1];
			}
			xaxis = xaxis - OpenMesh::dot(xaxis, normal)*normal;
			xaxis.normalize();
			yaxis = OpenMesh::cross(normal, xaxis);
		}
		inline std::complex<double> complex_from_vec(TriMesh::Point vec, TriMesh::Point tu, TriMesh::Point tv)
		{
			std::complex<double> cn(
				OpenMesh::dot(vec, tu),
				OpenMesh::dot(vec, tv)
			);
			cn /= std::sqrt(std::norm(cn));
			return cn;
		};

		inline TriMesh::Point vec_from_complex(std::complex<double> cn, TriMesh::Point tu, TriMesh::Point tv)
		{
			return cn.real()*tu + cn.imag()*tv;
		};

		void build_complex_based_laplacian_mat(std::vector<Eigen::Triplet<double>>& mat_entries,
			const std::vector<TriMesh::Point>& RefU, const std::vector<TriMesh::Point>& RefV);

		void solve_polyvector(const std::vector<CurvatureTensor>& crv_tensors, double lambda);

		void solve_polyvector_smoothonly(const std::vector<CurvatureTensor>& crv_tensors);

	};
}
