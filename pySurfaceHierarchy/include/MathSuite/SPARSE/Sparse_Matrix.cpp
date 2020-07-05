#include "Sparse_Matrix.h"
#include <ostream>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <set>

std::ostream & operator<<(std::ostream & output_stream, const Sparse_Matrix & mat)
{
	const std::vector< Sparse_Vector* >& entryset = mat.get_entryset();
	output_stream << mat.rows() <<" " << mat.cols() << " " << mat.get_nonzeros() << std::endl;
	if (mat.get_storage_type() == CCS)
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				output_stream << miter->index << ' ' << i << " " << std::scientific << miter->value << std::endl;
			}
		}
	}
	else
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				output_stream << i << ' ' << miter->index << " " << std::scientific << miter->value << std::endl;
			}
		}
	}
	return output_stream;
}

std::ostream & operator<<(std::ostream & output_stream, const Sparse_Matrix * mat)
{
	output_stream << *mat;
	return output_stream;
}
//////////////////////////////////////////////////////////////////////////
void multiply(const Sparse_Matrix *A, double *X, double *Y)
{
	memset(Y, 0, sizeof(double)*A->rows());
	const std::vector< Sparse_Vector* >& entryset = A->get_entryset();
	if (A->get_storage_type() == CCS)
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
			{
				Y[miter->index] += miter->value * X[i];
			}
		}
		if (A->issym_store_upper_or_lower())
		{
			for (size_t i = 0; i < entryset.size(); i++)
			{
				for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
				{
					if (miter->index != i)
						Y[i] += miter->value * X[miter->index];
				}
			}
		}
	}
	else
	{
#pragma omp parallel for schedule(dynamic)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)entryset.size(); i++)
		{
			for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
			{
				Y[i] += miter->value * X[miter->index];
			}
		}
		if (A->issym_store_upper_or_lower())
		{
			for (size_t i = 0; i < entryset.size(); i++)
			{
				for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
				{
					if (miter->index != i)
						Y[miter->index] += miter->value * X[i];
				}
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////
void transpose_multiply(const Sparse_Matrix *A, double *X, double *Y)
{
	memset(Y, 0, sizeof(double)*A->cols());
	const std::vector< Sparse_Vector* >& entryset = A->get_entryset();
	if (A->get_storage_type() != CCS)
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
			{
				Y[miter->index] += miter->value * X[i];
			}
		}
		if (A->issym_store_upper_or_lower())
		{
			for (size_t i = 0; i < entryset.size(); i++)
			{
				for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
				{
					if (miter->index != i)
						Y[i] += miter->value * X[miter->index];
				}
			}
		}
	}
	else
	{
#pragma omp parallel for schedule(dynamic)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)entryset.size(); i++)
		{
			for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
			{
				Y[i] += miter->value * X[miter->index];
			}
		}
		if (A->issym_store_upper_or_lower())
		{
			for (size_t i = 0; i < entryset.size(); i++)
			{
				for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
				{
					if (miter->index != i)
						Y[miter->index] += miter->value * X[i];
				}
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////
void transpose_self_multiply(const Sparse_Matrix *A, double *X, double *Y, double *tmp_vec)
{
	if (tmp_vec)
	{
		multiply(A, X, tmp_vec);
		transpose_multiply(A, tmp_vec, Y);
	}
	else
	{
		std::vector<double> tmp(A->rows());
		multiply(A, X, &tmp[0]);
		transpose_multiply(A, &tmp[0], Y);
	}
}
//////////////////////////////////////////////////////////////////////////
void self_transpose_multiply(const Sparse_Matrix *A, double *X, double *Y, double *tmp_vec)
{
	if (tmp_vec)
	{
		transpose_multiply(A, X, tmp_vec);
	}
	else
	{
		std::vector<double> tmp(A->cols());
		transpose_multiply(A, X, &tmp[0]);
		multiply(A, &tmp[0], Y);
	}
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* convert_sparse_matrix_storage(const Sparse_Matrix *A, SPARSE_STORAGE_TYPE store, SPARSE_SYMMETRIC_TYPE sym)
{
	if (A->get_storage_type() == store && A->get_symmetric_type() == sym)
	{
		Sparse_Matrix* new_mat = new Sparse_Matrix(A);
		return new_mat;
	}

	const std::vector< Sparse_Vector* >& entryset = A->get_entryset();

	Sparse_Matrix * new_mat = new Sparse_Matrix(A->rows(), A->cols(), A->issymmetric()?sym:NOSYM, store, A->get_num_rhs());
	bool both_sym_to_half_sym = A->get_symmetric_type()==SYM_BOTH && (sym == SYM_LOWER || sym == SYM_UPPER);
	bool half_sym_to_nosym = (A->get_symmetric_type() == SYM_LOWER || A->get_symmetric_type() == SYM_UPPER) && sym == NOSYM;

	if (A->get_storage_type() == CCS)
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				if (!(both_sym_to_half_sym && miter->index > i))
					new_mat->set_entry(miter->index, i, miter->value);
				if(half_sym_to_nosym && i != miter->index)
					new_mat->set_entry(i, miter->index, miter->value);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				if (!(both_sym_to_half_sym && miter->index > i))
					new_mat->set_entry(i, miter->index, miter->value);
				if(half_sym_to_nosym && i != miter->index)
					new_mat->set_entry(miter->index, i, miter->value);

			}
		}
	}

	std::copy(A->get_rhs().begin(), A->get_rhs().end(), new_mat->get_rhs().begin());

	return new_mat;
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* transpose(Sparse_Matrix* A, SPARSE_STORAGE_TYPE store, SPARSE_SYMMETRIC_TYPE sym)
{
	const std::vector< Sparse_Vector* >& entryset = A->get_entryset();
	Sparse_Matrix * new_mat = new Sparse_Matrix(A->cols(), A->rows(), A->issymmetric()?sym:NOSYM, store);
	bool both_sym_to_half_sym = A->get_symmetric_type()==SYM_BOTH && (sym == SYM_LOWER || sym == SYM_UPPER);

	if (A->get_symmetric_type() == sym && ((A->get_storage_type() == CCS && store != CCS) || (A->get_storage_type() != CCS && store == CCS) ) )
	{
		std::vector< Sparse_Vector* >& new_entryset = new_mat->get_entryset();
		for (size_t i = 0; i < new_entryset.size(); i++)
		{
			new_entryset[i]->assign(entryset[i]->begin(), entryset[i]->end());
		}
		return new_mat;
	}

	if (A->get_storage_type() == CCS)
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				if (!(both_sym_to_half_sym && miter->index > i))
					new_mat->set_entry(i, miter->index, miter->value);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				if (!(both_sym_to_half_sym && miter->index > i))
					new_mat->set_entry(miter->index, i, miter->value);
			}
		}
	}

	return new_mat;
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* TransposeTimesSelf(Sparse_Matrix* mat, SPARSE_STORAGE_TYPE store, SPARSE_SYMMETRIC_TYPE sym,  bool apply_to_rhs)
{
	Sparse_Matrix* new_mat = CCS_TransposeTimesSelf(mat, sym, apply_to_rhs);
	if (store != CCS)
	{
		Sparse_Matrix* convert_mat = convert_sparse_matrix_storage(new_mat, store, sym);
		delete new_mat;
		new_mat = convert_mat;
	}
	return new_mat;
	//Sparse_Matrix* cur_mat = mat;
	//if (mat->get_storage_type() == CCS || mat->issym_store_upper_or_lower())
	//{
	//	cur_mat = convert_sparse_matrix_storage(mat, CRS, mat->issym_store_upper_or_lower()?SYM_BOTH:NOSYM);
	//}
	//const std::vector< Sparse_Vector* >& entryset = cur_mat->get_entryset();
	//Sparse_Matrix* new_mat = new Sparse_Matrix(mat->cols(), mat->cols(), sym, store, mat->get_num_rhs());

	//if (new_mat->issym_store_upper_or_lower())
	//{
	//	for (size_t i = 0; i < entryset.size(); i++)
	//	{
	//		for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
	//			miter != entryset[i]->end(); miter++)
	//		{
	//			for (Sparse_Vector::const_iterator siter = miter;
	//				siter != entryset[i]->end(); siter++)
	//			{
	//				new_mat->fill_entry(miter->index, siter->index, miter->value*siter->value);
	//			}
	//		}
	//	}
	//}
	//else
	//{
	//	for (size_t i = 0; i < entryset.size(); i++)
	//	{
	//		for (Sparse_Vector::const_iterator miter = entryset[i]->begin();
	//			miter != entryset[i]->end(); miter++)
	//		{
	//			for (Sparse_Vector::const_iterator siter = entryset[i]->begin();
	//				siter != entryset[i]->end(); siter++)
	//			{
	//				if (sym != SYM_BOTH)
	//					new_mat->fill_entry(miter->index, siter->index, miter->value*siter->value);
	//				else
	//				{
	//					if (miter->index >= siter->index)
	//					{
	//						new_mat->fill_entry(miter->index, siter->index, miter->value*siter->value);
	//					}
	//				}
	//			}
	//		}
	//	}
	//}

	//if (apply_to_rhs)
	//{
	//	std::vector<double>& mat_rhs = mat->get_rhs();
	//	std::vector<double>& new_mat_rhs = new_mat->get_rhs();
	//	for (size_t i = 0; i < mat->get_num_rhs(); i++)
	//	{
	//		transpose_multiply(mat, &mat_rhs[i*mat->rows()], &new_mat_rhs[i*mat->cols()]);
	//	}
	//}

	//if (cur_mat != mat)
	//	delete cur_mat;
	//return new_mat;

}
//////////////////////////////////////////////////////////////////////////
void remove_zero_element(Sparse_Vector& vec, double tiny_value)
{
	double zero = std::max(0.0, tiny_value);
	for (ptrdiff_t i = (ptrdiff_t)vec.size() - 1; i >= 0; i--)
	{
		if (std::fabs(vec[i].value) <= zero)
		{
			vec.erase(vec.begin() + i);
		}
	}
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* CCS_TransposeTimesSelf(Sparse_Matrix* mat, SPARSE_SYMMETRIC_TYPE sym, bool apply_to_rhs)
{
	Sparse_Matrix *cur_left = transpose(mat, CCS, mat->issym_store_upper_or_lower()?SYM_BOTH:NOSYM), *cur_right = mat;
	if (mat->get_storage_type() != CCS || mat->issym_store_upper_or_lower())
	{
		cur_right = convert_sparse_matrix_storage(mat, CCS, mat->issym_store_upper_or_lower()?SYM_BOTH:NOSYM);
	}

	const std::vector< Sparse_Vector* >& left_entryset = cur_left->get_entryset();
	const std::vector< Sparse_Vector* >& right_entryset = cur_right->get_entryset();

	ptrdiff_t M = cur_left->rows();// N = cur_left->cols();

	Sparse_Matrix* new_mat = new Sparse_Matrix(cur_left->rows(), cur_left->rows(), sym, CCS, mat->get_num_rhs());
	std::vector< Sparse_Vector* >& new_entryset = new_mat->get_entryset();

	int max_threads = omp_get_max_threads();

	std::vector<double> V(max_threads*cur_left->rows(), 0);

	std::vector<Sparse_Vector> SV(max_threads); 
	for (int i = 0; i < max_threads; i++)
	{
		SV[i].reserve(cur_left->rows());
	}
#pragma omp parallel for schedule(dynamic)
	for (ptrdiff_t i = 0; i < M; i++)
	{
		int thread = omp_get_thread_num();
		ptrdiff_t offset = thread * cur_left->rows();
		std::set<size_t> tag;
		for (Sparse_Vector::const_iterator iter = right_entryset[i]->begin(); iter != right_entryset[i]->end(); iter++)
		{
			Sparse_Vector::const_iterator start = left_entryset[iter->index]->begin();
			Sparse_Vector::const_iterator end = left_entryset[iter->index]->end();
			if (sym == SYM_LOWER)
			{
				start = std::lower_bound(left_entryset[iter->index]->begin(), left_entryset[iter->index]->end(), Sparse_Entry(i));
			}
			else if (sym == SYM_UPPER)
			{
				end = std::upper_bound(left_entryset[iter->index]->begin(), left_entryset[iter->index]->end(), Sparse_Entry(i));
			}
			for (Sparse_Vector::const_iterator iter2 = start; iter2 != end; iter2++)
			{
				V[offset+iter2->index] += iter->value * iter2->value;
				tag.insert(offset+iter2->index);
			}
		}
		SV[thread].resize(0);

		for (std::set<size_t>::const_iterator siter = tag.begin(); siter != tag.end(); siter++)
		{
			SV[thread].push_back(Sparse_Entry(*siter-offset, V[*siter]));
			V[*siter] = 0; 

		}
		new_entryset[i]->assign(SV[thread].begin(), SV[thread].end());
	}

	delete cur_left;
	if (cur_right != mat) delete cur_right;

	if (apply_to_rhs)
	{
		std::vector<double>& mat_rhs = mat->get_rhs();
		std::vector<double>& new_mat_rhs = new_mat->get_rhs();
#pragma omp parallel for schedule(dynamic)
		for (ptrdiff_t i = 0; i < (ptrdiff_t)mat->get_num_rhs(); i++)
		{
			transpose_multiply(mat, &mat_rhs[i*mat->rows()], &new_mat_rhs[i*mat->cols()]);
		}
	}

	return new_mat;

}
//////////////////////////////////////////////////////////////////////////
#ifdef HAVE_CIMG
#include "CImg.h"

void save_sparse_pattern(const char filename[], const Sparse_Matrix* mat, unsigned int resolution)
{
	using namespace cimg_library;

	int width = 1, height = 1;

	if (mat->cols() < mat->rows())
	{
		height = resolution;
		width = height * mat->cols() / mat->rows();
	}
	else
	{
		width = resolution;
		height = width *  mat->rows() / mat->cols();
	}

	float scale_x = 1, scale_y = 1;
	//if ((size_t)width > mat->cols() || (size_t)height > mat->rows())
	{
		scale_x = (float)width / mat->cols();
		scale_y = (float)height / mat->rows();
	}

	CImg<unsigned char> image(width, height, 1, 3, 255);
	const std::vector< Sparse_Vector* >& entryset = mat->get_entryset();
	bool HAVE_sym = mat->issym_store_upper_or_lower();
	bool storage = mat->get_storage_type();
	unsigned char black[3] = {0, 0, 0};
	if (storage != CCS)
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for(Sparse_Vector::const_iterator iter =entryset[i]->begin(); iter != entryset[i]->end(); iter++)
			{
				image.draw_rectangle((int)(scale_y*iter->index), (int)(scale_x*i), (int)(scale_y*(iter->index+1)), (int)(scale_x*(i+1)), black);
				if (HAVE_sym)
				{
					image.draw_rectangle((int)(scale_x*i), (int)(scale_y*iter->index), (int)(scale_x*(i+1)), (int)(scale_y*(iter->index+1)), black);
				}
			}
		}
	}
	else
	{
		for (size_t i = 0; i < entryset.size(); i++)
		{
			for(Sparse_Vector::const_iterator iter = entryset[i]->begin(); iter != entryset[i]->end(); iter++)
			{
				image.draw_rectangle((int)(scale_x*i), (int)(scale_y*iter->index), (int)(scale_x*(i+1)), (int)(scale_y*(iter->index+1)), black);
				if (HAVE_sym)
				{
					image.draw_rectangle((int)(scale_y*iter->index), (int)(scale_x*i), (int)(scale_y*(iter->index+1)), (int)(scale_x*(i+1)), black);
				}
			}
		}
	}
	image.save_bmp(filename);
}
#endif


double DotProduct(const Sparse_Vector& v0, const Sparse_Vector& v1)
{
	Sparse_Vector::const_iterator iter0 = v0.begin(), iter1 = v1.begin();
	double val = 0;
	//bool isnotempty = false;
	while(iter0 != v0.end() && iter1 != v1.end())
	{
		if (iter0->index < iter1->index)
		{
			iter0++;
		}
		else if (iter0->index > iter1->index)
		{
			iter1++;
		}
		else
		{
			val += iter0->value * iter1->value;
			iter0++; iter1++;
		}
	}
	return val;
}