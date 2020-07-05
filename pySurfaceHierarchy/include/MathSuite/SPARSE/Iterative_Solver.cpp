#include "Iterative_Solver.h"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <cassert>

#ifdef HAVE_AMD
#pragma comment (lib, "amd.lib")
extern "C"
{
	void amd_order(int, int*, int*, int*, double*, double*);
}
#endif
#ifdef HAVE_METIS
#pragma comment (lib, "libmetis.lib")
extern "C"
{
	void METIS_NodeND(int*, int*, int*, int*, int*, int*, int*);
}
#endif

double DDOT(const size_t n, const double *x, const double *y)
{
	double result = 0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:result)
#endif
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
	{
		result += x[i] * y[i];
	}
	return result;
}

void DAXPY(const size_t n, const double alpha, const double *x,
	double *y)
{
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
	{
		y[i] += alpha * x[i];
	}
}

double DNRM2(const size_t n, const double *x)
{
	double result = 0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:result)
#endif
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
	{
		result += x[i] * x[i];
	}
	return std::sqrt(result);
}

void DSCAL(const size_t n, const double a, double *x)
{
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
	for (ptrdiff_t i = 0; i < (ptrdiff_t)n; i++)
	{
		x[i] *= a;
	}
}

////////////////////////////////////////////////////////////////////////////
bool solve_by_CG(Sparse_Matrix *m_sparse_matrix, int max_iter_num, int *num_iter, double tolerance)
{
	if (m_sparse_matrix->rows() < m_sparse_matrix->cols())
	{
		return false;
	}

	bool isequare = true;
	std::vector<double> multi_tmp;

	if (m_sparse_matrix->rows() > m_sparse_matrix->cols())
	{
		isequare = false;
		multi_tmp.resize(m_sparse_matrix->rows());
	}

	std::vector<double> &sol = m_sparse_matrix->get_solution();

	std::vector<double> r(m_sparse_matrix->cols()), rprev(m_sparse_matrix->cols()), diffr(m_sparse_matrix->cols());

	Sparse_Matrix *transpose = 0;
	Sparse_Matrix *self = m_sparse_matrix->get_storage_type() == CRS ?
	m_sparse_matrix : convert_sparse_matrix_storage(m_sparse_matrix, CRS, m_sparse_matrix->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);

	if (isequare)
	{
		//multiply(m_sparse_matrix, &sol[0], &r[0]);
		multiply(self, &sol[0], &r[0]);
	}
	else
	{
		transpose = (m_sparse_matrix->get_storage_type() == CCS && m_sparse_matrix->issym_store_upper_or_lower() == false) ?
		m_sparse_matrix : convert_sparse_matrix_storage(m_sparse_matrix, CCS, m_sparse_matrix->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);
		multiply(self, &sol[0], &multi_tmp[0]);
		transpose_multiply(transpose, &multi_tmp[0], &r[0]);
		//transpose_self_multiply(m_sparse_matrix, &sol[0], &r[0], &multi_tmp[0]);
	}

	std::vector<double> B;

	if (!isequare)
	{
		B.resize(m_sparse_matrix->cols());
		transpose_multiply(transpose, &m_sparse_matrix->get_rhs()[0], &B[0]);
		//transpose_multiply(m_sparse_matrix, &m_sparse_matrix->get_rhs()[0], &B[0]);
	}

	std::vector<double> &b = isequare ? m_sparse_matrix->get_rhs() : B;

	double normr = 0;

#pragma omp parallel for
	for (ptrdiff_t i = 0; i < (ptrdiff_t)r.size(); i++)
	{
		r[i] = b[i] - r[i];
	}

	double normb = DNRM2(b.size(), &b[0]);

	if (normb == 0)
	{
		normb = 1;
	}

	int iter = 0;
	size_t dim = m_sparse_matrix->cols();

	normr = DNRM2(dim, &r[0]);

	if ((normr / normb) < tolerance)
	{
		return true;
	}

	double one = 1;
	std::vector<double> p(r), Ap(dim);

	while (iter < max_iter_num)
	{
		memcpy(&rprev[0], &r[0], sizeof(double)*r.size());

		//Ap = A*p
		if (isequare)
		{
			//multiply(m_sparse_matrix, &p[0], &Ap[0]);
			multiply(self, &p[0], &Ap[0]);
		}
		else
		{
			multiply(self, &p[0], &multi_tmp[0]);
			transpose_multiply(transpose, &multi_tmp[0], &Ap[0]);
			//transpose_self_multiply(m_sparse_matrix, &p[0], &Ap[0], &multi_tmp[0]);
		}

		// rdot = dot(r, r)
		double rdot = DDOT(dim, &r[0], &r[0]);
		//alpha = dot(r, r) / dot(p, Ap);
		double alpha = rdot / DDOT(dim, &p[0], &Ap[0]);
		// sol = sol + alpha * p
		DAXPY(dim, alpha, &p[0], &sol[0]);
		//r = r - alpha * Ap;
		alpha = -alpha;
		DAXPY(dim, alpha, &Ap[0], &r[0]);

		normr = DNRM2(dim, &r[0]);

		if ((normr / normb) < tolerance)
		{
			break;
		}

		//beta = dot(r, r) / rdot
		//double beta = DDOT(dim, &r[0], &r[0]) / rdot;
#pragma omp parallel for
		for (ptrdiff_t i = 0; i < (ptrdiff_t)r.size(); i++)
		{
			diffr[i] = r[i] - rprev[i];
		}
		double beta = std::max(0.0, DDOT(dim, &r[0], &diffr[0]) / rdot);
		//double beta = DDOT(dim, &r[0], &diffr[0]) / rdot;
		// p = r + beta * p
		DSCAL(dim, beta, &p[0]);
		DAXPY(dim, one, &r[0], &p[0]);
		iter++;
	}

	if (num_iter)
	{
		*num_iter = iter;
	}

	if (self && self != m_sparse_matrix)
	{
		delete self;
	}
	if (transpose && transpose != m_sparse_matrix)
	{
		delete transpose;
	}

	return (normr / normb) < tolerance;
}
//////////////////////////////////////////////////////////////////////////
bool solve_by_CG_with_jacobian_pred(Sparse_Matrix *m_sparse_matrix, int max_iter_num, int *num_iter, double tolerance)
{
	if (m_sparse_matrix->rows() < m_sparse_matrix->cols())
	{
		return false;
	}
	else if (m_sparse_matrix->rows() > m_sparse_matrix->cols())
	{
		return solve_by_CG(m_sparse_matrix, max_iter_num, num_iter, tolerance);
	}

	Sparse_Matrix *self = m_sparse_matrix->get_storage_type() == CRS ?
	m_sparse_matrix : convert_sparse_matrix_storage(m_sparse_matrix, CRS, m_sparse_matrix->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);

	std::vector<double> diag(m_sparse_matrix->rows());

#pragma omp parallel for
	for (ptrdiff_t i = 0; i < (ptrdiff_t)diag.size(); i++)
	{
		diag[i] = 1.0 / m_sparse_matrix->get_entry(i, i);
	}

	std::vector<double> &sol = m_sparse_matrix->get_solution();
	std::vector<double> r(m_sparse_matrix->cols()), z(m_sparse_matrix->cols()), rprev(m_sparse_matrix->cols()), diffr(m_sparse_matrix->cols());

	multiply(self, &sol[0], &r[0]);   // r = Ax
	//multiply(m_sparse_matrix, &sol[0], &r[0]);   // r = Ax

	std::vector<double> &b = m_sparse_matrix->get_rhs();

	double normr = 0;

#pragma omp parallel for
	for (ptrdiff_t i = 0; i < (ptrdiff_t)r.size(); i++)// r = b - Ax
	{
		r[i] = b[i] - r[i];
		z[i] = diag[i] * r[i];
	}

	double normb = DNRM2(b.size(), &b[0]);

	if (normb == 0)
	{
		normb = 1;
	}

	int iter = 0;
	size_t dim = m_sparse_matrix->rows();
	double one = 1;

	normr = DNRM2(dim, &r[0]);

	if ((normr / normb) < tolerance)
	{
		return true;
	}

	std::vector<double> p(z), Ap(dim);

	while (iter < max_iter_num)
	{
		memcpy(&rprev[0], &r[0], sizeof(double)*r.size());

		//Ap = A*p
		multiply(self, &p[0], &Ap[0]);
		//multiply(m_sparse_matrix, &p[0], &Ap[0]);

		double rzdot = DDOT(dim, &r[0], &z[0]);
		//alpha = dot(r, z) / dot(p, Ap);
		double alpha = rzdot / DDOT(dim, &p[0], &Ap[0]);
		// sol = sol + alpha * p
		DAXPY(dim, alpha, &p[0], &sol[0]);
		//r = r - alpha * Ap;
		alpha = -alpha;
		DAXPY(dim, alpha, &Ap[0], &r[0]);

		normr = DNRM2(dim, &r[0]);

		if ((normr / normb) < tolerance)
		{
			break;
		}
#pragma omp parallel for
		for (ptrdiff_t i = 0; i < (ptrdiff_t)r.size(); i++)
		{
			z[i] = diag[i] * r[i];
		}

		//beta = dot(r, z) / rzdot
		//double beta = DDOT(dim, &r[0], &z[0]) / rzdot;
#pragma omp parallel for
		for (ptrdiff_t i = 0; i < (ptrdiff_t)r.size(); i++)
		{
			diffr[i] = r[i] - rprev[i];
		}
		double beta = std::max(0.0, DDOT(dim, &z[0], &diffr[0]) / rzdot);
		//double beta = DDOT(dim, &z[0], &diffr[0]) / rzdot;

		// p = r + beta * p
		DSCAL(dim, beta, &p[0]);
		DAXPY(dim, one, &z[0], &p[0]);
		iter++;
	}

	if (num_iter)
	{
		*num_iter = iter;
	}

	if (self && self != m_sparse_matrix)
	{
		delete self;
	}

	return 	(normr / normb) < tolerance;
}//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* incomplete_Cholesky_factorization(Sparse_Matrix *m_sparse_matrix)
{
	if (m_sparse_matrix->rows() != m_sparse_matrix->cols())
		return 0;
	Sparse_Matrix* A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, SYM_LOWER);
	A->remove_sym_info();
	std::vector< Sparse_Vector* >& entryset = A->get_entryset();

	size_t N = m_sparse_matrix->rows();
	std::vector<double> V(N);

	for (size_t k = 0; k < N; k++)
	{
		double& val = entryset[k]->front().value;
		val = std::sqrt(std::fabs(val));

		Sparse_Vector::iterator second = entryset[k]->begin(); second++;
		if (second != entryset[k]->end())
		{
			for (Sparse_Vector::iterator miter = second; miter != entryset[k]->end(); miter++)
			{
				miter->value /= val;
				V[miter->index] = miter->value;
			}
			for (Sparse_Vector::iterator miter = second; miter != entryset[k]->end(); miter++)
			{
				const size_t& j = miter->index;
				const double& jk = miter->value;
				for (Sparse_Vector::iterator miter2 = entryset[j]->begin(); miter2 != entryset[j]->end(); miter2++)
				{
					miter2->value -= V[miter2->index] * jk;
				}
			}
			for (Sparse_Vector::iterator miter = second; miter != entryset[k]->end(); miter++)
			{
				V[miter->index] = 0;
			}
		}
	}

	return A;
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* incomplete_Cholesky_factorization_by_filling(Sparse_Matrix *m_sparse_matrix, int fill_in_p)
{
	if (m_sparse_matrix->rows() != m_sparse_matrix->cols())
		return 0;
	Sparse_Matrix* A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, SYM_LOWER);
	A->remove_sym_info();
	std::vector< Sparse_Vector* >& entryset = A->get_entryset();

	size_t N = m_sparse_matrix->rows();
	std::vector<double> V(N);
	std::vector<size_t> used_idvec;
	used_idvec.reserve(N);
	std::vector<Sparse_Entry> new_col;
	new_col.reserve(m_sparse_matrix->rows());

	for (size_t k = 0; k < N; k++)
	{
		double& val = entryset[k]->front().value;
		val = std::max(1.0e-8, std::sqrt(std::fabs(val)));

		Sparse_Vector::iterator second = entryset[k]->begin(); second++;
		if (second != entryset[k]->end())
		{
			for (Sparse_Vector::iterator miter = second; miter != entryset[k]->end(); miter++)
			{
				miter->value /= val;
			}
			for (Sparse_Vector::iterator miter = second; miter != entryset[k]->end(); miter++)
			{
				const size_t& j = miter->index;
				const double& jk = miter->value;

				used_idvec.resize(0);
				for (Sparse_Vector::iterator miter3 = entryset[j]->begin(); miter3 != entryset[j]->end(); miter3++)
				{
					V[miter3->index] = miter3->value;
					used_idvec.push_back(miter3->index);
				}
				Sparse_Vector::iterator miter2 = miter; miter2++;
				if (miter2 != entryset[k]->end())
				{
					for (; miter2 != entryset[k]->end(); miter2++)
					{
						V[miter2->index] -= miter2->value * jk;
						used_idvec.push_back(miter2->index);
					}
				}
				//update j column
				new_col.resize(0);
				std::vector<size_t>::iterator it = std::unique(used_idvec.begin(), used_idvec.end());
				for (std::vector<size_t>::iterator iter = used_idvec.begin(); iter != it; iter++)
				{
					if (*iter != j)
					{
						new_col.push_back(Sparse_Entry(*iter, V[*iter]));
					}
					V[*iter] = 0;
				}

				size_t len = std::min(entryset[j]->size() + fill_in_p - 1, new_col.size());

				std::partial_sort(new_col.begin(), new_col.begin() + len, new_col.end(), first_ordering_sparse_vector);

				std::sort(new_col.begin(), new_col.begin() + len, second_ordering_sparse_vector);

				entryset[j]->resize(len + 1);
				Sparse_Vector::iterator start = entryset[j]->begin(); start++;
				if (len > 0)
				{
					std::copy(new_col.begin(), new_col.begin() + len, start);
				}
			}
		}
	}
	return A;
}
//////////////////////////////////////////////////////////////////////////
void back_substitution_with_incomplete_Cholesky(Sparse_Matrix* chol, double* R, double *X)
{
	std::vector< Sparse_Vector* >& entryset = chol->get_entryset();
	size_t N = chol->rows();

	memcpy(X, R, sizeof(double)*N);

	for (size_t k = 0; k < N; k++)
	{
		Sparse_Vector::iterator miter = entryset[k]->begin();
		X[k] /= miter->value;
		miter++;
		for (; miter != entryset[k]->end(); miter++)
		{
			X[miter->index] -= X[k] * miter->value;
		}
	}
	for (ptrdiff_t k = N - 1; k >= 0; k--)
	{
		Sparse_Vector::iterator miter = entryset[k]->begin();
		miter++;
		for (; miter != entryset[k]->end(); miter++)
		{
			X[k] -= X[miter->index] * miter->value;
		}
		X[k] /= entryset[k]->front().value;
	}
}
//////////////////////////////////////////////////////////////////////////
bool solve_by_CG_with_incomplete_Cholesky_pred(Sparse_Matrix *m_sparse_matrix, Sparse_Matrix* chol, int max_iter_num, int *num_iter, double tolerance)
{
	if (m_sparse_matrix->rows() < m_sparse_matrix->cols())
		return false;

	bool isequare = true;
	std::vector<double> multi_tmp;
	if (m_sparse_matrix->rows() > m_sparse_matrix->cols())
	{
		isequare = false;
		multi_tmp.resize(m_sparse_matrix->rows());
	}

	std::vector<double>& sol = m_sparse_matrix->get_solution();
	std::vector<double> r(m_sparse_matrix->rows()), z(m_sparse_matrix->rows()), rprev(m_sparse_matrix->rows()), diffr(m_sparse_matrix->rows());

	if (isequare)
		multiply(m_sparse_matrix, &sol[0], &r[0]);  // r = Ax
	else
		transpose_self_multiply(m_sparse_matrix, &sol[0], &r[0], &multi_tmp[0]);

	std::vector<double> B;
	if (!isequare)
	{
		B.resize(m_sparse_matrix->cols());
		transpose_multiply(m_sparse_matrix, &m_sparse_matrix->get_rhs()[0], &B[0]);
	}
	std::vector<double>& b = isequare ? m_sparse_matrix->get_rhs() : B;

	double normr = 0;
	for (size_t i = 0; i < r.size(); i++) // r = b - Ax
	{
		r[i] = b[i] - r[i];
	}

	double normb = DNRM2(b.size(), &b[0]);
	if (normb == 0)
	{
		normb = 1;
	}
	normr = DNRM2(r.size(), &r[0]);

	if ((normr / normb) < tolerance)
		return true;

	back_substitution_with_incomplete_Cholesky(chol, &r[0], &z[0]);

	int iter = 0;
	size_t dim = m_sparse_matrix->rows();
	double one = 1;

	std::vector<double> p(z), Ap(dim);
	while (iter < max_iter_num)
	{
		memcpy(&rprev[0], &r[0], sizeof(double)*r.size());
		//Ap = A*p
		if (isequare)
			multiply(m_sparse_matrix, &p[0], &Ap[0]);
		else
			transpose_self_multiply(m_sparse_matrix, &p[0], &Ap[0], &multi_tmp[0]);

		double rzdot = DDOT(dim, &r[0], &z[0]);
		//alpha = dot(r, z) / dot(p, Ap);
		double alpha = rzdot / DDOT(dim, &p[0], &Ap[0]);
		// sol = sol + alpha * p
		DAXPY(dim, alpha, &p[0], &sol[0]);
		//r = r - alpha * Ap;
		alpha = -alpha;
		DAXPY(dim, alpha, &Ap[0], &r[0]);

		normr = DNRM2(dim, &r[0]);
		if ((normr / normb) < tolerance)
		{
			break;
		}
		back_substitution_with_incomplete_Cholesky(chol, &r[0], &z[0]);
		//beta = dot(r, z) / rzdot
		//double beta = DDOT(dim, &r[0], &z[0])/rzdot;
		for (size_t i = 0; i < r.size(); i++)
		{
			diffr[i] = r[i] - rprev[i];
		}
		//double beta = std::max(0.0, DDOT(dim, &z[0], &diffr[0]) / rzdot);
		double beta = DDOT(dim, &z[0], &diffr[0]) / rzdot;
		// p = r + beta * p
		DSCAL(dim, beta, &p[0]);
		DAXPY(dim, one, &z[0], &p[0]);
		iter++;
	}
	if (num_iter)
	{
		*num_iter = iter;
	}
	return (normr / normb) < tolerance;
}
//////////////////////////////////////////////////////////////////////////
bool solve_by_CG_with_incomplete_Cholesky_pred(Sparse_Matrix *m_sparse_matrix, int max_iter_num, int *num_iter, double tolerance)
{
	if (m_sparse_matrix->rows() != m_sparse_matrix->cols())
		return false;
	Sparse_Matrix* chol = incomplete_Cholesky_factorization(m_sparse_matrix);
	bool suc = solve_by_CG_with_incomplete_Cholesky_pred(m_sparse_matrix, chol, max_iter_num, num_iter, tolerance);
	delete chol;
	return suc;
}
//////////////////////////////////////////////////////////////////////////
bool solve_by_CG_with_incomplete_Cholesky_pred_ordering(Sparse_Matrix *m_sparse_matrix, int max_iter_num, int *num_iter, double tolerance)
{
	if (m_sparse_matrix->rows() != m_sparse_matrix->cols())
		return false;
	std::vector<size_t> iperm;
	//Sparse_Matrix* ordered_mat = cmkOrdering(m_sparse_matrix, iperm);
	//Sparse_Matrix* ordered_mat = amdOrdering(m_sparse_matrix, iperm);
	Sparse_Matrix* ordered_mat = metisOrdering(m_sparse_matrix, iperm);
	//Sparse_Matrix* chol = incomplete_Cholesky_factorization_by_filling(ordered_mat, 0);

	Sparse_Matrix* chol = incomplete_Cholesky_factorization(ordered_mat);
	bool suc = solve_by_CG_with_incomplete_Cholesky_pred(ordered_mat, chol, max_iter_num, num_iter, tolerance);
	for (size_t i = 0; i < m_sparse_matrix->rows(); i++)
	{
		m_sparse_matrix->get_solution()[i] = ordered_mat->get_solution()[iperm[i]];
	}
	if (ordered_mat != m_sparse_matrix)
		delete ordered_mat;
	delete chol;
	return suc;
}
//////////////////////////////////////////////////////////////////////////
int solve_by_BiCGSTAB(Sparse_Matrix *m_sparse_matrix, int max_iter_num, int *num_iter, double tolerance)
{
	if (m_sparse_matrix->rows() != m_sparse_matrix->cols())
		return -1;
	double resid;
	double rho_1 = 1, rho_2 = 1, alpha = 1, beta = 1, omega = 1;
	const size_t N = m_sparse_matrix->rows();
	std::vector<double> p(N), phat(N), s(N), shat(N), t(N), v(N), r(N), rtilde(N);
	std::vector<double>& sol = m_sparse_matrix->get_solution();
	std::vector<double>& b = m_sparse_matrix->get_rhs();

	double normb = DNRM2(N, &b[0]);
	multiply(m_sparse_matrix, &sol[0], &r[0]);
	for (size_t i = 0; i < N; i++)
	{
		r[i] = b[i] - r[i];
	}
	memcpy(&rtilde[0], &r[0], sizeof(double)*N);

	if (normb == 0)
	{
		normb = 1;
	}

	double normr = DNRM2(N, &r[0]);

	if ((resid = normr / normb) <= tolerance)
	{
		return 0;
	}

	for (int iter = 0; iter < max_iter_num; iter++)
	{
		rho_1 = DDOT(N, &r[0], &rtilde[0]);
		if (std::fabs(rho_1) <= tolerance)
		{
			if (num_iter)
			{
				*num_iter = iter;
			}
			return 1;
		}
		if (iter == 1)
		{
			memcpy(&p[0], &r[0], sizeof(double)*N);
		}
		else
		{
			beta = (rho_1 / rho_2) * (alpha / omega);
			for (size_t i = 0; i < N; i++)
			{
				p[i] = r[i] + beta * (p[i] - omega * v[i]);
			}
		}

		// M.solve
		memcpy(&phat[0], &p[0], sizeof(double)*N);
		multiply(m_sparse_matrix, &phat[0], &v[0]);
		alpha = rho_1 / DDOT(N, &rtilde[0], &v[0]);
		for (size_t i = 0; i < N; i++)
		{
			s[i] = r[i] - alpha * v[i];
		}
		if ((resid = DNRM2(N, &s[0]) / normb) < tolerance)
		{
			for (size_t i = 0; i < N; i++)
			{
				sol[i] += alpha * phat[i];
			}

			if (num_iter)
			{
				*num_iter = iter;
			}
			return 2;
		}

		//M.solve
		memcpy(&shat[0], &s[0], sizeof(double)*N);
		multiply(m_sparse_matrix, &shat[0], &t[0]);
		omega = DDOT(N, &t[0], &s[0]) / DDOT(N, &t[0], &t[0]);
		for (size_t i = 0; i < N; i++)
		{
			sol[i] += alpha * phat[i] + omega * shat[i];
		}
		for (size_t i = 0; i < N; i++)
		{
			r[i] = s[i] - omega * t[i];
		}
		normr = DNRM2(N, &r[0]);
		rho_2 = rho_1;
		if ((resid = normr / normb) < tolerance)
		{
			if (num_iter)
			{
				*num_iter = iter;
			}
			return 3;
		}

		if (std::fabs(omega) < tolerance)
		{
			return 4;
		}
	}

	return 5;
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* cmkOrdering(Sparse_Matrix *m_sparse_matrix, std::vector<size_t>& iperm)
{
	Sparse_Matrix* mat = m_sparse_matrix;
	if (m_sparse_matrix->issym_store_upper_or_lower())
	{
		mat = convert_sparse_matrix_storage(mat, CCS, NOSYM);
	}
	std::vector< Sparse_Vector* >& entryset = mat->get_entryset();
	std::vector< bool> tag(mat->rows(), false);

	std::vector< ordering_tuple > adj_set, total_set;
	total_set.reserve(mat->rows());
	for (size_t i = 0; i < mat->rows(); i++)
	{
		total_set.push_back(ordering_tuple(entryset[i]->size(), i, m_sparse_matrix->get_entry(i, i)));
	}
	std::sort(total_set.begin(), total_set.end(), ordering_tuple_compare);
	std::vector< ordering_tuple >::iterator first = total_set.begin();
	std::vector<size_t> perm;
	perm.reserve(mat->rows());
	perm.push_back(total_set.front().index);
	tag[total_set.front().index] = true;

	adj_set.reserve(total_set.back().degree);

	for (size_t i = 0; i < perm.size(); i++)
	{
		adj_set.resize(0);
		for (Sparse_Vector::iterator miter = entryset[perm[i]]->begin(); miter != entryset[perm[i]]->end(); miter++)
		{
			if (!tag[miter->index])
			{
				adj_set.push_back(ordering_tuple(entryset[miter->index]->size(), miter->index, miter->value));
				tag[miter->index] = true;
			}
		}
		std::sort(adj_set.begin(), adj_set.end(), ordering_tuple_compare);
		for (std::vector< ordering_tuple >::iterator iter = adj_set.begin(); iter != adj_set.end(); iter++)
		{
			perm.push_back(iter->index);
		}
		if (adj_set.size() == 0)
		{
			if (perm.size() != mat->rows())
			{
				for (; first != total_set.end(); first++)
				{
					if (!tag[first->index])
					{
						perm.push_back(first->index);
						tag[first->index] = true;
						break;
					}
				}
			}
			else
				break;
		}
	}

	std::reverse(perm.begin(), perm.end());

	iperm.resize(perm.size());
	for (size_t i = 0; i < perm.size(); i++)
	{
		iperm[perm[i]] = i;
	}

	Sparse_Matrix* sorted_matrix = new Sparse_Matrix(perm.size(), perm.size(), SYM_LOWER, CCS);

	for (size_t i = 0; i < entryset.size(); i++)
	{
		for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
		{
			if (miter->index >= i)
			{
				sorted_matrix->set_entry(iperm[i], iperm[miter->index], miter->value);
			}
		}
		sorted_matrix->set_rhs_entry(iperm[i], m_sparse_matrix->get_rhs()[i]);
	}

	if (mat != m_sparse_matrix)
	{
		delete mat;
	}

	return sorted_matrix;
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* amdOrdering(Sparse_Matrix *m_sparse_matrix, std::vector<size_t>& iperm)
{
#ifndef HAVE_AMD
	iperm;
	return m_sparse_matrix;
#else
	if (!m_sparse_matrix->issquare())
		return m_sparse_matrix;
	iperm.resize(m_sparse_matrix->rows());
	std::vector<int> rowind, colptr, perm(iperm.size());
	std::vector<double> values;

	m_sparse_matrix->get_compress_data(rowind, colptr, values);
	int nv = (int)m_sparse_matrix->rows();
	amd_order(nv, &colptr[0], &rowind[0], &perm[0], 0, 0);
	iperm.assign(perm.begin(), perm.end());
	Sparse_Matrix* sorted_matrix = new Sparse_Matrix(iperm.size(), iperm.size(), SYM_LOWER, CCS);
	std::vector< Sparse_Vector* >& entryset = m_sparse_matrix->get_entryset();
	for (size_t i = 0; i < entryset.size(); i++)
	{
		for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
		{
			if (miter->index >= i)
			{
				sorted_matrix->set_entry(iperm[i], iperm[miter->index], miter->value);
			}
		}
		sorted_matrix->set_rhs_entry(iperm[i], m_sparse_matrix->get_rhs()[i]);
	}
	return sorted_matrix;
#endif
}
//////////////////////////////////////////////////////////////////////////
Sparse_Matrix* metisOrdering(Sparse_Matrix *m_sparse_matrix, std::vector<size_t>& iperm)
{
#ifndef HAVE_METIS
	iperm;
	return m_sparse_matrix;
#else
	if (!m_sparse_matrix->issquare())
		return m_sparse_matrix;
	Sparse_Matrix *mat = m_sparse_matrix;
	if (m_sparse_matrix->get_storage_type() != CCS || m_sparse_matrix->issym_store_upper_or_lower())
	{
		mat = convert_sparse_matrix_storage(m_sparse_matrix, CCS, SYM_BOTH);
	}

	iperm.resize(mat->rows());
	std::vector<int> rowind, colptr, perm(iperm.size()), inv_perm(iperm.size());
	std::vector< Sparse_Vector* >& entryset = mat->get_entryset();
	colptr .reserve(entryset.size()+1);
	int counter =0;
	colptr.push_back(counter);
	rowind.reserve(m_sparse_matrix->get_nonzeros());
	for (size_t i = 0; i < entryset.size(); i++)
	{
		for (Sparse_Vector::iterator iter = entryset[i]->begin(); iter != entryset[i]->end(); iter++)
		{
			if (iter->index != i)
			{
				counter++;
				rowind.push_back((int)iter->index);
			}
		}
		colptr.push_back(counter);
	}

	int nv = (int)m_sparse_matrix->rows(); int flag = 0; int option[8] = {0,0,0,0,0,0,0,0};
	METIS_NodeND(&nv, &colptr[0], &rowind[0], &flag, option, &perm[0], &inv_perm[0]);
	iperm.assign(perm.begin(), perm.end());
	Sparse_Matrix* sorted_matrix = new Sparse_Matrix(iperm.size(), iperm.size(), SYM_LOWER, CCS);
	std::vector< Sparse_Vector* >& entryset2 = m_sparse_matrix->get_entryset();
	for (size_t i = 0; i < entryset.size(); i++)
	{
		for (Sparse_Vector::iterator miter = entryset2[i]->begin(); miter != entryset2[i]->end(); miter++)
		{
			if (miter->index >= i)
			{
				sorted_matrix->set_entry(iperm[i], iperm[miter->index], miter->value);
			}
		}
		sorted_matrix->set_rhs_entry(iperm[i], m_sparse_matrix->get_rhs()[i]);
	}
	if (mat != m_sparse_matrix)
		delete mat;
	return sorted_matrix;
#endif
}
//////////////////////////////////////////////////////////////////////////
bool ordering_tuple_compare(const ordering_tuple& p1, const ordering_tuple& p2)
{
	if (p1.degree < p2.degree)
		return true;
	else if (p1.degree == p2.degree)
	{
		return std::fabs(p1.value) > std::fabs(p2.value);
	}
	return false;
}
//////////////////////////////////////////////////////////////////////////
bool first_ordering_sparse_vector(const Sparse_Entry& p1, const Sparse_Entry& p2)
{
	return std::fabs(p1.value) > std::fabs(p2.value);
}
//////////////////////////////////////////////////////////////////////////
bool second_ordering_sparse_vector(const Sparse_Entry& p1, const Sparse_Entry& p2)
{
	return p1.index < p2.index;
}

//////////////////////////////////////////////////////////////////////////
double solve_by_BBNNLS(Sparse_Matrix *m_sparse_matrix, std::vector<double> &lower_bound, int max_iter_num, int *num_iter)
{
	if (m_sparse_matrix == 0) return -1;

	ptrdiff_t nrow = m_sparse_matrix->rows(), ncol = m_sparse_matrix->cols();
	std::vector<double> out_x(m_sparse_matrix->get_solution());
	std::vector<double> out_grad(ncol), out_oldg(ncol), Ax(nrow), Ag(nrow), tmpAg(ncol), tmp(nrow);
	double step, numer;

	int iter = 0;
	int max_iter = max_iter_num;
	//double tiny_pos_value = 1.0e-12;
	double tolg = 1.0e-6;

	//init
	//compute f and g
	multiply(m_sparse_matrix, &out_x[0], &Ax[0]);
	if (lower_bound.size() == (size_t)ncol)
	{
		multiply(m_sparse_matrix, &lower_bound[0], &tmp[0]);
		DAXPY(nrow, 1, &tmp[0], &Ax[0]);
	}
	DAXPY(nrow, -1, &m_sparse_matrix->get_rhs()[0], &Ax[0]);

	//out_obj = 0.5 *DDOT(ncol, &Ax[0], &Ax[0]);
	transpose_multiply(m_sparse_matrix, &Ax[0], &out_grad[0]);
	memcpy(&out_oldg[0], &out_grad[0], sizeof(double)*ncol);
	//out_oldg.assign(out_grad.begin(), out_grad.end());

	while (1)
	{
		iter++;
		//initialization & checkTermination

		double pg = 0;
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+:pg)
#endif
		for (ptrdiff_t i = 0; i < ncol; i++)
		{
			if (out_x[i] != 0 || out_grad[i] < 0)
			{
				pg += out_grad[i] * out_grad[i];
			}
		}
		pg = std::sqrt(pg);
		if (pg < tolg)
			break;

		if (iter > max_iter)
			break;

		//ComputeBBStep
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
		for (ptrdiff_t i = 0; i < ncol; i++)
		{
			if (out_x[i] == 0 && out_grad[i] > 0)
				out_oldg[i] = 0;
		}
		multiply(m_sparse_matrix, &out_oldg[0], &Ag[0]);

		if (iter % 2 == 0)
		{
			step = DDOT(ncol, &out_oldg[0], &out_oldg[0]) / DDOT(nrow, &Ag[0], &Ag[0]);
		}
		else
		{
			numer = DDOT(nrow, &Ag[0], &Ag[0]);
			transpose_multiply(m_sparse_matrix, &Ag[0], &tmpAg[0]);
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
			for (ptrdiff_t i = 0; i < ncol; i++)
			{
				if (fabs(out_x[i]) < 1.0e-12 && out_grad[i] > 0)
					tmpAg[i] = 0;
			}
			step = numer / DDOT(ncol, &tmpAg[0], &tmpAg[0]);
		}

		DAXPY(ncol, -step, &out_grad[0], &out_x[0]);
		memcpy(&out_oldg[0], &out_grad[0], sizeof(double)*ncol);
		//out_oldg.assign(out_grad.begin(), out_grad.end());

		//project
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
		for (ptrdiff_t i = 0; i < ncol; i++)
		{
			if (out_x[i] < 0)
				out_x[i] = 0;
		}

		//compute f and g
		multiply(m_sparse_matrix, &out_x[0], &Ax[0]);
		if (lower_bound.size() == (size_t)ncol)
		{
			DAXPY(nrow, 1, &tmp[0], &Ax[0]);
		}
		DAXPY(nrow, -1, &m_sparse_matrix->get_rhs()[0], &Ax[0]);
		//out_obj = 0.5 *DDOT(ncol, &Ax[0], &Ax[0]);
		transpose_multiply(m_sparse_matrix, &Ax[0], &out_grad[0]);
	}

	if (lower_bound.size() == (size_t)ncol)
	{
		DAXPY(ncol, 1, &lower_bound[0], &out_x[0]);
	}

	m_sparse_matrix->get_solution().assign(out_x.begin(), out_x.end());

	if (num_iter) *num_iter = iter;

	return 0.5 *DDOT(ncol, &Ax[0], &Ax[0]);
}