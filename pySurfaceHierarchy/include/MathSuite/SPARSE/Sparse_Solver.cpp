#include "Sparse_Solver.h"
#include "Iterative_Solver.h"

//////////////////////////////////////////////////////////////////////////
//! TAUCS: Sparse Linear Solver
/*!
*	<A HREF= "http://www.tau.ac.il/~stoledo/taucs/"> TAUCS 2.2 </A>
*/
#ifdef USE_TAUCS
extern "C"
{
#include "taucs.h"
}
#pragma comment (lib, "libf2c.lib") // for fortran to c
#pragma comment (lib, "libmetis.lib") // for metis
#pragma comment (lib, "libtaucs.lib")
#endif

//////////////////////////////////////////////////////////////////////////
//! UMFPACK: Sparse Linear Solver
/*!
*	<A HREF= "http://www.cise.ufl.edu/research/sparse/umfpack/"> UMFPACK 5.0 </A>
*/
#ifdef USE_UMFPACK
#include "..\UMFPACK\umfpack.h"
#pragma comment (lib, "libmetis.lib")
#pragma comment (lib, "SuiteSparseLib.lib")

#endif

//////////////////////////////////////////////////////////////////////////
//! ARPACK: a collection of Fortran77 subroutines designed to solve large scale eigenvalue problems

/*!
*	<A HREF= "http://www.cise.ufl.edu/research/sparse/umfpack/"> ARPACK </A>
*/
#ifdef USE_ARPACK
#pragma comment (lib, "carpack.lib")
#endif

#ifdef USE_CHOLMOD
#include "..\CHOLMOD\cholmod.h"
#pragma comment (lib, "libmetis.lib")
#pragma comment (lib, "SuiteSparseLib.lib")
#endif

#ifdef USE_SUPERLU
#include "slu_ddefs.h"
#pragma comment (lib, "superlu.lib")
#endif

//////////////////////////////////////////////////////////////////////////
#ifdef USE_TAUCS
bool solve_by_TAUCS(Sparse_Matrix *m_sparse_matrix, bool spd, bool cg_solver)
{
	if (m_sparse_matrix == NULL)
	{
		return false;
	}
	bool spd_ = m_sparse_matrix->issymmetric() && spd;
	Sparse_Matrix *A = m_sparse_matrix;

	if (spd_ && m_sparse_matrix->get_symmetric_type() != SYM_LOWER)
	{
		A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, SYM_LOWER);
	}
	else if (spd_ == false && m_sparse_matrix->issym_store_upper_or_lower())
	{
		A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, SYM_BOTH);
	}
	else if (m_sparse_matrix->get_storage_type() != CCS)
		A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, m_sparse_matrix->get_symmetric_type());

	int row = (int)A->rows();
	int col = (int)A->cols();
	std::vector<int> rowind, colptr;
	std::vector<double> values;
	A->get_compress_data(rowind, colptr, values);
	int m_num_column_of_RHS = (int)A->get_num_rhs();
	std::vector<double>& B = A->get_rhs();
	std::vector<double>& solution = m_sparse_matrix->get_solution();

	taucs_ccs_matrix mat;
	mat.colptr = &colptr[0];
	mat.rowind = &rowind[0];
	mat.m = row;
	mat.n = col;
	mat.values.d = &values[0];

	if (spd_) {
		mat.flags = TAUCS_DOUBLE | TAUCS_LOWER | TAUCS_SYMMETRIC;

		char* opt[] = { "taucs.factor.LLT=true", "taucs.factor.mf=true", "taucs.factor.ordering=metis", NULL };
		//char* opt_cg[] = { "taucs.factor.LLT=true", "taucs.factor.ordering=metis",
		//	"taucs.factor.droptol=1e-2",
		//	"taucs.solve.cg=true",
		//	NULL };
		char* opt_cg[] = { "taucs.factor.ldlt=true", "taucs.factor.ordering=metis",
			"taucs.factor.droptol=1e-2",
			"taucs.solve.cg=true",
			NULL };

		int res = taucs_linsolve(&mat, NULL, m_num_column_of_RHS, &solution[0], &B[0], (cg_solver ? opt_cg : opt), NULL);

		if (A != m_sparse_matrix)
		{
			delete A;
		}
		return (res == TAUCS_SUCCESS);
	}
	else
	{
		mat.flags = TAUCS_DOUBLE;
		taucs_io_handle* LU;
		char fname[] = "taucs";
		int* perm = NULL;
		int* invperm = NULL;
		taucs_ccs_order(&mat, &perm, &invperm, "metis");
		LU = taucs_io_create_multifile(fname);

		//danger! if the matrix is singular
		taucs_ooc_factor_lu(&mat, perm, LU, taucs_available_memory_size()); // taucs_available_memory_size()*factor is ok,
		//note in debug mode, the program will stop working forever.
		for (int i = 0; i < m_num_column_of_RHS; i++)
		{
			taucs_ooc_solve_lu(LU, &solution[i*row], &B[i*row]);
		}

		if (perm) {
			free(perm);
		}
		if (invperm) {
			free(invperm);
		}

		taucs_io_delete(LU);

		if (A != m_sparse_matrix)
		{
			delete A;
		}
		return true;
	}
	return true;
}

#endif

#ifdef USE_UMFPACK
bool solve_by_UMFPACK(Sparse_Matrix *m_sparse_matrix)
{
	if (m_sparse_matrix == NULL)
	{
		return false;
	}

	Sparse_Matrix *A = m_sparse_matrix;

	if (m_sparse_matrix->get_storage_type() != CCS || m_sparse_matrix->issym_store_upper_or_lower())
	{
		A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, m_sparse_matrix->issym_store_upper_or_lower() ? SYM_BOTH : m_sparse_matrix->get_symmetric_type());
	}
	int row = (int)A->rows();
	int col = (int)A->cols();
	std::vector<int> rowind, colptr;
	std::vector<double> values;
	A->get_compress_data(rowind, colptr, values);
	int m_num_column_of_RHS = (int)A->get_num_rhs();
	std::vector<double>& B = A->get_rhs();
	std::vector<double>& solution = m_sparse_matrix->get_solution();

	double Info[UMFPACK_INFO];
	double *null = (double *)NULL;
	void *Symbolic, *Numeric;

	(void)umfpack_di_symbolic(row, col, &colptr[0], &rowind[0], &values[0], &Symbolic, null, null);
	int status = umfpack_di_numeric(&colptr[0], &rowind[0], &values[0], Symbolic, &Numeric, null, Info);
	if (status < 0 || (int)Info[UMFPACK_STATUS] == UMFPACK_WARNING_singular_matrix) {
		umfpack_di_free_symbolic(&Symbolic);
		umfpack_di_free_numeric(&Numeric);
		if (A != m_sparse_matrix)
		{
			delete A;
		}
		return false;
	}
	umfpack_di_free_symbolic(&Symbolic);

	for (int i = 0; i < m_num_column_of_RHS; i++)
	{
		(void)umfpack_di_solve(UMFPACK_A, &colptr[0], &rowind[0], &values[0], &solution[i*row], &B[i*row], Numeric, null, null);
	}

	umfpack_di_free_numeric(&Numeric);

	if (A != m_sparse_matrix)
	{
		delete A;
	}

	return true;
}

bool inverse_power_method_by_UMFPACK(Sparse_Matrix* m_sparse_matrix, double* target_eigen_vec, int max_iter, double tiny_value, double *min_eigenvalue)
{
	if (NULL == m_sparse_matrix)
	{
		return false;
	}

	if (m_sparse_matrix->issymmetric())
	{
		return false;
	}

	Sparse_Matrix *A = m_sparse_matrix;
	if (m_sparse_matrix->get_storage_type() != CCS || m_sparse_matrix->issym_store_upper_or_lower())
	{
		A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, m_sparse_matrix->issym_store_upper_or_lower() ? SYM_BOTH : m_sparse_matrix->get_symmetric_type());
	}

	int row = (int)A->rows();
	int col = (int)A->cols();
	std::vector<int> rowind, colptr;
	std::vector<double> values;
	A->get_compress_data(rowind, colptr, values);

	double Info[UMFPACK_INFO];
	double* null = (double*)NULL;
	void* Symbolic;
	void* Numeric;

	(void)umfpack_di_symbolic(row, col, &colptr[0], &rowind[0], &values[0], &Symbolic, null, null);

	int status = umfpack_di_numeric(&colptr[0], &rowind[0], &values[0], Symbolic, &Numeric, null, Info);

	if (status < 0 || UMFPACK_WARNING_singular_matrix == (int)Info[UMFPACK_STATUS])
	{
		umfpack_di_free_symbolic(&Symbolic);
		umfpack_di_free_numeric(&Numeric);
		if (A != m_sparse_matrix)
		{
			delete A;
		}
		return false;
	}

	umfpack_di_free_symbolic(&Symbolic);

	int n_step = 0;
	double* eigen_vec = new double[row];
	double* eigen_vec_next = new double[row];

	//set initial vector
	//you can specify ---  srand((unsigned int)time(0))
	for (int i = 0; i < row; ++i)
	{
		eigen_vec[i] = double(rand()) / RAND_MAX;
	}

	double diff = 0.0;
	static int inc = 1;
	int max_iter_ = max_iter == 0 ? 10 * row : max_iter;
	double len = 0, inv_len = 0;
	while (n_step < max_iter_)
	{
		(void)umfpack_di_solve(UMFPACK_A, &colptr[0], &rowind[0], &values[0], eigen_vec_next, eigen_vec, Numeric, null, null);

		std::swap(eigen_vec_next, eigen_vec);
		//normalize
		len = DNRM2(row, eigen_vec);
		inv_len = 1.0 / len;
		DSCAL(row, inv_len, eigen_vec);

		//check the difference, L-infty
		diff = 0.0;
		for (int i = 0; i < row; ++i)
		{
			diff = std::max<double>(diff, std::fabs(eigen_vec[i] - eigen_vec_next[i]));
		}

		if (diff < tiny_value)
			break;

		++n_step;
	}

	umfpack_di_free_numeric(&Numeric);

	memcpy(target_eigen_vec, eigen_vec, sizeof(double)*row);

	if (min_eigenvalue)
	{
		multiply(A, target_eigen_vec, eigen_vec);
		int sign = 1;
		for (int i = 0; i < row; i++)
		{
			double val = target_eigen_vec[i] * eigen_vec[i];
			if (val != 0)
			{
				sign = val > 0 ? 1 : -1;
				break;
			}
		}
		*min_eigenvalue = sign * DNRM2(row, eigen_vec) / len;
	}

	delete[] eigen_vec;
	delete[] eigen_vec_next;

	if (A != m_sparse_matrix)
		delete A;

	if (n_step == max_iter_ && diff > tiny_value)
		return false;

	return true;
}

#endif

#ifdef USE_CHOLMOD
bool solve_by_CHOLMOD(Sparse_Matrix *m_sparse_matrix)
{
	if (m_sparse_matrix == NULL || m_sparse_matrix->issymmetric() == false)
	{
		return false;
	}

	Sparse_Matrix *A = m_sparse_matrix;
	if (m_sparse_matrix->get_storage_type() != CCS || m_sparse_matrix->get_symmetric_type() != SYM_LOWER)
	{
		A = convert_sparse_matrix_storage(m_sparse_matrix, CCS, SYM_LOWER);
	}

	int row = (int)A->rows();
	int col = (int)A->cols();
	std::vector<int> rowind, colptr;
	std::vector<double> values;
	A->get_compress_data(rowind, colptr, values);
	int m_num_column_of_RHS = (int)A->get_num_rhs();
	std::vector<double>& B = A->get_rhs();
	std::vector<double>& solution = m_sparse_matrix->get_solution();

	cholmod_sparse mat;
	mat.nrow = row;
	mat.ncol = col;
	mat.nzmax = (int)values.size();
	mat.p = &colptr[0];
	mat.i = &rowind[0];
	mat.stype = -1;
	mat.x = &values[0];
	mat.itype = CHOLMOD_INT;
	mat.xtype = CHOLMOD_REAL;
	mat.dtype = CHOLMOD_DOUBLE;
	mat.packed = 0;
	mat.sorted = 1;

	std::vector<int> mnz(col);
	mat.nz = &mnz[0];
	for (int i = 1; i < col + 1; i++) {
		mnz[i - 1] = colptr[i] - colptr[i - 1];
	}

	cholmod_dense  b;
	cholmod_dense* x;

	b.nrow = row;
	b.ncol = m_num_column_of_RHS;
	b.nzmax = b.nrow*b.ncol;
	b.d = row;
	b.x = &B[0];
	b.xtype = CHOLMOD_REAL;
	b.dtype = CHOLMOD_DOUBLE;

	cholmod_common c;
	cholmod_start(&c);
	cholmod_factor* L = cholmod_analyze(&mat, &c);
	cholmod_factorize(&mat, L, &c);
	x = cholmod_solve(CHOLMOD_A, L, &b, &c);

	memcpy(&solution[0], x->x, m_num_column_of_RHS*row*sizeof(double));

	cholmod_free_factor(&L, &c);
	cholmod_free_dense(&x, &c);
	cholmod_finish(&c);

	if (A != m_sparse_matrix)
	{
		delete A;
	}

	return true;
}
#endif

#ifdef USE_ARPACK
extern "C"
{
	void dsaupd_(int *ido, char *bmat, int *n, char *which,
		int *nev, double *tol, double *resid, int *ncv,
		double *v, int *ldv, int *iparam, int *ipntr,
		double *workd, double *workl, int *lworkl,
		int *info);

	void dseupd_(int *rvec, char *All, int *select, double *d,
		double *v, int *ldv, double *sigma,
		char *bmat, int *n, char *which, int *nev,
		double *tol, double *resid, int *ncv, double *v2,
		int *ldv2, int *iparam, int *ipntr, double *workd,
		double *workl, int *lworkl, int *ierr);
}

bool solve_SVD_by_ARPACK(Sparse_Matrix *A, int num_of_eigenvalue, int eigentype,
	double *eigenvalues, double *eigenvectors)
{
	int n = (int)A->cols();
	int nev = num_of_eigenvalue;
	int ido = 0;
	char bmat[2] = "I";
	double tol = 0.0;
	double *resid = new double[n];
	int ncv = 4 * nev;
	if (ncv > n) ncv = n;
	int ldv = n;
	double *v = new double[ldv*ncv];;
	int iparam[11]; /* An array used to pass information to the routines
					about their functional modes. */
	iparam[0] = 1; // Specifies the shift strategy (1->exact)

	iparam[2] = 3 * n; // Maximum number of iterations

	iparam[6] = 1;/* Sets the mode of dsaupd.
				  1 is exact shifting,
				  2 is user-supplied shifts,
				  3 is shift-invert mode,
				  4 is buckling mode,
				  5 is Cayley mode. */

	int ipntr[11]; /* Indicates the locations in the work array workd
				   where the input and output vectors in the
				   callback routine are located. */

	double *workd = new double[3 * n];
	double *workl = new double[ncv*(ncv + 8)];
	int lworkl = ncv*(ncv + 8);
	int info = 0;
	int rvec = 1; // Changed from above
	int *select = new int[ncv];
	double *d = new double[2 * ncv];
	double sigma;
	int ierr;
	double *Z = new double[A->rows()];
	do {
		dsaupd_(&ido, bmat, &n, arpack_type[eigentype], &nev, &tol, resid,
			&ncv, v, &ldv, iparam, ipntr, workd, workl,
			&lworkl, &info);

		if ((ido == 1) || (ido == -1))
			transpose_self_multiply(A, workd + ipntr[0] - 1, workd + ipntr[1] - 1, Z);
	} while ((ido == 1) || (ido == -1));

	delete[] Z;
	if (info < 0)
	{
		//std::cout << "Error with dsaupd, info = " << info << "\n";
		//std::cout << "Check documentation in dsaupd\n\n";
	}
	else
	{
		dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat,
			&n, arpack_type[eigentype], &nev, &tol, resid, &ncv, v, &ldv,
			iparam, ipntr, workd, workl, &lworkl, &ierr);

		if (ierr != 0)
		{
			//std::cout << "Error with dseupd, info = " << ierr << "\n";
			//std::cout << "Check the documentation of dseupd.\n\n";
		}
		else if (info == 1)
		{
			//std::cout << "Maximum number of iterations reached.\n\n";
		}
		else if (info == 3)
		{
			//std::cout << "No shifts could be applied during implicit\n";
			//std::cout << "Arnoldi update, try increasing NCV.\n\n";
		}

		memcpy(eigenvalues, d, sizeof(double)*nev);
		int k = 0;
		for (int i = 0; i < nev; i++)
		{
			eigenvalues[i] = sqrt(fabs(d[i]));//sqrt(d[i]);
			int in = i*n;
			for (int j = 0; j < n; j++)
			{
				eigenvectors[k] = v[in + j];
				k++;
			}
		}

		delete[] resid;
		delete[] workd;
		delete[] workl;
		delete[] select;
		delete[] v;
		delete[] d;
	}

	return true;
}

bool solve_sym_eigensystem_by_ARPACK(Sparse_Matrix *A, int num_of_eigenvalue, int eigentype,
	double *eigenvalues, double *eigenvectors)
{
	int n = (int)A->cols();
	int nev = num_of_eigenvalue;
	int ido = 0;
	char bmat[2] = "I";
	double tol = 0.0;
	double *resid = new double[n];
	int ncv = 4 * nev;
	if (ncv > n) ncv = n;
	int ldv = n;
	double *v = new double[ldv*ncv];
	int iparam[11]; /* An array used to pass information to the routines
					about their functional modes. */
	iparam[0] = 1; // Specifies the shift strategy (1->exact)

	iparam[2] = 3 * n; // Maximum number of iterations

	iparam[6] = 1;/* Sets the mode of dsaupd.
				  1 is exact shifting,
				  2 is user-supplied shifts,
				  3 is shift-invert mode,
				  4 is buckling mode,
				  5 is Cayley mode. */

	int ipntr[11]; /* Indicates the locations in the work array workd
				   where the input and output vectors in the
				   callback routine are located. */

	double *workd = new double[3 * n];
	int lworkl = ncv*(ncv + 8);
	double *workl = new double[lworkl];
	int info = 0;
	int rvec = 1; // Changed from above
	int *select = new int[ncv];
	double *d = new double[nev];
	double sigma;
	int ierr;

	do
	{
		dsaupd_(&ido, bmat, &n, arpack_type[eigentype], &nev, &tol, resid,
			&ncv, v, &ldv, iparam, ipntr, workd, workl,
			&lworkl, &info);

		if ((ido == 1) || (ido == -1))
			multiply(A, workd + ipntr[0] - 1, workd + ipntr[1] - 1);
	} while ((ido == 1) || (ido == -1));

	if (info < 0)
	{
		//std::cout << "Error with dsaupd, info = " << info << "\n";
		//std::cout << "Check documentation in dsaupd\n\n";
	}
	else
	{
		dseupd_(&rvec, "All", select, d, v, &ldv, &sigma, bmat,
			&n, arpack_type[eigentype], &nev, &tol, resid, &ncv, v, &ldv,
			iparam, ipntr, workd, workl, &lworkl, &ierr);

		if (ierr != 0)
		{
			//std::cout << "Error with dseupd, info = " << ierr << "\n";
			//std::cout << "Check the documentation of dseupd or dneupd.\n\n";
		}
		else if (info == 1)
		{
			//std::cout << "Maximum number of iterations reached.\n\n";
		}
		else if (info == 3)
		{
			//std::cout << "No shifts could be applied during implicit\n";
			//std::cout << "Arnoldi update, try increasing NCV.\n\n";
		}

		memcpy(eigenvalues, d, sizeof(double)*nev);
		int k = 0;
		for (int i = 0; i < nev; i++)
		{
			int in = i*n;
			for (int j = 0; j < n; j++)
			{
				eigenvectors[k] = v[in + j];
				k++;
			}
		}
	}
	delete[] resid;
	delete[] workd;
	delete[] workl;
	delete[] select;
	delete[] v;
	delete[] d;
	return true;
}
#endif

#ifdef USE_SUPERLU

bool solve_by_SUPERLU(Sparse_Matrix *m_sparse_matrix)
{
	if (m_sparse_matrix == NULL)
	{
		return false;
	}
	Sparse_Matrix *mat = m_sparse_matrix;

	if (m_sparse_matrix->get_storage_type() != CCS || m_sparse_matrix->issym_store_upper_or_lower())
	{
		mat = convert_sparse_matrix_storage(m_sparse_matrix, CCS, m_sparse_matrix->issym_store_upper_or_lower() ? SYM_BOTH : m_sparse_matrix->get_symmetric_type());
	}
	int row = (int)mat->rows();
	int col = (int)mat->cols();
	std::vector<int> rowind, colptr;
	std::vector<double> values;
	mat->get_compress_data(rowind, colptr, values);
	int m_num_column_of_RHS = (int)mat->get_num_rhs();
	std::vector<double>& B = mat->get_rhs();
	std::vector<double>& solution = m_sparse_matrix->get_solution();

	SuperMatrix A, L, U, super_B;
	int      *perm_r; /* row permutations from partial pivoting */
	int      *perm_c; /* column permutation vector */
	int      info;
	superlu_options_t options;
	SuperLUStat_t stat;

	A.Dtype = SLU_D;
	A.Mtype = SLU_GE;
	A.Stype = SLU_NC;
	A.nrow = row;
	A.ncol = col;

	NCformat nc;
	nc.colptr = &colptr[0];
	nc.rowind = &rowind[0];
	nc.nzval = &values[0];
	nc.nnz = (int)values.size();
	A.Store = &nc;

	super_B.Dtype = SLU_D;
	super_B.Mtype = SLU_GE;
	super_B.Stype = SLU_DN;
	super_B.nrow = row;
	super_B.ncol = m_num_column_of_RHS;

	DNformat dn;
	super_B.Store = &dn;
	dn.lda = row;
	memcpy(&solution[0], &B[0], m_num_column_of_RHS*row*sizeof(double));
	dn.nzval = &solution[0];

	perm_r = intMalloc(row);
	perm_c = intMalloc(col);

	/* Set the default input options. */
	set_default_options(&options);
	options.ColPerm = COLAMD;

	/* Initialize the statistics variables. */
	StatInit(&stat);

	dgssv(&options, &A, perm_c, perm_r, &L, &U, &super_B, &stat, &info);

	/* De-allocate storage */
	SUPERLU_FREE(perm_r);
	SUPERLU_FREE(perm_c);
	Destroy_SuperNode_Matrix(&L);
	Destroy_CompCol_Matrix(&U);
	StatFree(&stat);
	if (mat != m_sparse_matrix)
	{
		delete mat;
	}
	return true;
}

#endif
//////////////////////////////////////////////////////////////////////////