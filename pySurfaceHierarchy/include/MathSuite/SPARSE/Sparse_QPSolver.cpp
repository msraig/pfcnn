#include "Sparse_Config.h"
#include "Sparse_QPSolver.h"

#define USE_MOSEK
#ifndef USE_MOSEK

#include "QpGenData.h"
#include "QpGenVars.h"
#include "QpGenResiduals.h"
#include "GondzioSolver.h"
#include "QpGenSparseMa27.h"

#pragma comment (lib, "libMA27.lib")
#pragma comment (lib, "libooqpbase.lib")
#pragma comment (lib, "libooqpgensparse.lib")
#pragma comment (lib, "libooqpgondzio.lib")
#pragma comment (lib, "libooqpsparse.lib")

int Sparse_QPSolver(int N,
					double *solution,
					Sparse_Matrix *Q,
					double *c,
					Sparse_Matrix *A,
					double *b,
					Sparse_Matrix *C,
					double *d,
					double *f,
					char *di,
					char *fi,
					double *l,
					double *u,
					char *li,
					char *ui
					)
{
	if (Q == 0 || Q->issymmetric() == false || Q->rows() != N)
	{
		return -1;
	}

	std::vector<int> Q_row_vec, Q_col_vec;
	std::vector<double> Q_val_vec;
	int *Q_rowid = NULL;
	int *Q_colid = NULL;
	double *Q_values = NULL;
	int Qnnz = 0;
	Sparse_Matrix *tmp_Q = Q;

	if (Q != NULL)
	{
		if (Q->get_storage_type() != TRIPLE || Q->get_symmetric_type() != SYM_LOWER)
		{
			tmp_Q = convert_sparse_matrix_storage(Q, TRIPLE, SYM_LOWER);
		}

		tmp_Q->get_compress_data(Q_row_vec, Q_col_vec, Q_val_vec);
		Q_rowid = &Q_row_vec[0];
		Q_colid = &Q_col_vec[0];
		Q_values = &Q_val_vec[0];
		Qnnz = (int)Q_val_vec.size();
	}

	int A_row = 0;
	int *A_rowid = NULL;
	int *A_colid = NULL;
	double *A_values = NULL;
	int Annz = 0;
	std::vector<int> A_row_vec, A_col_vec;
	std::vector<double> A_val_vec;
	Sparse_Matrix *tmp_A = A;

	if (A != NULL)
	{
		if (A->get_storage_type() != TRIPLE || A->issym_store_upper_or_lower())
		{
			tmp_A = convert_sparse_matrix_storage(A, TRIPLE, A->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);
		}

		tmp_A->get_compress_data(A_row_vec, A_col_vec, A_val_vec);
		A_rowid = &A_row_vec[0];
		A_colid = &A_col_vec[0];
		A_values = &A_val_vec[0];
		Annz = (int)A_val_vec.size();
		A_row = (int)A->rows();
	}

	int C_row = 0;
	int *C_rowid = NULL;
	int *C_colid = NULL;
	double *C_values = NULL;
	int Cnnz = 0;
	std::vector<int> C_row_vec, C_col_vec;
	std::vector<double> C_val_vec;
	Sparse_Matrix *tmp_C = C;

	if (C != NULL)
	{
		if (C->get_storage_type() != TRIPLE || C->issym_store_upper_or_lower())
		{
			tmp_C = convert_sparse_matrix_storage(C, TRIPLE, C->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);
		}

		tmp_C->get_compress_data(C_row_vec, C_col_vec, C_val_vec);
		C_rowid = &C_row_vec[0];
		C_colid = &C_col_vec[0];
		C_values = &C_val_vec[0];
		Cnnz = (int)C_val_vec.size();
		C_row = (int)C->rows();
	}

	QpGenSparseMa27 *qp = new QpGenSparseMa27( N, A_row, C_row, Qnnz, Annz, Cnnz);
	QpGenData *prob = (QpGenData *) qp->copyDataFromSparseTriple(
		c, Q_rowid, Qnnz, Q_colid, Q_values,
		l, li, u, ui,
		A_rowid, Annz, A_colid, A_values, b,
		C_rowid, Cnnz, C_colid, C_values,
		d, di, f, fi);
	QpGenVars *vars = (QpGenVars *) qp->makeVariables( prob );
	QpGenResiduals *resid = (QpGenResiduals *) qp->makeResiduals( prob );
	GondzioSolver *s = new GondzioSolver( qp, prob );
	int ierr = s->solve(prob, vars, resid);
	vars->x->copyIntoArray(solution);

	if (Q && tmp_Q != Q)
	{
		delete tmp_Q;
	}

	if (A && tmp_A != A)
	{
		delete tmp_A;
	}

	if (C && tmp_C != C)
	{
		delete tmp_C;
	}

	delete qp;
	delete prob;
	delete vars;
	delete resid;
	delete s;
	return ierr;
	return 0;
}

#else

#include "C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\h\\mosek.h"
#pragma comment (lib, "C:\\Program Files\\Mosek\\7\\tools\\platform\\win64x86\\bin\\mosek64_7_1.lib")


int Sparse_QPSolver(int N,
					double *solution,
					Sparse_Matrix *Q,
					double *c,
					Sparse_Matrix *A,
					double *b,
					Sparse_Matrix *C,
					double *d,
					double *f,
					char *di,
					char *fi,
					double *l,
					double *u,
					char *li,
					char *ui
					)
{
	if (Q == 0 || Q->issymmetric() == false || Q->rows() != N)
	{
		return -1;
	}

	int NUMCON = (A ? (int)A->rows() : 0) + (C ? (int)C->rows() : 0);
	int NUMVAR = (int)Q->rows();
	MSKenv_t      env = NULL;
	MSKtask_t     task = NULL;
	MSKrescodee   r;
	r = MSK_makeenv(&env, NULL);
	r = MSK_maketask(env, NUMCON, NUMVAR, &task);
	char optname[] = {"MSK_IPAR_NUM_THREADS"};
	MSK_putnaintparam(task, optname, 0);
	r = MSK_linkfunctotaskstream(task, MSK_STREAM_LOG, NULL, NULL);
	r = MSK_appendcons(task, NUMCON);
	r = MSK_appendvars(task, NUMVAR);

	for (int i = 0; i < NUMVAR && r == MSK_RES_OK; i++)
	{
		r = MSK_putcj(task, i, c[i]);
		MSKboundkeye mkey = MSK_BK_FR;

		if (li && ui)
		{
			if (li[i] == 0 && ui[i] == 1)
			{
				mkey = MSK_BK_UP;
			}
			else if (li[i] == 1 && ui[i] == 0)
			{
				mkey = MSK_BK_LO;
			}
			else if (li[i] == 1 && ui[i] == 1)
			{
				mkey = MSK_BK_RA;
			}
		}
		else if (li)
		{
			if (li[i] == 1)
			{ mkey = MSK_BK_LO; }
		}
		else if (ui)
		{
			if (ui[i] == 1)
			{ mkey = MSK_BK_UP; }
		}

		r = MSK_putvarbound(task, i, mkey, l ? l[i] : -MSK_INFINITY, u ? u[i] : MSK_INFINITY);
	}

	int constraints_counter = 0;

	if (A)
	{
		for (size_t i = 0; i < A->rows() && r == MSK_RES_OK; i++)
		{
			r = MSK_putconbound(task, (int)i, MSK_BK_FX, b[i], b[i]);
		}

		constraints_counter = (int)A->rows();
	}

	if (C)
	{
		for (size_t i = 0; i < C->rows() && r == MSK_RES_OK; i++)
		{
			MSKboundkeye mkey = MSK_BK_FR;

			if (di && fi)
			{
				if (di[i] == 0 && fi[i] == 1)
				{
					mkey = MSK_BK_UP;
				}
				else if (di[i] == 1 && fi[i] == 0)
				{
					mkey = MSK_BK_LO;
				}
				else if (di[i] == 1 && fi[i] == 1)
				{
					mkey = MSK_BK_RA;
				}
			}
			else if (di)
			{
				if (di[i] == 1)
				{ mkey = MSK_BK_LO; }
			}
			else if (fi)
			{
				if (fi[i] == 1)
				{ mkey = MSK_BK_UP; }
			}

			r = MSK_putconbound(task, constraints_counter + (int)i, mkey, d ? d[i] : -MSK_INFINITY, f ? f[i] : MSK_INFINITY);
		}
	}

	std::vector<int> Q_row_vec, Q_col_vec;
	std::vector<double> Q_val_vec;
	int *Q_rowid = NULL;
	int *Q_colid = NULL;
	double *Q_values = NULL;
	int Qnnz = 0;
	Sparse_Matrix *tmp_Q = Q;

	if (Q != NULL)
	{
		if (Q->get_storage_type() != TRIPLE || Q->get_symmetric_type() != SYM_LOWER)
		{
			tmp_Q = convert_sparse_matrix_storage(Q, TRIPLE, SYM_LOWER);
		}

		tmp_Q->get_compress_data(Q_row_vec, Q_col_vec, Q_val_vec);
		Q_rowid = &Q_row_vec[0];
		Q_colid = &Q_col_vec[0];
		Q_values = &Q_val_vec[0];
		Qnnz = (int)Q_val_vec.size();
		r = MSK_putqobj(task, Qnnz, Q_rowid, Q_colid, Q_values);
	}

	Sparse_Matrix *tmp_constaint_mat = 0;
	Sparse_Matrix *constaint_mat = 0;
	Sparse_Matrix *tmp_A = A;
	Sparse_Matrix *tmp_C = C;

	if (A && C)
	{
		tmp_constaint_mat = new Sparse_Matrix(NUMCON, NUMVAR, NOSYM, CCS);

		if (A != NULL)
		{
			if (A->get_storage_type() != CCS || A->issym_store_upper_or_lower())
			{
				tmp_A = convert_sparse_matrix_storage(A, CCS, A->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);
			}
		}

		if (C != NULL)
		{
			if (C->get_storage_type() != CCS || C->issym_store_upper_or_lower())
			{
				tmp_C = convert_sparse_matrix_storage(C, CCS, C->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);
			}
		}

		const std::vector<Sparse_Vector *> &entryset = tmp_A->get_entryset();

		for (size_t i = 0; i < entryset.size(); i++)
		{
			for (Sparse_Vector::const_iterator iter = entryset[i]->begin();
				iter != entryset[i]->end(); iter++)
			{
				tmp_constaint_mat->fill_entry(iter->index, i, iter->value);
			}
		}

		const std::vector<Sparse_Vector *> &entryset2 = tmp_C->get_entryset();

		for (size_t i = 0; i < entryset2.size(); i++)
		{
			for (Sparse_Vector::const_iterator iter = entryset2[i]->begin();
				iter != entryset2[i]->end(); iter++)
			{
				tmp_constaint_mat->fill_entry(tmp_A->rows() + iter->index, i, iter->value);
			}
		}
	}
	else
	{
		constaint_mat = A ? A : C;
		tmp_constaint_mat = constaint_mat;
		if (constaint_mat)
		{
			if (constaint_mat->get_storage_type() != CCS || constaint_mat->issym_store_upper_or_lower())
			{
				tmp_constaint_mat = convert_sparse_matrix_storage(constaint_mat, CCS, constaint_mat->issym_store_upper_or_lower() ? SYM_BOTH : NOSYM);
			}
		}
	}

	int *constaint_mat_rowid = 0;
	int *constaint_mat_colid = 0;
	double *constaint_mat_values = 0;
	std::vector<int> constaint_mat_row_vec, constaint_mat_col_vec;
	std::vector<double> constaint_mat_val_vec;

	if (tmp_constaint_mat)
	{
		tmp_constaint_mat->get_compress_data(constaint_mat_row_vec, constaint_mat_col_vec, constaint_mat_val_vec);
		constaint_mat_rowid = &constaint_mat_row_vec[0];
		constaint_mat_colid = &constaint_mat_col_vec[0];
		constaint_mat_values = &constaint_mat_val_vec[0];

		for (size_t j = 0; j < tmp_constaint_mat->cols() && r == MSK_RES_OK; j++)
		{
			r = MSK_putacol(task,
				(int)j,                       /* Variable (column) index.*/
				constaint_mat_colid[j + 1] - constaint_mat_colid[j], /* Number of non-zeros in column j.*/
				&constaint_mat_rowid[constaint_mat_colid[j]],        /* Pointer to row indexes of column j.*/
				constaint_mat_values + constaint_mat_colid[j]);    /* Pointer to Values of column j.*/
		}
	}

	MSKrescodee trmcode;
	r = MSK_optimizetrm(task,&trmcode);
	MSK_solutionsummary (task,MSK_STREAM_MSG);

	if (r == MSK_RES_OK)
	{
		MSKsolstae solsta;
		MSK_getsolsta (task,MSK_SOL_ITR,&solsta);
		MSK_getxx(task, MSK_SOL_ITR, solution);
	}

	if (Q && tmp_Q != Q)
	{
		delete tmp_Q;
	}

	if (tmp_constaint_mat && constaint_mat != tmp_constaint_mat)
	{
		delete tmp_constaint_mat;
	}

	if (A && tmp_A != A)
	{
		delete tmp_A;
	}

	if (C && tmp_C != C)
	{
		delete tmp_C;
	}

	return r != MSK_RES_OK;
}
#endif
