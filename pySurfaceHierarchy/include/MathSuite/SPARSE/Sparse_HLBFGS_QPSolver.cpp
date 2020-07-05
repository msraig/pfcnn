#include "Sparse_HLBFGS_QPSolver.h"
#include <iostream>

void QP_new_iteration(const size_t niter, const size_t call_iter,
					  const size_t n_vars, const std::vector<double> &variables, const double &func_value,
					  const std::vector<double> &gradient, const double &gnorm, void *user_pointer)
{
	//std::cout << niter << " " << call_iter << " " << func_value << " " << gnorm << std::endl;
	//std::cout << niter << " " << call_iter << " " << func_value << " " << gnorm << " ===== "  << variables[0] << " " << variables[1] << std::endl;
}

void QP_eval_func(const size_t N, const std::vector<double> &x, double &f, std::vector<double> &g, void *user_supply)
{
	Sparse_QPSolver_by_HLBFGS *mpointer = (Sparse_QPSolver_by_HLBFGS *)user_supply;
	multiply(mpointer->get_Q(), const_cast<double *>(&x[0]), &g[0]);
	f = 0;

	for (size_t i = 0; i < N; i++)
	{
		f += x[i] * g[i] / 2 + mpointer->get_Q()->get_rhs()[i] * x[i];
		g[i] += mpointer->get_Q()->get_rhs()[i];
	}
}
//////////////////////////////////////////////////////////////////////////
void QP_constraint_func(const size_t n_eqns, const size_t n_ieqns,
						const std::vector<double> &x, std::vector<double> &func_values, std::vector< std::vector<std::pair<size_t, double> > > &constraint_jacobian,
						void *user_pointer)
{
	if (n_eqns == 0 || constraint_jacobian.size() != n_eqns + n_ieqns) { return; }

	Sparse_QPSolver_by_HLBFGS *mpointer = (Sparse_QPSolver_by_HLBFGS *)user_pointer;

	if (mpointer->get_A())
	{
		multiply(mpointer->get_A(), const_cast<double *>(&x[0]), &func_values[0]);

		for (size_t i = 0; i < mpointer->get_A()->rows(); i++)
		{
			func_values[i] -= mpointer->get_A()->get_rhs()[i];
		}

		if (mpointer->get_first_status())
		{
			const std::vector< Sparse_Vector * > &entryset = mpointer->get_A()->get_entryset();

			if (mpointer->get_A()->get_storage_type() != CCS)
			{
				for (size_t i = 0; i < entryset.size(); i++)
				{
					for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
					{
						constraint_jacobian[i].push_back(std::pair<size_t, double>(miter->index, miter->value));
					}
				}
			}
			else
			{
				for (size_t i = 0; i < entryset.size(); i++)
				{
					for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
					{
						constraint_jacobian[miter->index].push_back(std::pair<size_t, double>(i, miter->value));
					}
				}
			}
		}
	}

	if (mpointer->get_C())
	{
		multiply(mpointer->get_C(), const_cast<double *>(&x[0]), &func_values[n_eqns]);

		for (size_t i = 0; i < mpointer->get_C()->rows(); i++)
		{
			func_values[n_eqns + n_ieqns/2 + i] = mpointer->get_cupp()[i] - func_values[n_eqns + i];
			func_values[n_eqns + i] -= mpointer->get_clow()[i];
		}

		if (mpointer->get_first_status())
		{
			const std::vector< Sparse_Vector * > &entryset = mpointer->get_C()->get_entryset();

			if (mpointer->get_C()->get_storage_type() != CCS)
			{
				for (size_t i = 0; i < entryset.size(); i++)
				{
					for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
					{
						constraint_jacobian[i+n_eqns].push_back(std::pair<size_t, double>(miter->index, miter->value));
						constraint_jacobian[i+n_eqns + n_ieqns/2].push_back(std::pair<size_t, double>(miter->index, -miter->value));
					}
				}
			}
			else
			{
				for (size_t i = 0; i < entryset.size(); i++)
				{
					for (Sparse_Vector::iterator miter = entryset[i]->begin(); miter != entryset[i]->end(); miter++)
					{
						constraint_jacobian[miter->index+n_eqns].push_back(std::pair<size_t, double>(i, miter->value));
						constraint_jacobian[miter->index+n_eqns + n_ieqns/2].push_back(std::pair<size_t, double>(i, -miter->value));
					}
				}
			}
		}
	}

	mpointer->set_first_status(false);
}

Sparse_QPSolver_by_HLBFGS::Sparse_QPSolver_by_HLBFGS(size_t N_,
													 double *solution_,
													 Sparse_Matrix *Q_,
													 Sparse_Matrix *A_,
													 Sparse_Matrix *C_, double *clow_, double *cupp_,
													 bool warm_start,
													 double *lambda_copy, double *sigma_copy
													 )
													 :N(N_),solution(solution_),Q(Q_),A(A_),C(C_),clow(clow_),cupp(cupp_)
{
	if (Q == 0 || N == 0 || Q->issquare() == false)
	{ return; }

	first_call_constraints = true;
	HLBFGS m_hlbfgs;
	m_hlbfgs.set_verbose(false);
	double parameters[20];
	size_t info[20];
	m_hlbfgs.get_advanced_setting(parameters, info);
	parameters[5] = 1.0e-3;// ||g||/max(1,||x||)
	parameters[6] = 1.0e-4;// ||g||
	m_hlbfgs.set_advanced_setting(parameters, info);
	m_hlbfgs.set_qp_solver(true);
	m_hlbfgs.set_number_of_variables(N);
	m_hlbfgs.set_number_of_equalities(A?A->rows():0);
	m_hlbfgs.set_number_of_inequalities(C?2*C->rows():0);
	m_hlbfgs.set_func_callback(QP_eval_func, 0, QP_constraint_func, QP_new_iteration, 0);

	if (A || C)
	{
		m_hlbfgs.optimize_with_constraints(solution, 15, std::max((size_t)30, N), this, warm_start, lambda_copy, sigma_copy);
		m_hlbfgs.copy_lambda(lambda_copy);
		m_hlbfgs.copy_sigma(sigma_copy);
	}
	else
	{ m_hlbfgs.optimize_without_constraints(solution, std::max((size_t)30, N), this); }
}