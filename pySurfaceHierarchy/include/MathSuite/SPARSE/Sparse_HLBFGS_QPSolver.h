#ifndef SPARSE_HLBFGS_QPSOLVER_H
#define SPARSE_HLBFGS_QPSOLVER_H

#include "Sparse_Matrix.h"
#include "HLBFGS.h"

void QP_eval_func(const size_t N, const std::vector<double> &x, double &f, std::vector<double> &g, void *user_supply);
void QP_constraint_func(const size_t n_eqns, const size_t n_ieqns,
						const std::vector<double> &x, std::vector<double> &func_values, std::vector< std::vector<std::pair<size_t, double> > > &constraint_jacobian,
						void *user_pointer);
void QP_new_iteration(const size_t niter, const size_t call_iter,
					  const size_t n_vars, const std::vector<double> &variables, const double &func_value,
					  const std::vector<double> &gradient, const double &gnorm, void *user_pointer);

class Sparse_QPSolver_by_HLBFGS
{
public:
	// min 1/2 x^T Q x + Qb x
	// s.t.
	// A x = Ab
	// clow_ <= C x <= cupp
	Sparse_QPSolver_by_HLBFGS(size_t N_,
		double *solution_,
		Sparse_Matrix *Q_,
		Sparse_Matrix *A_,
		Sparse_Matrix *C_, double *clow_, double *cupp_,
		bool warm_start = false,
		double *lambda_copy = 0, double *sigma_copy = 0
		);

	Sparse_Matrix *get_Q() {return Q;}
	Sparse_Matrix *get_A() {return A;}
	Sparse_Matrix *get_C() {return C;}
	double *get_clow() {return clow;}
	double *get_cupp() {return cupp;}
	bool get_first_status() {return first_call_constraints; }
	void set_first_status(bool status) {first_call_constraints = status; }
private:
	size_t N;
	double *solution;
	Sparse_Matrix *Q, *A, *C;
	double *clow, *cupp;
	bool first_call_constraints;
};
//@}

#endif //SPARSE_QPSOLVER_H
