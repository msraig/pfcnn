#ifndef SPARSE_QPSOLVER_H
#define SPARSE_QPSOLVER_H

#include "Sparse_Matrix.h"

/*! \addtogroup MathSuite
//@{
//////////////////////////////////////////////////////////////////////////
// OOQP: Object-Oriented Software for Quadratic Programming

// <A HREF= "http://pages.cs.wisc.edu/~swright/ooqp/"> OOQP </A>

//! Quadratic Programming Solver by OOQP library
/*!
*	minimize
\f[
1/2 X^T Q X + c^T X
\f]
subject to
\f$ A X = b, d \leq C X \leq f, l \leq X \leq u \f$.
Q, A, C are triple-style sparse matrices.
*
*	\param N number of variables
*	\param solution the array storing the solution
*	\param Q lower symmetric sparse matrix
*	\param c linear term
*	\param A linear constraints
*	\param b linear constraints value
*	\param C sparse matrix
*	\param d lower bound of C
*	\param f upper bound of C
*	\param di indicator of d: 0-free, 1-set
*	\param fi indicator of f: 0-free, 1-set
*	\param l lower bound of X
*	\param u upper bound of X
*	\param li indicator of l: 0-free, 1-set
*	\param ui indicator of u: 0-free, 1-set
*	\return solver's err info
*/
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
					);
//@}

#endif //SPARSE_QPSOLVER_H
