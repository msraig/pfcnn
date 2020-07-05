#ifndef SPARSE_CONFIG_H
#define SPARSE_CONFIG_H

//	Special version for Geometry4Architecture

/** \defgroup MathSuite MathSuite: Sparse Matrix, Solver, QPSolver */

//! BLAS and LAPACK
/*!
*	<A HREF="http://www.tacc.utexas.edu/resources/software/"> GOTOBLAS 1.26 </A>
*	<A HREF="http://www.netlib.org/lapack/"> LAPACK 3.1.1 </A> 
*/
//////////////////////////////////////////////////////////////////////////
//#pragma comment (lib, "libopenblas.lib") 
#pragma comment (lib, "BLAS.lib") 
#pragma comment (lib, "CLAPACK.lib") 
#pragma comment (lib, "libf2c.lib") 

#define USE_CHOLMOD
//#define USE_TAUCS
#define USE_UMFPACK
//#define USE_ARPACK
//#define USE_SUPERLU

#endif //SPARSE_CONFIG_H
