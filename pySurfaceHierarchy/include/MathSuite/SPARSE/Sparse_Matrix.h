#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <vector>
#include <algorithm>
#include <cstdlib>

#include "Sparse_Entry.h"

// \addtogroup MathSuite
//@{

//! Symmetric status
enum SPARSE_SYMMETRIC_TYPE
{
	NOSYM, /*!< general case */
	SYM_UPPER, /*!< symmetric (store upper triangular part) */
	SYM_LOWER, /*!< symmetric (store lower triangular part) */
	SYM_BOTH   /*!< symmetric (store both upper and lower triangular part) */
};

//!     Storage
enum SPARSE_STORAGE_TYPE
{
	CCS, /*!< compress column format */
	CRS, /*!< compress row format */
	TRIPLE /*!< row-wise coordinate format */
};

//! Array type
enum ARRAY_TYPE
{
	FORTRAN_TYPE, /*!< the index starts from 1 */
	C_TYPE /*!< the index starts from 0 */
};

typedef std::vector<Sparse_Entry> Sparse_Vector;

//////////////////////////////////////////////////////////////////////////
//! Lite Sparse Matrix Class
class Sparse_Matrix
{
private:

	SPARSE_SYMMETRIC_TYPE symmetric_type;
	SPARSE_STORAGE_TYPE storage_type;

	size_t nrows; //!< number of rows
	size_t ncols; //!< number of columns
	size_t m_num_column_of_RHS; //!< number of rhs
	std::vector< Sparse_Vector* > entryset;
	std::vector< double > right_hand_side, solution;

public:

	//! Sparse matrix constructor
	/*!
	* \param m row dimension
	* \param n column dimension
	* \param symmetrictype
	* \param symmetrictype 
	*/
	Sparse_Matrix(const size_t m, const size_t n, const SPARSE_SYMMETRIC_TYPE symmetrictype = NOSYM, const SPARSE_STORAGE_TYPE storagetype = CCS, const size_t num_column_of_RHS = 1) :
	  symmetric_type(symmetrictype), storage_type(storagetype), nrows(m), ncols(n), m_num_column_of_RHS(num_column_of_RHS)
	  {
		  if (nrows != ncols)
		  {
			  symmetric_type = NOSYM;
		  }
		  if (storage_type == CCS)
		  {
			  entryset.resize(ncols);
			  for (size_t i = 0; i < ncols; i++)
			  {
				  entryset[i] = new Sparse_Vector;
			  }
		  }
		  else
		  {
			  entryset.resize(nrows);
			  for (size_t i = 0; i < nrows; i++)
			  {
				  entryset[i] = new Sparse_Vector;
			  }
		  }
		  right_hand_side.resize(m_num_column_of_RHS*nrows);
		  solution.resize(m_num_column_of_RHS*ncols);
		  std::fill(right_hand_side.begin(), right_hand_side.end(), 0);
		  std::fill(solution.begin(), solution.end(), 0);
	  }

	  Sparse_Matrix(const Sparse_Matrix* mat)
	  {
		  symmetric_type = mat->get_symmetric_type();
		  storage_type = mat->get_storage_type();
		  const std::vector<Sparse_Vector*>& mat_entryset = mat->get_entryset();
		  nrows = mat->rows(); ncols = mat->cols();
		  m_num_column_of_RHS = mat->get_num_rhs();
		  entryset.resize(mat_entryset.size());
		  for (size_t i = 0; i < mat_entryset.size(); i++)
		  {
			  entryset[i] = new Sparse_Vector;
			  entryset[i]->assign(mat_entryset[i]->begin(), mat_entryset[i]->end());
		  }
		  right_hand_side.assign(mat->get_rhs().begin(), mat->get_rhs().end());
		  solution.assign(mat->get_solution().begin(), mat->get_solution().end());
	  }
	  //! Sparse matrix destructor
	  ~Sparse_Matrix()
	  {
		  for (size_t i = 0; i < entryset.size(); i++)
		  {
			  delete entryset[i];
		  }
	  }

	  //! clear
	  void reset2zero()
	  {
		  for (size_t i = 0; i < entryset.size(); i++)
		  {
			  entryset[i]->resize(0);
		  }
		  clear_rhs();
	  }

	  //! clear_right_hand_side
	  void clear_rhs()
	  {
		  std::fill(right_hand_side.begin(), right_hand_side.end(), 0);
	  }

	  //! adjust rhs
	  void adjust_num_of_rhs(const size_t num_column_of_RHS)
	  {
		  m_num_column_of_RHS = num_column_of_RHS;
		  right_hand_side.resize(m_num_column_of_RHS*nrows);
		  solution.resize(m_num_column_of_RHS*ncols);
		  std::fill(right_hand_side.begin(), right_hand_side.end(), 0);
		  std::fill(solution.begin(), solution.end(), 0);
	  }

	  //! fill the right hand side
	  void fill_rhs_entry(const size_t row_index, const double val)
	  {
		  if (row_index < right_hand_side.size())
			  right_hand_side[row_index] += val;
	  }
	  //! set the right hand side
	  void set_rhs_entry(const size_t row_index, const double val)
	  {
		  if (row_index < right_hand_side.size())
			  right_hand_side[row_index] = val;
	  }
	  //! fill the right hand side
	  void fill_rhs_entry(const size_t row_index, const size_t which_col, const double val)
	  {
		  size_t position = which_col * nrows + row_index;
		  if (position < right_hand_side.size())
			  right_hand_side[position] += val;
	  }
	  //! set the right hand side
	  void set_rhs_entry(const size_t row_index, const size_t which_col, const double val)
	  {
		  size_t position = which_col * nrows + row_index;
		  if (position < right_hand_side.size())
			  right_hand_side[position] = val;
	  }
	  //! fill matrix entry \f$  Mat_{row_index, col_index} += val \f$
	  void fill_entry(const size_t row_index, const size_t col_index, const double val = 0)
	  {
		  if (row_index >= nrows || col_index >= ncols || val == 0)
			  return;

		  if (symmetric_type == NOSYM)
		  {
			  fill_internal_entry(row_index, col_index, val);
		  }
		  else if (symmetric_type == SYM_UPPER)
		  {
			  if (row_index <= col_index)
			  {
				  fill_internal_entry(row_index, col_index, val);
			  }
			  else
			  {
				  fill_internal_entry(col_index, row_index, val);
			  }
		  }
		  else if (symmetric_type == SYM_LOWER)
		  {
			  if (row_index <= col_index)
			  {
				  fill_internal_entry(col_index, row_index, val);
			  }
			  else
			  {
				  fill_internal_entry(row_index, col_index, val);
			  }
		  }
		  else if (symmetric_type == SYM_BOTH)
		  {
			  fill_internal_entry(row_index, col_index, val);

			  if (row_index != col_index)
			  {
				  fill_internal_entry(col_index, row_index, val);
			  }
		  }
	  }
	  //! set matrix entry \f$  Mat_{row_index, col_index} = val \f$
	  void set_entry(const size_t row_index, const size_t col_index, const double val = 0)
	  {
		  if (row_index >= nrows || col_index >= ncols)
			  return;

		  if (symmetric_type == NOSYM)
		  {
			  set_internal_entry(row_index, col_index, val);
		  }
		  else if (symmetric_type == SYM_UPPER)
		  {
			  if (row_index <= col_index)
			  {
				  set_internal_entry(row_index, col_index, val);
			  }
			  else
			  {
				  set_internal_entry(col_index, row_index, val);
			  }
		  }
		  else if (symmetric_type == SYM_LOWER)
		  {
			  if (row_index <= col_index)
			  {
				  set_internal_entry(col_index, row_index, val);
			  }
			  else
			  {
				  set_internal_entry(row_index, col_index, val);
			  }
		  }
		  else if (symmetric_type == SYM_BOTH)
		  {
			  set_internal_entry(row_index, col_index, val);

			  if (row_index != col_index)
			  {
				  set_internal_entry(col_index, row_index, val);
			  }
		  }
	  }
	  //! get matrix entry $Mat_{row_index, col_index}$
	  double get_entry(const size_t row_index, const size_t col_index)
	  {
		  if (row_index >= nrows || col_index >= ncols)
			  return 0;

		  size_t ri = row_index, ci = col_index;

		  if (symmetric_type == SYM_LOWER && ri < ci)
		  {
			  std::swap(ri, ci);
		  }
		  else if (symmetric_type == SYM_UPPER && ri > ci)
		  {
			  std::swap(ri, ci);
		  }

		  const size_t search_index = (storage_type == CCS ? ri : ci);
		  const size_t pos_index = (storage_type == CCS ? ci : ri);

		  Sparse_Entry forcompare(search_index);
		  Sparse_Vector::iterator iter = 
			  std::lower_bound(entryset[pos_index]->begin(), entryset[pos_index]->end(), forcompare);
		  if (iter != entryset[pos_index]->end())
		  {
			  if (iter->index == search_index)
			  {
				  return iter->value;
			  }
		  }
		  return 0;
	  }
	  //! get the pointer of matrix entry $Mat_{row_index, col_index}$
	  double* get_entry_pointer(const size_t row_index, const size_t col_index)
	  {
		  if (row_index >= nrows || col_index >= ncols)
			  return 0;

		  size_t ri = row_index, ci = col_index;

		  if (symmetric_type == SYM_LOWER && ri < ci)
		  {
			  std::swap(ri, ci);
		  }
		  else if (symmetric_type == SYM_UPPER && ri > ci)
		  {
			  std::swap(ri, ci);
		  }

		  const size_t search_index = (storage_type == CCS ? ri : ci);
		  const size_t pos_index = (storage_type == CCS ? ci : ri);

		  Sparse_Entry forcompare(search_index);
		  Sparse_Vector::iterator iter = 
			  std::lower_bound(entryset[pos_index]->begin(), entryset[pos_index]->end(), forcompare);
		  if (iter != entryset[pos_index]->end())
		  {
			  if (iter->index == search_index)
			  {
				  return &(iter->value);
			  }
		  }
		  return 0;
	  }
	  //////////////////////////////////////////////////////////////////////////

	  //! get the number of nonzeros
	  inline const size_t get_nonzeros() const
	  {
		  size_t nnz = 0;
		  for (size_t i = 0; i < entryset.size(); i++)
		  {
			  nnz += entryset[i]->size();
		  }
		  return nnz;
	  }
	  //! get the row dimension
	  inline const size_t rows() const 
	  {
		  return nrows;
	  }
	  //! get the column dimension
	  inline const size_t cols() const 
	  {
		  return ncols;
	  }
	  //! return the number column of the right hand side
	  inline const size_t get_num_rhs() const
	  {
		  return m_num_column_of_RHS;
	  }
	  //! return the symmetric state
	  inline const bool issymmetric() const 
	  {
		  return symmetric_type != NOSYM;
	  }

	  //! tell whether the matrix is upper or lower symmetric
	  inline const bool issym_store_upper_or_lower() const 
	  {
		  return (symmetric_type == SYM_LOWER) || (symmetric_type == SYM_UPPER);
	  }

	  //! return symmetric state
	  inline const SPARSE_SYMMETRIC_TYPE get_symmetric_type() const
	  {
		  return symmetric_type;
	  }

	  //! tell whether the matrix is square
	  inline const bool issquare() const
	  {
		  return nrows == ncols;
	  }

	  //! return the storage format
	  inline const SPARSE_STORAGE_TYPE get_storage_type() const
	  {
		  return storage_type;
	  }
	  //! get the entryset
	  inline const std::vector<Sparse_Vector*>& get_entryset() const
	  {
		  return entryset;
	  }
	  inline std::vector<Sparse_Vector*>& get_entryset() 
	  {
		  return entryset;
	  }
	  //! get the rhs
	  inline const std::vector<double>& get_rhs() const
	  {
		  return right_hand_side;
	  }
	  inline std::vector<double>& get_rhs()
	  {
		  return right_hand_side;
	  }
	  //! get the solution
	  inline const std::vector<double>& get_solution() const
	  {
		  return solution;
	  }
	  inline std::vector<double>& get_solution()
	  {
		  return solution;
	  }
	  //////////////////////////////////////////////////////////////////////////
	  void get_compress_data(std::vector<size_t>& rowind, std::vector<size_t>& colptr, std::vector<double>& values, const ARRAY_TYPE array_type = C_TYPE,
		  std::vector<double>* diag = 0)
	  {
		  if (diag && nrows == ncols)
		  {
			  std::fill(diag->begin(), diag->end(), 0.0);
			  for (size_t i = 0; i < diag->size(); i++)
			  {
				  Sparse_Entry forcompare(i);
				  Sparse_Vector::iterator iter = 
					  std::lower_bound(entryset[i]->begin(), entryset[i]->end(), forcompare);
				  if ( iter != entryset[i]->end() && iter->index == i )
				  {
					  (*diag)[i] = iter->value;
					  entryset[i]->erase(iter);
				  }
			  }
		  }

		  int inc = (array_type == C_TYPE ? 0 : 1);
		  if (storage_type == CCS)
		  {
			  colptr.resize(ncols + 1);
			  colptr[0] = inc;
			  for (size_t j = 1; j < ncols + 1; j++)
			  {
				  colptr[j] = entryset[j - 1]->size() + colptr[j - 1];
			  }

			  size_t nonzeros = colptr[ncols];

			  if (nonzeros > 0)
			  {
				  rowind.resize(nonzeros);
				  values.resize(nonzeros);

				  size_t k = 0;
				  for (size_t i = 0; i < entryset.size(); i++)
				  {
					  for (Sparse_Vector::const_iterator iter = entryset[i]->begin();
						  iter != entryset[i]->end(); iter++)
					  {
						  rowind[k] = iter->index + inc;
						  values[k] = iter->value;
						  k++;
					  }
				  }
			  }
		  }
		  else if (storage_type == CRS)
		  {
			  rowind.resize(nrows + 1);
			  rowind[0] = inc;
			  for (size_t j = 1; j < nrows + 1; j++)
			  {
				  rowind[j] = entryset[j - 1]->size() + rowind[j - 1];
			  }
			  size_t nonzeros = rowind[nrows];
			  if (nonzeros > 0)
			  {
				  colptr.resize(nonzeros);
				  values.resize(nonzeros);

				  size_t k = 0;
				  for (size_t i = 0; i < entryset.size(); i++)
				  {
					  for (Sparse_Vector::const_iterator iter = entryset[i]->begin();
						  iter != entryset[i]->end(); iter++)
					  {
						  colptr[k] = iter->index + inc;
						  values[k] = iter->value;
						  k++;
					  }
				  }
			  }
		  }
		  else if (storage_type == TRIPLE)
		  {
			  size_t nonzeros = 0;
			  for (size_t i = 0; i < nrows; i++)
			  {
				  nonzeros += entryset[i]->size();
			  }

			  if (nonzeros > 0)
			  {
				  rowind.resize(nonzeros);
				  colptr.resize(nonzeros);
				  values.resize(nonzeros);

				  size_t k = 0;
				  for (size_t i = 0; i < entryset.size(); i++)
				  {
					  for (Sparse_Vector::const_iterator iter = entryset[i]->begin();
						  iter != entryset[i]->end(); iter++)
					  {
						  rowind[k] = i + inc;
						  colptr[k] = iter->index + inc;
						  values[k] = iter->value;
						  k++;
					  }
				  }
			  }
		  }
	  }
	  //////////////////////////////////////////////////////////////////////////
	  void get_compress_data(std::vector<int>& rowind, std::vector<int>& colptr, std::vector<double>& values, const ARRAY_TYPE array_type = C_TYPE,
		  std::vector<double>* diag = 0)
	  {
		  if (diag && nrows == ncols)
		  {
			  std::fill(diag->begin(), diag->end(), 0.0);
			  for (size_t i = 0; i < diag->size(); i++)
			  {
				  Sparse_Entry forcompare(i);
				  Sparse_Vector::iterator iter = 
					  std::lower_bound(entryset[i]->begin(), entryset[i]->end(), forcompare);
				  if ( iter != entryset[i]->end() && iter->index == i )
				  {
					  (*diag)[i] = iter->value;
					  entryset[i]->erase(iter);
				  }
			  }
		  }

		  int inc = (array_type == C_TYPE ? 0 : 1);
		  if (storage_type == CCS)
		  {
			  colptr.resize(ncols + 1);
			  colptr[0] = inc;
			  for (size_t j = 1; j < ncols + 1; j++)
			  {
				  colptr[j] = (int) ( entryset[j - 1]->size() + colptr[j - 1] );
			  }

			  size_t nonzeros = colptr[ncols];

			  if (nonzeros > 0)
			  {
				  rowind.resize(nonzeros);
				  values.resize(nonzeros);

				  size_t k = 0;
				  for (size_t i = 0; i < entryset.size(); i++)
				  {
					  for (Sparse_Vector::const_iterator iter = entryset[i]->begin();
						  iter != entryset[i]->end(); iter++)
					  {
						  rowind[k] = (int)(iter->index + inc);
						  values[k] = iter->value;
						  k++;
					  }
				  }
			  }
		  }
		  else if (storage_type == CRS)
		  {
			  rowind.resize(nrows + 1);
			  rowind[0] = inc;
			  for (size_t j = 1; j < nrows + 1; j++)
			  {
				  rowind[j] = (int)(entryset[j - 1]->size() + rowind[j - 1]);
			  }
			  size_t nonzeros = rowind[nrows];
			  if (nonzeros > 0)
			  {
				  colptr.resize(nonzeros);
				  values.resize(nonzeros);

				  size_t k = 0;
				  for (size_t i = 0; i < entryset.size(); i++)
				  {
					  for (Sparse_Vector::const_iterator iter = entryset[i]->begin();
						  iter != entryset[i]->end(); iter++)
					  {
						  colptr[k] = (int)(iter->index + inc);
						  values[k] = iter->value;
						  k++;
					  }
				  }
			  }
		  }
		  else if (storage_type == TRIPLE)
		  {
			  size_t nonzeros = 0;
			  for (size_t i = 0; i < nrows; i++)
			  {
				  nonzeros += entryset[i]->size();
			  }

			  if (nonzeros > 0)
			  {
				  rowind.resize(nonzeros);
				  colptr.resize(nonzeros);
				  values.resize(nonzeros);

				  size_t k = 0;
				  for (size_t i = 0; i < entryset.size(); i++)
				  {
					  for (Sparse_Vector::const_iterator iter = entryset[i]->begin();
						  iter != entryset[i]->end(); iter++)
					  {
						  rowind[k] = (int)(i + inc);
						  colptr[k] = (int)(iter->index + inc);
						  values[k] = iter->value;
						  k++;
					  }
				  }
			  }
		  }
	  }
	  //! remove sym info
	  inline void remove_sym_info()
	  {
		  symmetric_type = NOSYM;
	  }
	  //////////////////////////////////////////////////////////////////////////
private:
	//! fill_internal_entry
	void fill_internal_entry(const size_t row_index, const size_t col_index, const double val)
	{
		const size_t search_index = (storage_type == CCS ? row_index : col_index);
		const size_t pos_index = (storage_type == CCS ? col_index : row_index);

		Sparse_Entry forcompare(search_index);
		Sparse_Vector::iterator iter =
			std::lower_bound(entryset[pos_index]->begin(), entryset[pos_index]->end(), forcompare);
		if (iter != entryset[pos_index]->end())
		{
			if (iter->index == search_index)
			{
				iter->value += val;
				if (iter->value == 0)
					entryset[pos_index]->erase(iter);
			}
			else
				entryset[pos_index]->insert(iter, Sparse_Entry(search_index, val));
		}
		else
		{
			entryset[pos_index]->insert(entryset[pos_index]->end(), Sparse_Entry(search_index, val));
		}
	}
	//! set_internal_entry
	void set_internal_entry(const size_t row_index, const size_t col_index, const double val)
	{
		const size_t search_index = (storage_type == CCS ? row_index : col_index);
		const size_t pos_index = (storage_type == CCS ? col_index : row_index);

		Sparse_Entry forcompare(search_index);
		Sparse_Vector::iterator iter =
			std::lower_bound(entryset[pos_index]->begin(), entryset[pos_index]->end(), forcompare);
		if (iter != entryset[pos_index]->end())
		{
			if (iter->index == search_index)
			{
				iter->value = val;
				if (iter->value == 0)
					entryset[pos_index]->erase(iter);
			}
			else if ( val != 0)
				entryset[pos_index]->insert(iter, Sparse_Entry(search_index, val));
		}
		else if ( val != 0)
		{
			entryset[pos_index]->insert(entryset[pos_index]->end(), Sparse_Entry(search_index, val));
		}
	}
	//////////////////////////////////////////////////////////////////////////
};

//! print sparse matrix
std::ostream & operator<<(std::ostream & output_stream, const Sparse_Matrix * mat);
std::ostream & operator<<(std::ostream & output_stream, const Sparse_Matrix & mat);
//@}


//! multiplication \f$ Y = A X \f$
/*!
*	\param A sparse matrix
*	\param X input column vector
*	\param Y output column vector
*/
void multiply(const Sparse_Matrix *A, double *X, double *Y); 
//! multiplication \f$ Y = A^T X \f$
/*!
*	\param A sparse matrix
*	\param X input column vector
*	\param Y output column vector
*/
void transpose_multiply(const Sparse_Matrix *A, double *X, double *Y); 

//! multiplication \f$ Y = (A^T A) X \f$
/*!
*	\param A sparse matrix
*	\param X input column vector
*	\param Y output column vector
*/
void transpose_self_multiply(const Sparse_Matrix *A, double *X, double *Y, double *tmp_vec); 
//! multiplication \f$ Y = (A A^T) X \f$
/*!
*	\param A sparse matrix
*	\param X input column vector
*	\param Y output column vector
*/
void self_transpose_multiply(const Sparse_Matrix *A, double *X, double *Y, double *tmp_vec); 
//! convert matrix storage
/*!
*	\param A sparse matrix
*	\param store the specified storage type
*	\param sym how to store the converted symmetric matrix. only valid, when A is symmetric
*/
Sparse_Matrix* convert_sparse_matrix_storage(const Sparse_Matrix *A, SPARSE_STORAGE_TYPE store, SPARSE_SYMMETRIC_TYPE sym);
//! matrix transposition
/*!
*	\param A sparse matrix
*	\param store the specified storage type
*	\param sym how to store the converted symmetric matrix. only valid, when A is symmetric
*/
Sparse_Matrix* transpose(Sparse_Matrix* A, SPARSE_STORAGE_TYPE store, SPARSE_SYMMETRIC_TYPE sym);
//! compute \f$ A^T A \f$
/*!
*	\param A sparse matrix
*	\param positive indicate whether the result is symmetric definite positive.
*	\param store the specified storage type
*	\param sym how to store the converted symmetric matrix. only valid, when A is symmetric
*	\param apply_to_rhs  perform A^T B
*/
Sparse_Matrix* TransposeTimesSelf(Sparse_Matrix* mat, SPARSE_STORAGE_TYPE store, SPARSE_SYMMETRIC_TYPE sym, bool apply_to_rhs);

//! compute \f$ A^T A \f$
Sparse_Matrix* CCS_TransposeTimesSelf(Sparse_Matrix* mat, SPARSE_SYMMETRIC_TYPE sym, bool apply_to_rhs);

//! utils: remove zero elements from a sparse vector
void remove_zero_element(Sparse_Vector& vec, double tiny_value);

//! utils: save sparse pattern to a BMP file
void save_sparse_pattern(const char filename[], const Sparse_Matrix* mat, unsigned int resolution = 400);

double DotProduct(const Sparse_Vector& v0, const Sparse_Vector& v1);

//////////////////////////////////////////////////////////////////////////
template<typename MYINT>
class Sorted_Vector
{
public:
	Sorted_Vector(){}
	Sorted_Vector(const size_t size){data.reserve(size);}
	Sorted_Vector(const Sorted_Vector<MYINT>& vec) {data.assign(vec.data.begin(), vec.data.end());}
	void allocate(const size_t size) {data.reserve(size);}
	void insert(const MYINT& e) 
	{
		std::vector<MYINT>::iterator iter = std::lower_bound(data.begin(), data.end(), e);

		if (iter != data.end())
		{
			if (*iter != e)
				data.insert(iter, e);
		}
		else
		{
			data.push_back(e);
		}
	}
	void clear() {data.resize(0);}
public:
	std::vector<MYINT> data;
};

#endif //Sparse_Matrix_H
