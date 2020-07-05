#ifndef SPARSE_ELIMINATION_H
#define SPARSE_ELIMINATION_H

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ostream>
#include <set>
#include <climits>
#include <cassert>

template <typename INTEX_TYPE, typename VALUE_TYPE>
class Compress_Entry
{
public:
	INTEX_TYPE index;
	VALUE_TYPE value;
public:
	//////////////////////////////////////////////////////////////////////////
	Compress_Entry() :index(0),value(0) {}
	Compress_Entry(const INTEX_TYPE ind, const VALUE_TYPE v = 0)
		:index(ind), value(v)
	{}
	//////////////////////////////////////////////////////////////////////////
	inline bool operator< (const Compress_Entry<INTEX_TYPE, VALUE_TYPE>& m_r) const
	{
		return index < m_r.index;
	}
	//////////////////////////////////////////////////////////////////////////
};
template <typename INTEX_TYPE, typename VALUE_TYPE>
class Coordinate_Entry
{
public:
	INTEX_TYPE row_index, col_index;
	VALUE_TYPE value;
public:
	//////////////////////////////////////////////////////////////////////////
	Coordinate_Entry() :row_index(0), col_index(0), value(0) {}
	Coordinate_Entry(const INTEX_TYPE row_ind, const INTEX_TYPE col_ind, const VALUE_TYPE val = 0)
		:row_index(row_ind), col_index(col_ind), value(val)
	{}
};

//////////////////////////////////////////////////////////////////////////
typedef size_t MY_INDEX_TYPE;
typedef ptrdiff_t MY_VALUE_TYPE;
typedef Compress_Entry<MY_INDEX_TYPE, MY_VALUE_TYPE> Integer_Entry;
typedef Coordinate_Entry<MY_INDEX_TYPE, MY_VALUE_TYPE> Integer_Coordinate_Entry;

typedef std::vector<Integer_Entry> Integer_Vector;
typedef std::vector<Integer_Vector*> Integer_Vector_Store;
typedef std::vector<Integer_Coordinate_Entry> Coordinate_Vector;

//////////////////////////////////////////////////////////////////////////
class RowTriple
{
public:
	size_t row_id, length, time_step;
	//////////////////////////////////////////////////////////////////////////
public:
	RowTriple():row_id(0),length(0),time_step(0) {}
	RowTriple(const size_t ind, const size_t len, const size_t t)
		:row_id(ind), length(len), time_step(t)
	{}
};

bool operator< (const RowTriple& m_l, const RowTriple& m_r);
//////////////////////////////////////////////////////////////////////////
class Integer_Matrix
{
private:
	Integer_Vector_Store row_matrix, col_matrix;
	bool dynamic_increase_matrix_size;
	//////////////////////////////////////////////////////////////////////////
public:
	Integer_Matrix(const size_t n_row = 0, const size_t n_col = 0)
	{
		row_matrix.resize(n_row);
		col_matrix.resize(n_col);
		for (size_t i = 0; i < n_row; i++)
		{
			row_matrix[i] = new Integer_Vector;
		}
		for (size_t j = 0; j < n_col; j++)
		{
			col_matrix[j] = new Integer_Vector;
		}
		dynamic_increase_matrix_size = false;
	}
	//////////////////////////////////////////////////////////////////////////
	~Integer_Matrix()
	{
		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			delete row_matrix[i];
		}
		for (size_t j = 0; j < col_matrix.size(); j++)
		{
			delete col_matrix[j];
		}
	}
	//////////////////////////////////////////////////////////////////////////
	Integer_Matrix(const Integer_Matrix& mat)
	{
		copy(mat);
	}
	//////////////////////////////////////////////////////////////////////////
	Integer_Matrix& operator = (const Integer_Matrix& mat)
	{
		copy(mat);
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	void copy(const Integer_Matrix& mat)
	{
		dynamic_increase_matrix_size = mat.get_dynamic_growing();

		const Integer_Vector_Store& mat_rows = mat.get_row_matrix();
		const Integer_Vector_Store& mat_cols = mat.get_col_matrix();
		if (mat.nrows() >= row_matrix.size())
		{
			const size_t cur_len = row_matrix.size();
			row_matrix.resize(mat.nrows());
			for (size_t i = cur_len; i < row_matrix.size(); i++)
			{
				row_matrix[i] = new Integer_Vector;
			}
		}
		else
		{
			for (size_t i = 0; i < mat.nrows(); i++)
			{
				row_matrix[i]->resize(0);
			}
			for (size_t i = mat.nrows(); i < row_matrix.size(); i++)
			{
				delete row_matrix[i];
			}
			row_matrix.resize(mat.nrows());
		}
		for (size_t i = 0; i < mat_rows.size(); i++)
		{
			row_matrix[i]->resize(mat_rows[i]->size());
			row_matrix[i]->assign(mat_rows[i]->begin(), mat_rows[i]->end());
		}

		if (mat.ncols() >= col_matrix.size())
		{
			const size_t cur_len = col_matrix.size();
			col_matrix.resize(mat.ncols());
			for (size_t i = cur_len; i < col_matrix.size(); i++)
			{
				col_matrix[i] = new Integer_Vector;
			}
		}
		else
		{
			for (size_t i = 0; i < mat.ncols(); i++)
			{
				col_matrix[i]->resize(0);
			}
			for (size_t i = mat.ncols(); i < col_matrix.size(); i++)
			{
				delete col_matrix[i];
			}
			col_matrix.resize(mat.ncols());
		}

		for (size_t i = 0; i < mat_cols.size(); i++)
		{
			col_matrix[i]->resize(mat_cols[i]->size());
			col_matrix[i]->assign(mat_cols[i]->begin(), mat_cols[i]->end());
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void clear()
	{
		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			row_matrix[i]->resize(0);
		}
		for (size_t j = 0; j < col_matrix.size(); j++)
		{
			col_matrix[j]->resize(0);
		}
	}
	//////////////////////////////////////////////////////////////////////////
	inline const Integer_Vector_Store& get_row_matrix() const {return row_matrix;}
	inline const Integer_Vector_Store& get_col_matrix() const {return col_matrix;}
	inline const size_t nrows() const {return row_matrix.size();}
	inline const size_t ncols() const {return col_matrix.size();}
	void set_dynamic_growing(const bool dynamic) {dynamic_increase_matrix_size = dynamic;}
	bool get_dynamic_growing() const {return dynamic_increase_matrix_size; }
	//////////////////////////////////////////////////////////////////////////
	void append_rows(const size_t num_rows)
	{
		for (size_t i = 0; i < num_rows; i++)
			row_matrix.push_back(new Integer_Vector);
	}
	//////////////////////////////////////////////////////////////////////////
	void append_cols(const size_t num_cols)
	{
		for (size_t i = 0; i < num_cols; i++)
			col_matrix.push_back(new Integer_Vector);
	}
	//////////////////////////////////////////////////////////////////////////
	void delete_row(const size_t cur_row_id)
	{
		if (cur_row_id >= row_matrix.size())
			return;
		clear_row(cur_row_id);
		delete row_matrix[cur_row_id];
		row_matrix.erase(row_matrix.begin()+cur_row_id);
	}
	//////////////////////////////////////////////////////////////////////////
	void delete_col(const size_t cur_col_id)
	{
		if (cur_col_id >= col_matrix.size())
			return;
		delete col_matrix[cur_col_id];
		col_matrix.erase(col_matrix.begin()+cur_col_id);
	}
	//////////////////////////////////////////////////////////////////////////
	void clear_row(const size_t cur_row_id)
	{
		if (cur_row_id >= row_matrix.size())
			return;

		for (size_t i = 0; i < col_matrix.size(); i++)
		{
			Integer_Entry forcompare(cur_row_id);
			Integer_Vector::iterator iter =
				std::lower_bound(col_matrix[i]->begin(), col_matrix[i]->end(), forcompare);

			if (iter != col_matrix[i]->end() && iter->index == cur_row_id)
			{
				col_matrix[i]->erase(iter);
			}
		}
		row_matrix[cur_row_id]->resize(0);
	}
	//////////////////////////////////////////////////////////////////////////
	void clear_col(const size_t cur_col_id)
	{
		if (cur_col_id >= col_matrix.size())
			return;

		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			Integer_Entry forcompare(cur_col_id);
			Integer_Vector::iterator iter =
				std::lower_bound(row_matrix[i]->begin(), row_matrix[i]->end(), forcompare);

			if (iter != row_matrix[i]->end() && iter->index == cur_col_id)
			{
				row_matrix[i]->erase(iter);
			}
		}
		col_matrix[cur_col_id]->resize(0);
	}
	//////////////////////////////////////////////////////////////////////////
	void fill_entry(const size_t row_index, const size_t col_index, const MY_VALUE_TYPE val)
	{
		if (val == 0)
			return;

		if ((row_index >= row_matrix.size() || col_index >= col_matrix.size()))
		{
			if (dynamic_increase_matrix_size)
			{
				if (row_index >= row_matrix.size())
					append_rows(row_index + 1 - row_matrix.size());
				if (col_index >= col_matrix.size())
					append_cols(col_index + 1 - col_matrix.size());
			}
			else
				return;
		}

		fill_internal_entry(row_matrix[row_index], col_index, val);
		fill_internal_entry(col_matrix[col_index], row_index, val);
	}
	//////////////////////////////////////////////////////////////////////////
	void set_entry(const size_t row_index, const size_t col_index, const MY_VALUE_TYPE val)
	{
		if ((row_index >= row_matrix.size() || col_index >= col_matrix.size()))
		{
			if (dynamic_increase_matrix_size && val != 0)
			{
				if (row_index >= row_matrix.size())
					append_rows(row_index + 1 - row_matrix.size());
				if (col_index >= col_matrix.size())
					append_cols(col_index + 1 - col_matrix.size());
			}
			else
				return;
		}

		set_internal_entry(row_matrix[row_index], col_index, val);
		set_internal_entry(col_matrix[col_index], row_index, val);
	}
	//////////////////////////////////////////////////////////////////////////
	const size_t GaussianEliminate(Coordinate_Vector& elim_vars, bool has_constant_column = false)
	{
		const size_t num_rows = row_matrix.size();
		elim_vars.clear();
		if (num_rows < 1)
			return 0;

		std::vector<bool> row_tag(num_rows, false), col_tag(col_matrix.size(), false);
		std::set<RowTriple> rowqueue;

		if (has_constant_column)
		{
			col_tag.back() = true;
		}

		size_t time_step = 0;
		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			if (!row_matrix[i]->empty())
				rowqueue.insert(RowTriple(i, row_matrix[i]->size(), time_step));
		}

		time_step++;

		Integer_Vector copy_Vec;

		while (!rowqueue.empty())
		{
			const size_t cur_row_id = rowqueue.begin()->row_id;
			rowqueue.erase(rowqueue.begin());

			if (row_tag[cur_row_id])
				continue;

			row_tag[cur_row_id] = true;

			MY_VALUE_TYPE cur_col_id = -1;
			size_t col_size = (std::numeric_limits<size_t>::max)();
			//find a column with the smallest length
			MY_VALUE_TYPE cur_value = 0;
			for (Integer_Vector::const_iterator riter = row_matrix[cur_row_id]->begin();
				riter != row_matrix[cur_row_id]->end(); riter++)
			{
				if (col_tag[riter->index] == false && col_matrix[riter->index]->size() < col_size)
				{
					col_size = col_matrix[riter->index]->size();
					cur_col_id = riter->index;
					cur_value = riter->value;
					if (col_size == 1)
						break;
				}
			}
			if (cur_col_id == -1)
				continue;

			assert(cur_value != 0); // if cur_value = 0,  it means that the intermediate integer is too large and it exceeds the length of MY_VALUE_TYPE. 

			col_tag[cur_col_id] = true;

			copy_Vec.resize(col_matrix[cur_col_id]->size());
			copy_Vec.assign(col_matrix[cur_col_id]->begin(), col_matrix[cur_col_id]->end());

			elim_vars.push_back(Integer_Coordinate_Entry(cur_row_id, cur_col_id)); 

			for (Integer_Vector::const_reverse_iterator riter = copy_Vec.rbegin(); riter != copy_Vec.rend(); riter++)
			{
				if (riter->index == cur_row_id)
					continue;

				const MY_VALUE_TYPE gcd_val = GCD(std::abs(cur_value), std::abs(riter->value));
				const MY_VALUE_TYPE factor_head = -riter->value / gcd_val;
				const MY_VALUE_TYPE factor_cur_row = cur_value / gcd_val;

				scale_vector(row_matrix[riter->index], factor_cur_row);
				for (Integer_Vector::const_iterator miter = row_matrix[riter->index]->begin(); miter != row_matrix[riter->index]->end(); miter++)
				{
					scale_internal_entry(col_matrix[miter->index], riter->index, factor_cur_row);
				}
				for (Integer_Vector::const_reverse_iterator miter = row_matrix[cur_row_id]->rbegin(); miter != row_matrix[cur_row_id]->rend(); miter++)
				{
					fill_entry(riter->index, miter->index, miter->value*factor_head);
				}
			}

			for (Integer_Vector::const_iterator iter = copy_Vec.begin(); iter != copy_Vec.end(); iter++)
			{
				normalize_row(iter->index);
			}

			bool inc_time = false;
			for (Integer_Vector::const_iterator iter = copy_Vec.begin(); iter != copy_Vec.end(); iter++)
			{
				if (iter->index == cur_row_id)
					continue;
				else
				{
					rowqueue.insert(RowTriple(iter->index, row_matrix[iter->index]->size(), time_step));
					inc_time = true;
				}
			}
			if (inc_time)
				time_step++;
		}

		for (Coordinate_Vector::iterator iter = elim_vars.begin(); iter != elim_vars.end(); iter++)
		{
			if (row_matrix[iter->row_index]->size() <= col_matrix[iter->col_index]->size())
			{
				iter->value = std::lower_bound(row_matrix[iter->row_index]->begin(),row_matrix[iter->row_index]->end(), iter->col_index)->value;
			}
			else
			{
				iter->value = std::lower_bound(col_matrix[iter->col_index]->begin(),col_matrix[iter->col_index]->end(), iter->row_index)->value;
			}
		}

		return elim_vars.size();
	}
	//////////////////////////////////////////////////////////////////////////
	const size_t number_of_nonzeros()
	{
		size_t nnz = 0;
		for (Integer_Vector_Store::const_iterator iter = row_matrix.begin(); iter != row_matrix.end(); iter++)
		{
			nnz += (*iter)->size();
		}
		return nnz;
	}
	//////////////////////////////////////////////////////////////////////////
	void substitute_solution(const Coordinate_Vector& elim_vars, const double* partial_solution, double* all_solution)
	{
		std::vector<bool> is_variables(ncols(), true);
		for (Coordinate_Vector::const_iterator iter = elim_vars.begin(); iter != elim_vars.end(); iter++)
		{
			is_variables[iter->col_index] = false;
		}
		size_t count = 0;
		for (size_t i = 0; i < ncols(); i++)
		{
			if (!is_variables[i])
			{
				all_solution[i] = partial_solution[count];
				count++;
			}
		}
		for (Coordinate_Vector::const_iterator iter = elim_vars.begin(); iter != elim_vars.end(); iter++)
		{
			const size_t& row_id = iter->row_index;
			all_solution[iter->col_index] = 0;
			double denom = 1;
			for (Integer_Vector::const_iterator viter = row_matrix[row_id]->begin(); viter != row_matrix[row_id]->end(); viter++)
			{
				if (viter->index != iter->col_index)
				{
					all_solution[iter->col_index] += viter->value * all_solution[viter->index];
				}
				else
				{
					denom = -(double)viter->value;
				}
			}
			all_solution[iter->col_index] /= denom;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void random_solution(std::vector<double>& solution)
	{
		Coordinate_Vector elim_vars;
		GaussianEliminate(elim_vars);
		solution.resize(ncols(), 0);
		std::vector<bool> independent(ncols(), true);
		for (size_t k = 0; k < elim_vars.size(); k++ )
		{
			independent[elim_vars[k].col_index] = false;
		}
		for (size_t k = 0; k < solution.size(); k++)
		{
			if (independent[k])
			{
				solution[k] = (double) (rand() % (100*solution.size()));
			}
		}

		for (size_t i = 0; i < elim_vars.size(); i++)
		{
			const size_t var = elim_vars[i].col_index;
			//retrieve the value from linear constraints
			solution[var] = 0;
			Integer_Vector::iterator miter = row_matrix[elim_vars[i].row_index]->begin();
			for (;miter !=  row_matrix[elim_vars[i].row_index]->end(); miter++)
			{
				if (miter->index != var)
				{
					solution[var] -= solution[miter->index] * miter->value;
				}
			}
			solution[var] /= elim_vars[i].value;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	bool eliminate_known_variable(const size_t which_col, const ptrdiff_t known_var)
	{
		bool valid = true;
		size_t last_column = col_matrix.size() - 1;
		assert(which_col < last_column);
		Integer_Vector copy_Vec(col_matrix[which_col]->size());
		copy_Vec.assign(col_matrix[which_col]->begin(), col_matrix[which_col]->end());
		for (Integer_Vector::const_reverse_iterator riter = copy_Vec.rbegin(); riter != copy_Vec.rend(); riter++)
		{
			set_entry(riter->index, which_col, 0);
			fill_entry(riter->index, last_column, riter->value * known_var);
			if (row_matrix[riter->index]->size() == 1)
			{
				valid = false;
			}
		}
		return valid;
	}
	//////////////////////////////////////////////////////////////////////////
	bool check_consistent(const size_t which_col, const ptrdiff_t known_var)
	{
		size_t last_column = col_matrix.size() - 1;
		assert(which_col < last_column);

		for (Integer_Vector::const_reverse_iterator riter = col_matrix[which_col]->rbegin(); riter != col_matrix[which_col]->rend(); riter++)
		{
			if(row_matrix[riter->index]->size() == 1)
			{
				if (known_var != 0)
				{
					return false;
				}
			}
			else if (row_matrix[riter->index]->size() == 2)
			{
				const Integer_Entry& back = row_matrix[riter->index]->back();
				if (back.index == last_column)
				{
					const Integer_Entry& front = row_matrix[riter->index]->front();
					if (front.value * known_var  + back.value != 0)
					{
						return false;
					}
				}
			}
		}
		return true;
	}
	//////////////////////////////////////////////////////////////////////////
private:
	void fill_internal_entry(Integer_Vector* vec, const size_t id, const MY_VALUE_TYPE val)
	{
		Integer_Entry forcompare(id);
		Integer_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end())
		{
			if (iter->index == id)
			{
				iter->value += val;
				if (iter->value == 0)
					vec->erase(iter);
			}
			else if (val != 0)
				vec->insert(iter, Integer_Entry(id, val));
		}
		else if (val != 0)
		{
			vec->push_back(Integer_Entry(id, val));
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void set_internal_entry(Integer_Vector* vec, const size_t id, const MY_VALUE_TYPE val)
	{
		Integer_Entry forcompare(id);
		Integer_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end())
		{
			if (iter->index == id)
			{
				iter->value = val;
				if (iter->value == 0)
					vec->erase(iter);
			}
			else if (val != 0)
				vec->insert(iter, Integer_Entry(id, val));
		}
		else if (val != 0)
		{
			vec->push_back(Integer_Entry(id, val));
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void scale_internal_entry(Integer_Vector* vec, const size_t id, const MY_VALUE_TYPE scale)
	{
		if (scale == 1)
			return;
		Integer_Entry forcompare(id);
		Integer_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end() && iter->index == id)
		{
			if (scale == 0)
				vec->erase(iter);
			else
				iter->value *= scale;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void fill_internal_entry_with_scale(Integer_Vector* vec, const size_t id, const MY_VALUE_TYPE scale, const MY_VALUE_TYPE val)
	{
		Integer_Entry forcompare(id);
		Integer_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end())
		{
			if (iter->index == id)
			{
				iter->value = iter->value * scale + val;
				if (iter->value == 0)
					vec->erase(iter);
			}
			else if (val != 0)
				vec->insert(iter, Integer_Entry(id, val));
		}
		else if (val != 0)
		{
			vec->push_back(Integer_Entry(id, val));
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void scale_vector(Integer_Vector* vec, const MY_VALUE_TYPE scale)
	{
		if (scale == 1)
			return;
		if (scale == 0)
		{
			vec->resize(0);
			return;
		}
		for (Integer_Vector::iterator iter = vec->begin(); iter != vec->end(); iter++)
		{
			iter->value *= scale;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	const ptrdiff_t normalize_row(const size_t row_id)
	{

		if (row_matrix[row_id]->empty())
			return 1;
		Integer_Vector::iterator iter = row_matrix[row_id]->begin();
		size_t gcd_value = std::abs(iter->value);
		if (gcd_value == 1)
			return 1;

		for (; iter != row_matrix[row_id]->end(); iter++)
		{
			gcd_value = GCD(gcd_value, std::abs(iter->value));
			if (gcd_value == 1)
				return 1;
		}

		for (iter = row_matrix[row_id]->begin(); iter != row_matrix[row_id]->end(); iter++)
		{
			iter->value /= (MY_VALUE_TYPE)gcd_value;
			set_internal_entry(col_matrix[iter->index], row_id, iter->value);
		}

		return gcd_value;
	}
	//////////////////////////////////////////////////////////////////////////
	size_t GCD(size_t a, size_t b)
	{
		bool done = true;
		while( done )
		{
			a = a % b;
			if( a == 0 )
				return b;
			b = b % a;
			if( b == 0 )
				return a;
		}
		return 1;
	}
	//////////////////////////////////////////////////////////////////////////
};

std::ostream& operator<<(std::ostream &s, const Integer_Matrix &A);
//////////////////////////////////////////////////////////////////////////
typedef double MY_REAL_TYPE;
typedef Compress_Entry<MY_INDEX_TYPE, MY_REAL_TYPE> Real_Entry;

typedef std::vector<Real_Entry> Real_Vector;
typedef std::vector<Real_Vector*> Real_Vector_Store;


class Real_Matrix
{
private:
	Real_Vector_Store row_matrix, col_matrix;
	bool dynamic_increase_matrix_size;
	//////////////////////////////////////////////////////////////////////////
public:
	Real_Matrix(const size_t n_row = 0, const size_t n_col = 0)
	{
		row_matrix.resize(n_row);
		col_matrix.resize(n_col);
		for (size_t i = 0; i < n_row; i++)
		{
			row_matrix[i] = new Real_Vector;
		}
		for (size_t j = 0; j < n_col; j++)
		{
			col_matrix[j] = new Real_Vector;
		}
		dynamic_increase_matrix_size = false;
	}
	//////////////////////////////////////////////////////////////////////////
	~Real_Matrix()
	{
		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			delete row_matrix[i];
		}
		for (size_t j = 0; j < col_matrix.size(); j++)
		{
			delete col_matrix[j];
		}
	}
	//////////////////////////////////////////////////////////////////////////
	Real_Matrix(const Real_Matrix& mat)
	{
		copy(mat);
	}
	//////////////////////////////////////////////////////////////////////////
	Real_Matrix& operator = (const Real_Matrix& mat)
	{
		copy(mat);
		return *this;
	}
	//////////////////////////////////////////////////////////////////////////
	void copy(const Real_Matrix& mat)
	{
		dynamic_increase_matrix_size = mat.get_dynamic_growing();

		const Real_Vector_Store& mat_rows = mat.get_row_matrix();
		const Real_Vector_Store& mat_cols = mat.get_col_matrix();
		if (mat.nrows() >= row_matrix.size())
		{
			const size_t cur_len = row_matrix.size();
			row_matrix.resize(mat.nrows());
			for (size_t i = cur_len; i < row_matrix.size(); i++)
			{
				row_matrix[i] = new Real_Vector;
			}
		}
		else
		{
			for (size_t i = 0; i < mat.nrows(); i++)
			{
				row_matrix[i]->resize(0);
			}
			for (size_t i = mat.nrows(); i < row_matrix.size(); i++)
			{
				delete row_matrix[i];
			}
			row_matrix.resize(mat.nrows());
		}
		for (size_t i = 0; i < mat_rows.size(); i++)
		{
			row_matrix[i]->resize(mat_rows[i]->size());
			row_matrix[i]->assign(mat_rows[i]->begin(), mat_rows[i]->end());
		}

		if (mat.ncols() >= col_matrix.size())
		{
			const size_t cur_len = col_matrix.size();
			col_matrix.resize(mat.ncols());
			for (size_t i = cur_len; i < col_matrix.size(); i++)
			{
				col_matrix[i] = new Real_Vector;
			}
		}
		else
		{
			for (size_t i = 0; i < mat.ncols(); i++)
			{
				col_matrix[i]->resize(0);
			}
			for (size_t i = mat.ncols(); i < col_matrix.size(); i++)
			{
				delete col_matrix[i];
			}
			col_matrix.resize(mat.ncols());
		}

		for (size_t i = 0; i < mat_cols.size(); i++)
		{
			col_matrix[i]->resize(mat_cols[i]->size());
			col_matrix[i]->assign(mat_cols[i]->begin(), mat_cols[i]->end());
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void clear()
	{
		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			row_matrix[i]->resize(0);
		}
		for (size_t j = 0; j < col_matrix.size(); j++)
		{
			col_matrix[j]->resize(0);
		}
	}
	//////////////////////////////////////////////////////////////////////////
	inline const Real_Vector_Store& get_row_matrix() const {return row_matrix;}
	inline const Real_Vector_Store& get_col_matrix() const {return col_matrix;}
	inline const size_t nrows() const {return row_matrix.size();}
	inline const size_t ncols() const {return col_matrix.size();}
	void set_dynamic_growing(const bool dynamic) {dynamic_increase_matrix_size = dynamic;}
	bool get_dynamic_growing() const {return dynamic_increase_matrix_size; }
	//////////////////////////////////////////////////////////////////////////
	void append_rows(const size_t num_rows)
	{
		for (size_t i = 0; i < num_rows; i++)
			row_matrix.push_back(new Real_Vector);
	}
	//////////////////////////////////////////////////////////////////////////
	void append_cols(const size_t num_cols)
	{
		for (size_t i = 0; i < num_cols; i++)
			col_matrix.push_back(new Real_Vector);
	}
	//////////////////////////////////////////////////////////////////////////
	void delete_row(const size_t cur_row_id)
	{
		if (cur_row_id >= row_matrix.size())
			return;
		clear_row(cur_row_id);
		delete row_matrix[cur_row_id];
		row_matrix.erase(row_matrix.begin()+cur_row_id);
	}
	//////////////////////////////////////////////////////////////////////////
	void delete_col(const size_t cur_col_id)
	{
		if (cur_col_id >= col_matrix.size())
			return;
		delete col_matrix[cur_col_id];
		col_matrix.erase(col_matrix.begin()+cur_col_id);
	}
	//////////////////////////////////////////////////////////////////////////
	void clear_row(const size_t cur_row_id)
	{
		if (cur_row_id >= row_matrix.size())
			return;

		for (size_t i = 0; i < col_matrix.size(); i++)
		{
			Real_Entry forcompare(cur_row_id);
			Real_Vector::iterator iter =
				std::lower_bound(col_matrix[i]->begin(), col_matrix[i]->end(), forcompare);

			if (iter != col_matrix[i]->end() && iter->index == cur_row_id)
			{
				col_matrix[i]->erase(iter);
			}
		}
		row_matrix[cur_row_id]->resize(0);
	}
	//////////////////////////////////////////////////////////////////////////
	void clear_col(const size_t cur_col_id)
	{
		if (cur_col_id >= col_matrix.size())
			return;

		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			Real_Entry forcompare(cur_col_id);
			Real_Vector::iterator iter =
				std::lower_bound(row_matrix[i]->begin(), row_matrix[i]->end(), forcompare);

			if (iter != row_matrix[i]->end() && iter->index == cur_col_id)
			{
				row_matrix[i]->erase(iter);
			}
		}
		col_matrix[cur_col_id]->resize(0);
	}
	//////////////////////////////////////////////////////////////////////////
	void fill_entry(const size_t row_index, const size_t col_index, const MY_REAL_TYPE val)
	{
		if (val == 0)
			return;

		if ((row_index >= row_matrix.size() || col_index >= col_matrix.size()))
		{
			if (dynamic_increase_matrix_size)
			{
				if (row_index >= row_matrix.size())
					append_rows(row_index + 1 - row_matrix.size());
				if (col_index >= col_matrix.size())
					append_cols(col_index + 1 - col_matrix.size());
			}
			else
				return;
		}

		fill_internal_entry(row_matrix[row_index], col_index, val);
		fill_internal_entry(col_matrix[col_index], row_index, val);
	}
	//////////////////////////////////////////////////////////////////////////
	void set_entry(const size_t row_index, const size_t col_index, const MY_REAL_TYPE val)
	{
		if ((row_index >= row_matrix.size() || col_index >= col_matrix.size()))
		{
			if (dynamic_increase_matrix_size && val != 0)
			{
				if (row_index >= row_matrix.size())
					append_rows(row_index + 1 - row_matrix.size());
				if (col_index >= col_matrix.size())
					append_cols(col_index + 1 - col_matrix.size());
			}
			else
				return;
		}

		set_internal_entry(row_matrix[row_index], col_index, val);
		set_internal_entry(col_matrix[col_index], row_index, val);
	}
	//////////////////////////////////////////////////////////////////////////
	void GaussianEliminate(bool has_constant_column = false) //this function is not tested yet
	{
		const size_t num_rows = row_matrix.size();
		if (num_rows < 1)
			return ;

		std::vector<bool> row_tag(num_rows, false), col_tag(col_matrix.size(), false);
		std::set<RowTriple> rowqueue;

		if (has_constant_column)
		{
			col_tag.back() = true;
		}

		size_t time_step = 0;
		for (size_t i = 0; i < row_matrix.size(); i++)
		{
			if (!row_matrix[i]->empty())
				rowqueue.insert(RowTriple(i, row_matrix[i]->size(), time_step));
		}

		time_step++;

		Real_Vector copy_Vec;

		while (!rowqueue.empty())
		{
			const size_t cur_row_id = rowqueue.begin()->row_id;
			rowqueue.erase(rowqueue.begin());

			if (row_tag[cur_row_id])
				continue;

			row_tag[cur_row_id] = true;

			ptrdiff_t cur_col_id = -1;
			size_t col_size = (std::numeric_limits<size_t>::max)();
			//find a column with the smallest length
			MY_REAL_TYPE cur_value = 0;
			for (Real_Vector::const_iterator riter = row_matrix[cur_row_id]->begin();
				riter != row_matrix[cur_row_id]->end(); riter++)
			{
				if (col_tag[riter->index] == false && col_matrix[riter->index]->size() < col_size)
				{
					col_size = col_matrix[riter->index]->size();
					cur_col_id = riter->index;
					cur_value = riter->value;
					if (col_size == 1)
						break;
				}
			}
			if (cur_col_id == -1)
				continue;

			col_tag[cur_col_id] = true;

			copy_Vec.resize(col_matrix[cur_col_id]->size());
			copy_Vec.assign(col_matrix[cur_col_id]->begin(), col_matrix[cur_col_id]->end());

			for (Real_Vector::const_reverse_iterator riter = copy_Vec.rbegin(); riter != copy_Vec.rend(); riter++)
			{
				if (riter->index == cur_row_id)
					continue;

				const MY_REAL_TYPE factor =-riter->value / cur_value;

				for (Real_Vector::const_reverse_iterator miter = row_matrix[cur_row_id]->rbegin(); miter != row_matrix[cur_row_id]->rend(); miter++)
				{
					fill_entry(riter->index, miter->index, miter->value*factor);
				}
			}

			for (Real_Vector::const_iterator iter = copy_Vec.begin(); iter != copy_Vec.end(); iter++)
			{
				truncate_row(iter->index);
			}

			bool inc_time = false;
			for (Real_Vector::const_iterator iter = copy_Vec.begin(); iter != copy_Vec.end(); iter++)
			{
				if (iter->index == cur_row_id)
					continue;
				else
				{
					rowqueue.insert(RowTriple(iter->index, row_matrix[iter->index]->size(), time_step));
					inc_time = true;
				}
			}
			if (inc_time)
				time_step++;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	const size_t number_of_nonzeros()
	{
		size_t nnz = 0;
		for (Real_Vector_Store::const_iterator iter = row_matrix.begin(); iter != row_matrix.end(); iter++)
		{
			nnz += (*iter)->size();
		}
		return nnz;
	}
	//////////////////////////////////////////////////////////////////////////
private:
	void fill_internal_entry(Real_Vector* vec, const size_t id, const MY_REAL_TYPE val)
	{
		Real_Entry forcompare(id);
		Real_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end())
		{
			if (iter->index == id)
			{
				iter->value += val;
				if (iter->value == 0)
					vec->erase(iter);
			}
			else if (val != 0)
				vec->insert(iter, Real_Entry(id, val));
		}
		else if (val != 0)
		{
			vec->push_back(Real_Entry(id, val));
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void set_internal_entry(Real_Vector* vec, const size_t id, const MY_REAL_TYPE val)
	{
		Real_Entry forcompare(id);
		Real_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end())
		{
			if (iter->index == id)
			{
				iter->value = val;
				if (iter->value == 0)
					vec->erase(iter);
			}
			else if (val != 0)
				vec->insert(iter, Real_Entry(id, val));
		}
		else if (val != 0)
		{
			vec->push_back(Real_Entry(id, val));
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void scale_internal_entry(Real_Vector* vec, const size_t id, const MY_REAL_TYPE scale)
	{
		if (scale == 1)
			return;
		Real_Entry forcompare(id);
		Real_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end() && iter->index == id)
		{
			if (scale == 0)
				vec->erase(iter);
			else
				iter->value *= scale;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void fill_internal_entry_with_scale(Real_Vector* vec, const size_t id, const MY_REAL_TYPE scale, const MY_REAL_TYPE val)
	{
		Real_Entry forcompare(id);
		Real_Vector::iterator iter =
			std::lower_bound(vec->begin(), vec->end(), forcompare);

		if (iter != vec->end())
		{
			if (iter->index == id)
			{
				iter->value = iter->value * scale + val;
				if (iter->value == 0)
					vec->erase(iter);
			}
			else if (val != 0)
				vec->insert(iter, Real_Entry(id, val));
		}
		else if (val != 0)
		{
			vec->push_back(Real_Entry(id, val));
		}
	}
	//////////////////////////////////////////////////////////////////////////
	void truncate_row(const size_t row_id)
	{
		if (row_matrix[row_id]->empty())
			return;
		
		std::vector<size_t> elim_pos;

		for (Real_Vector::const_reverse_iterator riter = row_matrix[row_id]->rbegin(); riter != row_matrix[row_id]->rend(); riter++)
		{
			if (std::fabs(riter->value) < 1.0e-12)
				elim_pos.push_back(riter->index);
		}

		for (std::vector<size_t>::const_iterator iter = elim_pos.begin(); iter != elim_pos.end(); iter++)
		{
			set_entry(row_id, *iter, 0);
		}
	}
	//////////////////////////////////////////////////////////////////////////
};


std::ostream& operator<<(std::ostream &s, const Real_Matrix &A);
//////////////////////////////////////////////////////////////////////////
#endif
