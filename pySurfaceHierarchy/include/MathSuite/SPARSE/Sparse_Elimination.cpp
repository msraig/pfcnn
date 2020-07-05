#include "Sparse_Elimination.h"
//////////////////////////////////////////////////////////////////////////
bool operator< (const RowTriple& m_l, const RowTriple& m_r)
{
	if (m_l.row_id != m_r.row_id)
	{
		if (m_l.length != m_r.length)
		{
			return m_l.length < m_r.length;
		}
		else
			return m_l.row_id < m_r.row_id;
	}
	else if (m_l.row_id == m_r.row_id)
	{
		return m_l.time_step > m_r.time_step;
	}
	return false;
}
//////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream &s, const Integer_Matrix &A)
{
	const Integer_Vector_Store& entryset = A.get_row_matrix();
	s << "=============== MATRIX DATA ==================" << std::endl;
	s << A.nrows() <<" " << A.ncols() << std::endl;
	for (size_t i = 0; i < entryset.size(); i++)
	{
		if (!entryset[i]->empty())
		{
			s << "( ";
			for (Integer_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				s << "(r" << miter->index << "," << miter->value <<") ";
			}
			s << ")" << std::endl;
		}
	}
	return s;
}
//////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream &s, const Real_Matrix &A)
{
	const Real_Vector_Store& entryset = A.get_row_matrix();
	s << "=============== MATRIX DATA ==================" << std::endl;
	s << A.nrows() <<" " << A.ncols() << std::endl;
	for (size_t i = 0; i < entryset.size(); i++)
	{
		if (!entryset[i]->empty())
		{
			s << "( ";
			for (Real_Vector::const_iterator miter = entryset[i]->begin();
				miter != entryset[i]->end(); miter++)
			{
				s << "(r" << miter->index << "," << miter->value <<") ";
			}
			s << ")" << std::endl;
		}
	}
	return s;
}
//////////////////////////////////////////////////////////////////////////