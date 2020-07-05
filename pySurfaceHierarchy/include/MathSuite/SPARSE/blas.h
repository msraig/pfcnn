#ifndef _BLAS_H_
#define _BLAS_H_


#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Upper case declaration */

/* BLAS Level1 */
                              
double DCABS1(const Complex16 *z);
float SASUM(const INT *n, const float *x, const INT *incx);
void  SAXPY(const INT *n, const float *alpha, const float *x, const INT *incx,
            float *y, const INT *incy);
void  SAXPYI(const INT *nz, const float *a, const float *x, const INT *indx,float *y);
float SCASUM(const INT *n, const Complex8 *x, const INT *incx); 
float SCNRM2(const INT *n, const Complex8 *x, const INT *incx); 
void  SCOPY(const INT *n, const float *x, const INT *incx, float *y, const INT *incy);
float SDOT(const INT *n, const float *x, const INT *incx,
           const float *y, const INT *incy);
double DSDOT(const INT *n, const float *x, const INT *incx, 
             const float *y, const INT *incy);
float SDOTI(const INT *nz, const float *x, const INT *indx, const float *y);
void  SGTHR(const INT *nz, const float *y, float *x, const INT *indx);
void  SGTHRZ(const INT *nz, float *y, float *x, const INT *indx);
float SNRM2(const INT *n, const float *x, const INT *incx);
void  SROT(const INT *n, float *x, const INT *incx, float *y, const INT *incy,
           const float *c, const float *s);
void  SROTG(float *a,float *b,float *c,float *s);
void  SROTI(const INT *nz, float *x, const INT *indx, float *y, const float *c, const float *s);
void  SROTM(const INT *n, float *x, const INT *incx, float *y, const INT *incy, const float *param);
void  SROTMG(float *d1, float *d2, float *x1, const float *y1, float *param);
void  SSCAL(const INT *n, const float *a, float *x, const INT *incx);
void  SSCTR(const INT *nz, const float *x, const INT *indx, float *y);
void  SSWAP(const INT *n, float *x, const INT *incx, float *y, const INT *incy);
INT   ISAMAX(const INT *n, const float *x, const INT *incx);
INT   ISAMIN(const INT *n, const float *x, const INT *incx);

void CAXPY(const INT *n, const Complex8 *alpha, const Complex8 *x, const INT *incx,
           Complex8 *y, const INT *incy); 
void CAXPYI(const INT *nz, const Complex8 *a, 
            const Complex8 *x, const INT *indx, Complex8 *y); 
void CCOPY(const INT *n, const Complex8 *x, const INT *incx,
           Complex8 *y, const INT *incy); 
void CDOTC(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const  Complex8 *y, const INT *incy); 
void CDOTCI(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void CDOTU(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const  Complex8 *y, const INT *incy); 
void CDOTUI(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void CGTHR(const INT *nz, const Complex8 *y, Complex8 *x, const INT *indx); 
void CGTHRZ(const INT *nz, Complex8 *y, Complex8 *x, const INT *indx); 
void CROTG(Complex8 *a, const Complex8 *b, float *c, Complex8 *s); 
void CSCAL(const INT *n, const Complex8 *a, Complex8 *x, const INT *incx); 
void CSCTR(const INT *nz, const Complex8 *x, const INT *indx, Complex8 *y); 
void CSROT(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy,
           const float *c, const float *s); 
void CSSCAL(const INT *n, const float *a, Complex8 *x, const INT *incx); 
void CSWAP(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy); 
INT  ICAMAX(const INT *n, const Complex8 *x, const INT *incx); 
INT  ICAMIN(const INT *n, const Complex8 *x, const INT *incx); 

double DASUM(const INT *n, const double *x, const INT *incx);
void   DAXPY(const INT *n, const double *alpha, const double *x, const INT *incx,
             double *y, const INT *incy);
void   DAXPYI(const INT *nz, const double *a, const double *x, const INT *indx, double *y);
void   DCOPY(const INT *n, const double *x, const INT *incx, double *y, const INT *incy);
double DDOT(const  INT *n, const double *x, const INT *incx, 
            const double *y, const INT *incy);
float  SDSDOT(const INT *n, const double *x, const INT *incx, 
              const double *y, const INT *incy);
double DDOTI(const INT *nz, const double *x, const INT *indx, const double *y);
void   DGTHR(const INT *nz, const double *y, double *x, const INT *indx);
void   DGTHRZ(const INT *nz, double *y, double *x, const INT *indx);
double DNRM2(const INT *n, const double *x, const INT *incx);
void   DROT(const INT *n, double *x, const INT *incx, double *y, const INT *incy,
            const double *c, const double *s);
void   DROTG(double *a,double *b,double *c,double *s);
void   DROTI(const INT *nz, double *x, const INT *indx, double *y, const double *c, const double *s);
void   DROTM(const INT *n, double *x, const INT *incx, double *y, const INT *incy, const double *param);
void   DROTMG(double *d1, double *d2, double *x1, const double *y1, double *param);
void   DSCAL(const INT *n, const double *a, double *x, const INT *incx);
void   DSCTR(const INT *nz, const double *x, const INT *indx, double *y);
void   DSWAP(const INT *n, double *x, const INT *incx, double *y, const INT *incy);
double DZASUM(const INT *n, const Complex16 *x, const INT *incx); 
double DZNRM2(const INT *n, const Complex16 *x, const INT *incx); 
INT    IDAMAX(const INT *n, const double *x, const INT *incx);
INT    IDAMIN(const INT *n, const double *x, const INT *incx);

void ZAXPY(const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void ZAXPYI(const INT *nz, const Complex16 *a,
            const Complex16 *x, const INT *indx, Complex16 *y); 
void ZCOPY(const INT *n, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void ZDOTC(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const  Complex16 *y, const INT *incy); 
void ZDOTCI(Complex16 *pres,const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void ZDOTU(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy); 
void ZDOTUI(Complex16 *pres, const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void ZDROT(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy,
           const double *c, const double *s); 
void ZDSCAL(const INT *n, const double *a, Complex16 *x, const INT *incx); 
void ZGTHR(const INT *nz, const Complex16 *y, Complex16 *x, const INT *indx); 
void ZGTHRZ(const INT *nz, Complex16 *y, Complex16 *x, const INT *indx); 
void ZROTG(Complex16 *a, const Complex16 *b, double *c, Complex16 *s); 
void ZSCAL(const INT *n, const Complex16 *a, Complex16 *x, const INT *incx); 
void ZSCTR(const INT *nz, const Complex16 *x, const INT *indx, Complex16 *y); 
void ZSWAP(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy); 
INT  IZAMAX(const INT *n, const Complex16 *x, const INT *incx); 
INT  IZAMIN(const  INT *n,const  Complex16 *x, const INT *incx); 

/* BLAS Level2 */

void SGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const float *alpha, const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void SGEMV(const char *trans, const INT *m, const INT *n, const float *alpha,
           const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void SGER(const INT *m, const INT *n, const float *alpha, const float *x, const INT *incx,
          const float *y, const INT *incy, float *a, const INT *lda);
void SSBMV(const char *uplo, const INT *n, const INT *k, 
           const float *alpha, const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void SSPMV(const char *uplo, const INT *n, const float *alpha, const float *ap,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void SSPR(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx, float *ap);
void SSPR2(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *ap);
void SSYMV(const char *uplo, const INT *n, const float *alpha, const float *a, const INT *lda,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void SSYR(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
          float *a, const INT *lda);
void SSYR2(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *a, const INT *lda);
void STBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void STBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void STPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *ap, float *x, const INT *incx);
void STPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *ap, float *x, const INT *incx);
void STRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const float *a, const INT *lda, float *b, const INT *incx);
void STRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *a, const INT *lda, float *x, const INT *incx);

void CGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void CGEMV(const char *trans, const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void CGERC(const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void CGERU(const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void CHBMV(const char *uplo, const INT *n, const INT *k, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void CHEMV(const char *uplo, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void CHER(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *a, const INT *lda); 
void CHER2(const char *uplo, const INT *n, const Complex8 *alpha,
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void CHPMV(const char *uplo, const INT *n, const Complex8 *alpha, const Complex8 *ap,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void CHPR(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *ap); 
void CHPR2(const char *uplo, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *ap); 
void CTBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void CTBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void CTPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void CTPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void CTRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *incx); 
void CTRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 

void DGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const double *alpha, const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void DGEMV(const char *trans, const INT *m, const INT *n, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void DGER(const INT *m, const INT *n, const double *alpha, const double *x, const INT *incx,
          const double *y, const INT *incy, double *a, const INT *lda);
void DSBMV(const char *uplo, const INT *n, const INT *k, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void DSPMV(const char *uplo, const INT *n, const double *alpha, const double *ap,
           const double *x, const INT *incx, const double *beta, double *y, const INT *incy);
void DSPR(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx, double *ap);
void DSPR2(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx, 
           const double *y, const INT *incy, double *ap);
void DSYMV(const char *uplo, const INT *n, const double *alpha, const double *a, const INT *lda,
           const double *x, const INT *incx, const double *beta, double *y, const INT *incy);
void DSYR(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
          double *a, const INT *lda);
void DSYR2(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
           const double *y, const INT *incy, double *a, const INT *lda);
void DTBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void DTBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void DTPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void DTPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void DTRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const double *a, const INT *lda, double *b, const INT *incx);
void DTRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *a, const INT *lda, double *x, const INT *incx);

void ZGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void ZGEMV(const char *trans, const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void ZGERC(const INT *m, const INT *n, const Complex16 *alpha, 
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void ZGERU(const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void ZHBMV(const char *uplo, const INT *n, const INT *k, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void ZHEMV(const char *uplo, const INT *n, const Complex16 *alpha, 
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void ZHER(const char *uplo, const INT *n, const double *alpha,
          const Complex16 *x, const INT *incx, Complex16 *a, const INT *lda); 
void ZHER2(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void ZHPMV(const char *uplo, const INT *n, const Complex16 *alpha, const Complex16 *ap,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void ZHPR(const char *uplo, const INT *n, const double *alpha, const Complex16 *x,
          const INT *incx, Complex16 *ap); 
void ZHPR2(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *ap); 
void ZTBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ZTBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ZTPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *ap, Complex16 *x, const INT *incx); 
void ZTPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           Complex16 *ap, Complex16 *x, const INT *incx); 
void ZTRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *incx); 
void ZTRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 

/* BLAS Level3 */

void SGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void SSYMM(const char *side, const char *uplo, const INT *m, const INT *n, 
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void SSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k, 
            const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
            const float *beta, float *c, const INT *ldc);
void SSYRK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, 
           const float *beta, float *c, const INT *ldc);
void STRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);
void STRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);

void CGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void CHEMM(const char *side, const char *uplo, const INT *m, const INT *n, 
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void CHER2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb, const float *beta,
            Complex8 *c, const INT *ldc); 
void CHERK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const Complex8 *a, const INT *lda,
           const float *beta, Complex8 *c, const INT *ldc); 
void CSYMM(const char *side, const char *uplo, const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *b, const INT *ldb,
           const Complex8 *beta, Complex8 *c, const INT *ldc); 
void CSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb,
            const Complex8 *beta, Complex8 *c, const INT *ldc); 
void CSYRK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *beta, Complex8 *c, const INT *ldc); 
void CTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda,
           Complex8 *b, const INT *ldb); 
void CTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda,
           Complex8 *b, const INT *ldb); 

void DGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void DSYMM(const char *side, const char *uplo, const INT *m, const INT *n, 
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void DSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
            double *beta, double *c, const INT *ldc);
void DSYRK(const char *uplo, const char *trans, const INT *n, const INT *k, 
           const double *alpha, const double *a, const INT *lda, const double *beta,
           double *c, const INT *ldc);
void DTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);
void DTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);

void ZGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void ZHEMM(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void ZHER2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const double *beta,
            Complex16 *c, const INT *ldc); 
void ZHERK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const double *alpha, const Complex16 *a, const INT *lda,
           const double *beta, Complex16 *c, const INT *ldc); 
void ZSYMM(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void ZSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const Complex16 *beta,
            Complex16 *c, const INT *ldc); 
void ZSYRK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *beta, Complex16 *c, const INT *ldc); 
void ZTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha, 
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 
void ZTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 

/* Lower case declaration */

/* BLAS Level1 */
double dcabs1(Complex16 *z);
float sasum(const INT *n, const float *x, const INT *incx);
void  saxpy(const INT *n, const float *alpha, const float *x, const INT *incx,
            float *y, const INT *incy);
void  saxpyi(const INT *nz, const float *a, const float *x, const INT *indx, float *y);
float scasum(const INT *n, const Complex8 *x, const INT *incx); 
float scnrm2(const INT *n, const Complex8 *x, const INT *incx); 
void  scopy(const INT *n, const float *x, const INT *incx, float *y, const INT *incy);
float sdot(const INT *n, const float *x, const INT *incx, const float *y, const INT *incy);
float sdoti(const INT *nz, const float *x, const INT *indx, const float *y);
void  sgthr(const INT *nz, const float *y, float *x, const INT *indx);
void  sgthrz(const INT *nz, float *y, float *x, const INT *indx);
float snrm2(const INT *n, const float *x, const INT *incx);
void  srot(const INT *n, float *x, const INT *incx, float *y, const INT *incy,
           const float *c, const float *s);
void  srotg(float *a,float *b,float *c,float *s);
void  sroti(const INT *nz, float *x, const INT *indx, float *y, const float *c, const float *s);
void  srotm(const INT *n, float *x, const INT *incx, float *y, const INT *incy, const float *param);
void  srotmg(float *d1, float *d2, float *x1, const float *y1, float *param);
void  sscal(const INT *n, const float *a, float *x, const INT *incx);
void  ssctr(const INT *nz, const float *x, const INT *indx, float *y);
void  sswap(const INT *n, float *x, const INT *incx, float *y, const INT *incy);
INT   isamax(const INT *n, const float *x, const INT *incx);
INT   isamin(const INT *n, const float *x, const INT *incx);

void caxpy(const INT *n, const Complex8 *alpha, const Complex8 *x, const INT *incx,
           Complex8 *y, const INT *incy); 
void caxpyi(const INT *nz, const Complex8 *a, const Complex8 *x, const INT *indx,
            Complex8 *y); 
void ccopy(const INT *n, const Complex8 *x, const INT *incx, Complex8 *y, const INT *incy); 
void cdotc(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const Complex8 *y, const INT *incy); 
void cdotci(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void cdotu(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const Complex8 *y, const INT *incy); 
void cdotui(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void cgthr(const INT *nz, const Complex8 *y, Complex8 *x, const INT *indx); 
void cgthrz(const INT *nz, Complex8 *y, Complex8 *x, const INT *indx); 
void crotg(Complex8 *a, const Complex8 *b, float *c, Complex8 *s); 
void cscal(const INT *n, const Complex8 *a, Complex8 *x, const INT *incx); 
void csctr(const INT *nz, const Complex8 *x, const INT *indx, Complex8 *y); 
void csrot(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy,
           const float *c, const float *s); 
void csscal(const INT *n, const float *a, Complex8 *x, const INT *incx); 
void cswap(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy); 
INT  icamax(const INT *n, const Complex8 *x, const INT *incx); 
INT  icamin(const INT *n, const Complex8 *x, const INT *incx); 

double dasum(const INT *n, const double *x, const INT *incx);
void   daxpy(const INT *n, const double *alpha, const double *x, const INT *incx,
             double *y, const INT *incy);
void   daxpyi(const INT *nz, const double *a, const double *x, const INT *indx, double *y);
void   dcopy(const INT *n, const double *x, const INT *incx, double *y, const INT *incy);
double ddot(const INT *n, const double *x, const INT *incx, const double *y, const INT *incy);
double ddoti(const INT *nz, const double *x, const INT *indx, const double *y);
void   dgthr(const INT *nz, const double *y, double *x, const INT *indx);
void   dgthrz(const INT *nz, double *y, double *x, const INT *indx);
double dnrm2(const INT *n, const double *x, const INT *incx);
void   drot(const INT *n, double *x, const INT *incx, double *y, const INT *incy,
            const double *c, const double *s);
void   drotg(double *a, double *b, double *c, double *s);
void   droti(const INT *nz, double *x, const INT *indx, double *y, const double *c, const double *s);
void   drotm(const INT *n, double *x, const INT *incx, double *y, const INT *incy, const double *param);
void   drotmg(double *d1, double *d2, double *x1, const double *y1, double *param);
void   dscal(const INT *n, const double *a, double *x, const INT *incx);
void   dsctr(const INT *nz, const double *x, const INT *indx, double *y);
void   dswap(const INT *n, double *x, const INT *incx, double *y, const INT *incy);
double dzasum(const INT *n, const Complex16 *x, const INT *incx); 
double dznrm2(const INT *n, const Complex16 *x, const INT *incx); 
INT    idamax(const INT *n, const double *x, const INT *incx);
INT    idamin(const INT *n, const double *x, const INT *incx);

void zaxpy(const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void zaxpyi(const INT *nz, const Complex16 *a, const Complex16 *x, const INT *indx,
            Complex16 *y); 
void zcopy(const INT *n, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void zdotc(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy); 
void zdotci(Complex16 *pres, const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void zdotu(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy); 
void zdotui(Complex16 *pres, const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void zdrot(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy,
           const double *c, const double *s); 
void zdscal(const INT *n, const double *a, Complex16 *x, const INT *incx); 
void zgthr(const INT *nz, const Complex16 *y, Complex16 *x, const INT *indx); 
void zgthrz(const INT *nz, Complex16 *y, Complex16 *x, const INT *indx); 
void zrotg(Complex16 *a, const Complex16 *b, double *c, Complex16 *s); 
void zscal(const INT *n, const Complex16 *a, Complex16 *x, const INT *incx); 
void zsctr(const INT *nz, const Complex16 *x, const INT *indx, Complex16 *y); 
void zswap(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy); 
INT  izamax(const INT *n, const Complex16 *x, const INT *incx); 
INT  izamin(const INT *n, const Complex16 *x, const INT *incx); 

/* blas level2 */

void sgbmv(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const float *alpha, const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void sgemv(const char *trans, const INT *m, const INT *n, const float *alpha,
           const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void sger(const INT *m, const INT *n, const float *alpha, const float *x, const INT *incx,
          const float *y, const INT *incy, float *a, const INT *lda);
void ssbmv(const char *uplo, const INT *n, const INT *k, const float *alpha,
           const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void sspmv(const char *uplo, const INT *n, const float *alpha, const float *ap,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void sspr(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
          float *ap);
void sspr2(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *ap);
void ssymv(const char *uplo, const INT *n, const float *alpha, const float *a, const INT *lda,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void ssyr(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
          float *a, const INT *lda);
void ssyr2(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *a, const INT *lda);
void stbmv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void stbsv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void stpmv(const char *uplo, const char *trans, const char *diag, const INT *n, const float *ap,
           float *x, const INT *incx);
void stpsv(const char *uplo, const char *trans, const char *diag, const INT *n, const float *ap,
           float *x, const INT *incx);
void strmv(const char *uplo, const char *transa, const char *diag, const INT *n, const float *a,
           const INT *lda, float *b, const INT *incx);
void strsv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *a, const INT *lda, float *x, const INT *incx);

void cgbmv(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void cgemv(const char *trans, const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void cgerc(const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void cgeru(const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void chbmv(const char *uplo, const INT *n, const INT *k, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void chemv(const char *uplo, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void cher(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *a, const INT *lda); 
void cher2(const char *uplo, const INT *n, const Complex8 *alpha,
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void chpmv(const char *uplo, const INT *n, const Complex8 *alpha, const Complex8 *ap,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void chpr(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *ap); 
void chpr2(const char *uplo, const INT *n, const Complex8 *alpha, const Complex8 *x, const INT *incx,
           const Complex8 *y, const INT *incy, Complex8 *ap); 
void ctbmv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void ctbsv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void ctpmv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void ctpsv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void ctrmv(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *incx); 
void ctrsv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 

void dgbmv(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const double *alpha, const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void dgemv(const char *trans, const INT *m, const INT *n, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void dger(const INT *m, const INT *n, const double *alpha, const double *x, const INT *incx,
          const double *y, const INT *incy, double *a, const INT *lda);
void dsbmv(const char *uplo, const INT *n, const INT *k, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void dspmv(const char *uplo, const INT *n, const double *alpha, const double *ap,
           const double *x, const INT *incx, const double *beta,
           double *y, const INT *incy);
void dspr(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
          double *ap);
void dspr2(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
           const double *y, const INT *incy, double *ap);
void dsymv(const char *uplo, const INT *n, const double *alpha, const double *a, const INT *lda,
           const double *x, const INT *incx, const double *beta, double *y, const INT *incy);
void dsyr(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
          double *a, const INT *lda);
void dsyr2(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
           const double *y, const INT *incy, double *a, const INT *lda);
void dtbmv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void dtbsv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void dtpmv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void dtpsv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void dtrmv(const char *uplo, const char *transa, const char *diag, const INT *n,
           const double *a, const INT *lda, double *b, const INT *incx);
void dtrsv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *a, const INT *lda, double *x, const INT *incx);

void zgbmv(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void zgemv(const char *trans, const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void zgerc(const INT *m, const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy, Complex16 *a, const INT *lda); 
void zgeru(const INT *m, const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy, Complex16 *a, const INT *lda); 
void zhbmv(const char *uplo, const INT *n, const INT *k, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void zhemv(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void zher(const char *uplo, const INT *n, const double *alpha, const Complex16 *x, const INT *incx,
          Complex16 *a, const INT *lda); 
void zher2(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void zhpmv(const char *uplo, const INT *n, const Complex16 *alpha, const Complex16 *ap,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void zhpr(const char *uplo, const INT *n, const double *alpha, const Complex16 *x, const INT *incx,
          Complex16 *ap); 
void zhpr2(const char *uplo, const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy, Complex16 *ap); 
void ztbmv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ztbsv(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ztpmv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *ap, Complex16 *x, const INT *incx); 
void ztpsv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *ap, Complex16 *x, const INT *incx); 
void ztrmv(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *incx); 
void ztrsv(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 

/* blas level3 */

void sgemm(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void ssymm(const char *side, const char *uplo, const INT *m, const INT *n,
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void ssyr2k(const char *uplo, const char *trans, const INT *n, const INT *k,
            const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
            float *beta, float *c, const INT *ldc);
void ssyrk(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, const float *beta,
           float *c, const INT *ldc);
void strmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);
void strsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);

void cgemm(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void chemm(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void cher2k(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb, const float *beta,
            Complex8 *c, const INT *ldc); 
void cherk(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const Complex8 *a, const INT *lda, const float *beta,
           Complex8 *c, const INT *ldc); 
void csymm(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void csyr2k(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb, const Complex8 *beta,
            Complex8 *c, const INT *ldc); 
void csyrk(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *beta, Complex8 *c, const INT *ldc); 
void ctrmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *ldb); 
void ctrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *ldb); 

void dgemm(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void dsymm(const char *side, const char *uplo, const INT *m, const INT *n,
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void dsyr2k(const char *uplo, const char *trans, const INT *n, const INT *k,
            const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
            const double *beta, double *c, const INT *ldc);
void dsyrk(const char *uplo, const char *trans, const INT *n, const INT *k,
           const double *alpha, const double *a, const INT *lda, const double *beta,
           double *c, const INT *ldc);
void dtrmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);
void dtrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);

void zgemm(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void zhemm(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void zher2k(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const double *beta,
            Complex16 *c, const INT *ldc); 
void zherk(const char *uplo, const char *trans, const INT *n, const INT *k,
           const double *alpha, const Complex16 *a, const INT *lda,
           const double *beta, Complex16 *c, const INT *ldc); 
void zsymm(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void zsyr2k(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const Complex16 *beta,
            Complex16 *c, const INT *ldc); 
void zsyrk(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *beta, Complex16 *c, const INT *ldc); 
void ztrmm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 
void ztrsm(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha, 
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 

/* Upper case declaration */

/* BLAS Level1 */
                              
double DCABS1(const Complex16 *z);
float SASUM(const INT *n, const float *x, const INT *incx);
void  SAXPY(const INT *n, const float *alpha, const float *x, const INT *incx,
            float *y, const INT *incy);
void  SAXPYI(const INT *nz, const float *a, const float *x, const INT *indx,float *y);
float SCASUM(const INT *n, const Complex8 *x, const INT *incx); 
float SCNRM2(const INT *n, const Complex8 *x, const INT *incx); 
void  SCOPY(const INT *n, const float *x, const INT *incx, float *y, const INT *incy);
float SDOT(const INT *n, const float *x, const INT *incx,
           const float *y, const INT *incy);
double DSDOT(const INT *n, const float *x, const INT *incx, 
             const float *y, const INT *incy);
float SDOTI(const INT *nz, const float *x, const INT *indx, const float *y);
void  SGTHR(const INT *nz, const float *y, float *x, const INT *indx);
void  SGTHRZ(const INT *nz, float *y, float *x, const INT *indx);
float SNRM2(const INT *n, const float *x, const INT *incx);
void  SROT(const INT *n, float *x, const INT *incx, float *y, const INT *incy,
           const float *c, const float *s);
void  SROTG(float *a,float *b,float *c,float *s);
void  SROTI(const INT *nz, float *x, const INT *indx, float *y, const float *c, const float *s);
void  SROTM(const INT *n, float *x, const INT *incx, float *y, const INT *incy, const float *param);
void  SROTMG(float *d1, float *d2, float *x1, const float *y1, float *param);
void  SSCAL(const INT *n, const float *a, float *x, const INT *incx);
void  SSCTR(const INT *nz, const float *x, const INT *indx, float *y);
void  SSWAP(const INT *n, float *x, const INT *incx, float *y, const INT *incy);
INT   ISAMAX(const INT *n, const float *x, const INT *incx);
INT   ISAMIN(const INT *n, const float *x, const INT *incx);

void CAXPY(const INT *n, const Complex8 *alpha, const Complex8 *x, const INT *incx,
           Complex8 *y, const INT *incy); 
void CAXPYI(const INT *nz, const Complex8 *a, 
            const Complex8 *x, const INT *indx, Complex8 *y); 
void CCOPY(const INT *n, const Complex8 *x, const INT *incx,
           Complex8 *y, const INT *incy); 
void CDOTC(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const  Complex8 *y, const INT *incy); 
void CDOTCI(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void CDOTU(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const  Complex8 *y, const INT *incy); 
void CDOTUI(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void CGTHR(const INT *nz, const Complex8 *y, Complex8 *x, const INT *indx); 
void CGTHRZ(const INT *nz, Complex8 *y, Complex8 *x, const INT *indx); 
void CROTG(Complex8 *a, const Complex8 *b, float *c, Complex8 *s); 
void CSCAL(const INT *n, const Complex8 *a, Complex8 *x, const INT *incx); 
void CSCTR(const INT *nz, const Complex8 *x, const INT *indx, Complex8 *y); 
void CSROT(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy,
           const float *c, const float *s); 
void CSSCAL(const INT *n, const float *a, Complex8 *x, const INT *incx); 
void CSWAP(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy); 
INT  ICAMAX(const INT *n, const Complex8 *x, const INT *incx); 
INT  ICAMIN(const INT *n, const Complex8 *x, const INT *incx); 

double DASUM(const INT *n, const double *x, const INT *incx);
void   DAXPY(const INT *n, const double *alpha, const double *x, const INT *incx,
             double *y, const INT *incy);
void   DAXPYI(const INT *nz, const double *a, const double *x, const INT *indx, double *y);
void   DCOPY(const INT *n, const double *x, const INT *incx, double *y, const INT *incy);
double DDOT(const  INT *n, const double *x, const INT *incx, 
            const double *y, const INT *incy);
float  SDSDOT(const INT *n, const double *x, const INT *incx, 
              const double *y, const INT *incy);
double DDOTI(const INT *nz, const double *x, const INT *indx, const double *y);
void   DGTHR(const INT *nz, const double *y, double *x, const INT *indx);
void   DGTHRZ(const INT *nz, double *y, double *x, const INT *indx);
double DNRM2(const INT *n, const double *x, const INT *incx);
void   DROT(const INT *n, double *x, const INT *incx, double *y, const INT *incy,
            const double *c, const double *s);
void   DROTG(double *a,double *b,double *c,double *s);
void   DROTI(const INT *nz, double *x, const INT *indx, double *y, const double *c, const double *s);
void   DROTM(const INT *n, double *x, const INT *incx, double *y, const INT *incy, const double *param);
void   DROTMG(double *d1, double *d2, double *x1, const double *y1, double *param);
void   DSCAL(const INT *n, const double *a, double *x, const INT *incx);
void   DSCTR(const INT *nz, const double *x, const INT *indx, double *y);
void   DSWAP(const INT *n, double *x, const INT *incx, double *y, const INT *incy);
double DZASUM(const INT *n, const Complex16 *x, const INT *incx); 
double DZNRM2(const INT *n, const Complex16 *x, const INT *incx); 
INT    IDAMAX(const INT *n, const double *x, const INT *incx);
INT    IDAMIN(const INT *n, const double *x, const INT *incx);

void ZAXPY(const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void ZAXPYI(const INT *nz, const Complex16 *a,
            const Complex16 *x, const INT *indx, Complex16 *y); 
void ZCOPY(const INT *n, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void ZDOTC(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const  Complex16 *y, const INT *incy); 
void ZDOTCI(Complex16 *pres,const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void ZDOTU(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy); 
void ZDOTUI(Complex16 *pres, const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void ZDROT(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy,
           const double *c, const double *s); 
void ZDSCAL(const INT *n, const double *a, Complex16 *x, const INT *incx); 
void ZGTHR(const INT *nz, const Complex16 *y, Complex16 *x, const INT *indx); 
void ZGTHRZ(const INT *nz, Complex16 *y, Complex16 *x, const INT *indx); 
void ZROTG(Complex16 *a, const Complex16 *b, double *c, Complex16 *s); 
void ZSCAL(const INT *n, const Complex16 *a, Complex16 *x, const INT *incx); 
void ZSCTR(const INT *nz, const Complex16 *x, const INT *indx, Complex16 *y); 
void ZSWAP(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy); 
INT  IZAMAX(const INT *n, const Complex16 *x, const INT *incx); 
INT  IZAMIN(const  INT *n,const  Complex16 *x, const INT *incx); 

/* BLAS Level2 */

void SGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const float *alpha, const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void SGEMV(const char *trans, const INT *m, const INT *n, const float *alpha,
           const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void SGER(const INT *m, const INT *n, const float *alpha, const float *x, const INT *incx,
          const float *y, const INT *incy, float *a, const INT *lda);
void SSBMV(const char *uplo, const INT *n, const INT *k, 
           const float *alpha, const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void SSPMV(const char *uplo, const INT *n, const float *alpha, const float *ap,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void SSPR(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx, float *ap);
void SSPR2(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *ap);
void SSYMV(const char *uplo, const INT *n, const float *alpha, const float *a, const INT *lda,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void SSYR(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
          float *a, const INT *lda);
void SSYR2(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *a, const INT *lda);
void STBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void STBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void STPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *ap, float *x, const INT *incx);
void STPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *ap, float *x, const INT *incx);
void STRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const float *a, const INT *lda, float *b, const INT *incx);
void STRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *a, const INT *lda, float *x, const INT *incx);

void CGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void CGEMV(const char *trans, const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void CGERC(const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void CGERU(const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void CHBMV(const char *uplo, const INT *n, const INT *k, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void CHEMV(const char *uplo, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void CHER(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *a, const INT *lda); 
void CHER2(const char *uplo, const INT *n, const Complex8 *alpha,
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void CHPMV(const char *uplo, const INT *n, const Complex8 *alpha, const Complex8 *ap,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void CHPR(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *ap); 
void CHPR2(const char *uplo, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *ap); 
void CTBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void CTBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void CTPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void CTPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void CTRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *incx); 
void CTRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 

void DGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const double *alpha, const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void DGEMV(const char *trans, const INT *m, const INT *n, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void DGER(const INT *m, const INT *n, const double *alpha, const double *x, const INT *incx,
          const double *y, const INT *incy, double *a, const INT *lda);
void DSBMV(const char *uplo, const INT *n, const INT *k, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void DSPMV(const char *uplo, const INT *n, const double *alpha, const double *ap,
           const double *x, const INT *incx, const double *beta, double *y, const INT *incy);
void DSPR(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx, double *ap);
void DSPR2(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx, 
           const double *y, const INT *incy, double *ap);
void DSYMV(const char *uplo, const INT *n, const double *alpha, const double *a, const INT *lda,
           const double *x, const INT *incx, const double *beta, double *y, const INT *incy);
void DSYR(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
          double *a, const INT *lda);
void DSYR2(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
           const double *y, const INT *incy, double *a, const INT *lda);
void DTBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void DTBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void DTPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void DTPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void DTRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const double *a, const INT *lda, double *b, const INT *incx);
void DTRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *a, const INT *lda, double *x, const INT *incx);

void ZGBMV(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void ZGEMV(const char *trans, const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void ZGERC(const INT *m, const INT *n, const Complex16 *alpha, 
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void ZGERU(const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void ZHBMV(const char *uplo, const INT *n, const INT *k, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void ZHEMV(const char *uplo, const INT *n, const Complex16 *alpha, 
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void ZHER(const char *uplo, const INT *n, const double *alpha,
          const Complex16 *x, const INT *incx, Complex16 *a, const INT *lda); 
void ZHER2(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void ZHPMV(const char *uplo, const INT *n, const Complex16 *alpha, const Complex16 *ap,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void ZHPR(const char *uplo, const INT *n, const double *alpha, const Complex16 *x,
          const INT *incx, Complex16 *ap); 
void ZHPR2(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *ap); 
void ZTBMV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ZTBSV(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ZTPMV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *ap, Complex16 *x, const INT *incx); 
void ZTPSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           Complex16 *ap, Complex16 *x, const INT *incx); 
void ZTRMV(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *incx); 
void ZTRSV(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 

/* BLAS Level3 */

void SGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void SSYMM(const char *side, const char *uplo, const INT *m, const INT *n, 
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void SSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k, 
            const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
            const float *beta, float *c, const INT *ldc);
void SSYRK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, 
           const float *beta, float *c, const INT *ldc);
void STRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);
void STRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);

void CGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void CHEMM(const char *side, const char *uplo, const INT *m, const INT *n, 
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void CHER2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb, const float *beta,
            Complex8 *c, const INT *ldc); 
void CHERK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const Complex8 *a, const INT *lda,
           const float *beta, Complex8 *c, const INT *ldc); 
void CSYMM(const char *side, const char *uplo, const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *b, const INT *ldb,
           const Complex8 *beta, Complex8 *c, const INT *ldc); 
void CSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb,
            const Complex8 *beta, Complex8 *c, const INT *ldc); 
void CSYRK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *beta, Complex8 *c, const INT *ldc); 
void CTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda,
           Complex8 *b, const INT *ldb); 
void CTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda,
           Complex8 *b, const INT *ldb); 

void DGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void DSYMM(const char *side, const char *uplo, const INT *m, const INT *n, 
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void DSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
            double *beta, double *c, const INT *ldc);
void DSYRK(const char *uplo, const char *trans, const INT *n, const INT *k, 
           const double *alpha, const double *a, const INT *lda, const double *beta,
           double *c, const INT *ldc);
void DTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);
void DTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);

void ZGEMM(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void ZHEMM(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void ZHER2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const double *beta,
            Complex16 *c, const INT *ldc); 
void ZHERK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const double *alpha, const Complex16 *a, const INT *lda,
           const double *beta, Complex16 *c, const INT *ldc); 
void ZSYMM(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void ZSYR2K(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const Complex16 *beta,
            Complex16 *c, const INT *ldc); 
void ZSYRK(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *beta, Complex16 *c, const INT *ldc); 
void ZTRMM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha, 
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 
void ZTRSM(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 

/* Lower case declaration */

/* BLAS Level1 */
double dcabs1_(Complex16 *z);
float sasum_(const INT *n, const float *x, const INT *incx);
void  saxpy_(const INT *n, const float *alpha, const float *x, const INT *incx,
            float *y, const INT *incy);
void  saxpyi_(const INT *nz, const float *a, const float *x, const INT *indx, float *y);
float scasum_(const INT *n, const Complex8 *x, const INT *incx); 
float scnrm2_(const INT *n, const Complex8 *x, const INT *incx); 
void  scopy_(const INT *n, const float *x, const INT *incx, float *y, const INT *incy);
float sdot_(const INT *n, const float *x, const INT *incx, const float *y, const INT *incy);
float sdoti_(const INT *nz, const float *x, const INT *indx, const float *y);
void  sgthr_(const INT *nz, const float *y, float *x, const INT *indx);
void  sgthrz_(const INT *nz, float *y, float *x, const INT *indx);
float snrm2_(const INT *n, const float *x, const INT *incx);
void  srot_(const INT *n, float *x, const INT *incx, float *y, const INT *incy,
           const float *c, const float *s);
void  srotg_(float *a,float *b,float *c,float *s);
void  sroti_(const INT *nz, float *x, const INT *indx, float *y, const float *c, const float *s);
void  srotm_(const INT *n, float *x, const INT *incx, float *y, const INT *incy, const float *param);
void  srotmg_(float *d1, float *d2, float *x1, const float *y1, float *param);
void  sscal_(const INT *n, const float *a, float *x, const INT *incx);
void  ssctr_(const INT *nz, const float *x, const INT *indx, float *y);
void  sswap_(const INT *n, float *x, const INT *incx, float *y, const INT *incy);
INT   isamax_(const INT *n, const float *x, const INT *incx);
INT   isamin_(const INT *n, const float *x, const INT *incx);

void caxpy_(const INT *n, const Complex8 *alpha, const Complex8 *x, const INT *incx,
           Complex8 *y, const INT *incy); 
void caxpyi_(const INT *nz, const Complex8 *a, const Complex8 *x, const INT *indx,
            Complex8 *y); 
void ccopy_(const INT *n, const Complex8 *x, const INT *incx, Complex8 *y, const INT *incy); 
void cdotc_(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const Complex8 *y, const INT *incy); 
void cdotci_(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void cdotu_(Complex8 *pres, const INT *n, const Complex8 *x, const INT *incx,
           const Complex8 *y, const INT *incy); 
void cdotui_(Complex8 *pres, const INT *nz, const Complex8 *x, const INT *indx,
            const Complex8 *y); 
void cgthr_(const INT *nz, const Complex8 *y, Complex8 *x, const INT *indx); 
void cgthrz_(const INT *nz, Complex8 *y, Complex8 *x, const INT *indx); 
void crotg_(Complex8 *a, const Complex8 *b, float *c, Complex8 *s); 
void cscal_(const INT *n, const Complex8 *a, Complex8 *x, const INT *incx); 
void csctr_(const INT *nz, const Complex8 *x, const INT *indx, Complex8 *y); 
void csrot_(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy,
           const float *c, const float *s); 
void csscal_(const INT *n, const float *a, Complex8 *x, const INT *incx); 
void cswap_(const INT *n, Complex8 *x, const INT *incx, Complex8 *y, const INT *incy); 
INT  icamax_(const INT *n, const Complex8 *x, const INT *incx); 
INT  icamin_(const INT *n, const Complex8 *x, const INT *incx); 

double dasum_(const INT *n, const double *x, const INT *incx);
void   daxpy_(const INT *n, const double *alpha, const double *x, const INT *incx,
             double *y, const INT *incy);
void   daxpyi_(const INT *nz, const double *a, const double *x, const INT *indx, double *y);
void   dcopy_(const INT *n, const double *x, const INT *incx, double *y, const INT *incy);
double ddot_(const INT *n, const double *x, const INT *incx, const double *y, const INT *incy);
double ddoti_(const INT *nz, const double *x, const INT *indx, const double *y);
void   dgthr_(const INT *nz, const double *y, double *x, const INT *indx);
void   dgthrz_(const INT *nz, double *y, double *x, const INT *indx);
double dnrm2_(const INT *n, const double *x, const INT *incx);
void   drot_(const INT *n, double *x, const INT *incx, double *y, const INT *incy,
            const double *c, const double *s);
void   drotg_(double *a, double *b, double *c, double *s);
void   droti_(const INT *nz, double *x, const INT *indx, double *y, const double *c, const double *s);
void   drotm_(const INT *n, double *x, const INT *incx, double *y, const INT *incy, const double *param);
void   drotmg_(double *d1, double *d2, double *x1, const double *y1, double *param);
void   dscal_(const INT *n, const double *a, double *x, const INT *incx);
void   dsctr_(const INT *nz, const double *x, const INT *indx, double *y);
void   dswap_(const INT *n, double *x, const INT *incx, double *y, const INT *incy);
double dzasum_(const INT *n, const Complex16 *x, const INT *incx); 
double dznrm2_(const INT *n, const Complex16 *x, const INT *incx); 
INT    idamax_(const INT *n, const double *x, const INT *incx);
INT    idamin_(const INT *n, const double *x, const INT *incx);

void zaxpy_(const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void zaxpyi_(const INT *nz, const Complex16 *a, const Complex16 *x, const INT *indx,
            Complex16 *y); 
void zcopy_(const INT *n, const Complex16 *x, const INT *incx,
           Complex16 *y, const INT *incy); 
void zdotc_(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy); 
void zdotci_(Complex16 *pres, const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void zdotu_(Complex16 *pres, const INT *n, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy); 
void zdotui_(Complex16 *pres, const INT *nz, const Complex16 *x, const INT *indx,
            const Complex16 *y); 
void zdrot_(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy,
           const double *c, const double *s); 
void zdscal_(const INT *n, const double *a, Complex16 *x, const INT *incx); 
void zgthr_(const INT *nz, const Complex16 *y, Complex16 *x, const INT *indx); 
void zgthrz_(const INT *nz, Complex16 *y, Complex16 *x, const INT *indx); 
void zrotg_(Complex16 *a, const Complex16 *b, double *c, Complex16 *s); 
void zscal_(const INT *n, const Complex16 *a, Complex16 *x, const INT *incx); 
void zsctr_(const INT *nz, const Complex16 *x, const INT *indx, Complex16 *y); 
void zswap_(const INT *n, Complex16 *x, const INT *incx, Complex16 *y, const INT *incy); 
INT  izamax_(const INT *n, const Complex16 *x, const INT *incx); 
INT  izamin_(const INT *n, const Complex16 *x, const INT *incx); 

/* blas level2 */

void sgbmv_(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const float *alpha, const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void sgemv_(const char *trans, const INT *m, const INT *n, const float *alpha,
           const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void sger_(const INT *m, const INT *n, const float *alpha, const float *x, const INT *incx,
          const float *y, const INT *incy, float *a, const INT *lda);
void ssbmv_(const char *uplo, const INT *n, const INT *k, const float *alpha,
           const float *a, const INT *lda, const float *x, const INT *incx,
           const float *beta, float *y, const INT *incy);
void sspmv_(const char *uplo, const INT *n, const float *alpha, const float *ap,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void sspr_(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
          float *ap);
void sspr2_(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *ap);
void ssymv_(const char *uplo, const INT *n, const float *alpha, const float *a, const INT *lda,
           const float *x, const INT *incx, const float *beta, float *y, const INT *incy);
void ssyr_(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
          float *a, const INT *lda);
void ssyr2_(const char *uplo, const INT *n, const float *alpha, const float *x, const INT *incx,
           const float *y, const INT *incy, float *a, const INT *lda);
void stbmv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void stbsv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const float *a, const INT *lda, float *x, const INT *incx);
void stpmv_(const char *uplo, const char *trans, const char *diag, const INT *n, const float *ap,
           float *x, const INT *incx);
void stpsv_(const char *uplo, const char *trans, const char *diag, const INT *n, const float *ap,
           float *x, const INT *incx);
void strmv_(const char *uplo, const char *transa, const char *diag, const INT *n, const float *a,
           const INT *lda, float *b, const INT *incx);
void strsv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const float *a, const INT *lda, float *x, const INT *incx);

void cgbmv_(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void cgemv_(const char *trans, const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void cgerc_(const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void cgeru_(const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void chbmv_(const char *uplo, const INT *n, const INT *k, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void chemv_(const char *uplo, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, const Complex8 *x, const INT *incx,
           const Complex8 *beta, Complex8 *y, const INT *incy); 
void cher_(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *a, const INT *lda); 
void cher2_(const char *uplo, const INT *n, const Complex8 *alpha,
           const Complex8 *x, const INT *incx, const Complex8 *y, const INT *incy,
           Complex8 *a, const INT *lda); 
void chpmv_(const char *uplo, const INT *n, const Complex8 *alpha, const Complex8 *ap,
           const Complex8 *x, const INT *incx, const Complex8 *beta,
           Complex8 *y, const INT *incy); 
void chpr_(const char *uplo, const INT *n, const float *alpha, const Complex8 *x, const INT *incx,
          Complex8 *ap); 
void chpr2_(const char *uplo, const INT *n, const Complex8 *alpha, const Complex8 *x, const INT *incx,
           const Complex8 *y, const INT *incy, Complex8 *ap); 
void ctbmv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void ctbsv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 
void ctpmv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void ctpsv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *ap, Complex8 *x, const INT *incx); 
void ctrmv_(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *incx); 
void ctrsv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex8 *a, const INT *lda, Complex8 *x, const INT *incx); 

void dgbmv_(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const double *alpha, const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void dgemv_(const char *trans, const INT *m, const INT *n, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void dger_(const INT *m, const INT *n, const double *alpha, const double *x, const INT *incx,
          const double *y, const INT *incy, double *a, const INT *lda);
void dsbmv_(const char *uplo, const INT *n, const INT *k, const double *alpha,
           const double *a, const INT *lda, const double *x, const INT *incx,
           const double *beta, double *y, const INT *incy);
void dspmv_(const char *uplo, const INT *n, const double *alpha, const double *ap,
           const double *x, const INT *incx, const double *beta,
           double *y, const INT *incy);
void dspr_(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
          double *ap);
void dspr2_(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
           const double *y, const INT *incy, double *ap);
void dsymv_(const char *uplo, const INT *n, const double *alpha, const double *a, const INT *lda,
           const double *x, const INT *incx, const double *beta, double *y, const INT *incy);
void dsyr_(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
          double *a, const INT *lda);
void dsyr2_(const char *uplo, const INT *n, const double *alpha, const double *x, const INT *incx,
           const double *y, const INT *incy, double *a, const INT *lda);
void dtbmv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void dtbsv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const double *a, const INT *lda, double *x, const INT *incx);
void dtpmv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void dtpsv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *ap, double *x, const INT *incx);
void dtrmv_(const char *uplo, const char *transa, const char *diag, const INT *n,
           const double *a, const INT *lda, double *b, const INT *incx);
void dtrsv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const double *a, const INT *lda, double *x, const INT *incx);

void zgbmv_(const char *trans, const INT *m, const INT *n, const INT *kl, const INT *ku,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void zgemv_(const char *trans, const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void zgerc_(const INT *m, const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy, Complex16 *a, const INT *lda); 
void zgeru_(const INT *m, const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy, Complex16 *a, const INT *lda); 
void zhbmv_(const char *uplo, const INT *n, const INT *k, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void zhemv_(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, const Complex16 *x, const INT *incx,
           const Complex16 *beta, Complex16 *y, const INT *incy); 
void zher_(const char *uplo, const INT *n, const double *alpha, const Complex16 *x, const INT *incx,
          Complex16 *a, const INT *lda); 
void zher2_(const char *uplo, const INT *n, const Complex16 *alpha,
           const Complex16 *x, const INT *incx, const Complex16 *y, const INT *incy,
           Complex16 *a, const INT *lda); 
void zhpmv_(const char *uplo, const INT *n, const Complex16 *alpha, const Complex16 *ap,
           const Complex16 *x, const INT *incx, const Complex16 *beta,
           Complex16 *y, const INT *incy); 
void zhpr_(const char *uplo, const INT *n, const double *alpha, const Complex16 *x, const INT *incx,
          Complex16 *ap); 
void zhpr2_(const char *uplo, const INT *n, const Complex16 *alpha, const Complex16 *x, const INT *incx,
           const Complex16 *y, const INT *incy, Complex16 *ap); 
void ztbmv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ztbsv_(const char *uplo, const char *trans, const char *diag, const INT *n, const INT *k,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 
void ztpmv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *ap, Complex16 *x, const INT *incx); 
void ztpsv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *ap, Complex16 *x, const INT *incx); 
void ztrmv_(const char *uplo, const char *transa, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *incx); 
void ztrsv_(const char *uplo, const char *trans, const char *diag, const INT *n,
           const Complex16 *a, const INT *lda, Complex16 *x, const INT *incx); 

/* blas level3 */

void sgemm_(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void ssymm_(const char *side, const char *uplo, const INT *m, const INT *n,
           const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
           const float *beta, float *c, const INT *ldc);
void ssyr2k_(const char *uplo, const char *trans, const INT *n, const INT *k,
            const float *alpha, const float *a, const INT *lda, const float *b, const INT *ldb,
            float *beta, float *c, const INT *ldc);
void ssyrk_(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const float *a, const INT *lda, const float *beta,
           float *c, const INT *ldc);
void strmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);
void strsm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const float *alpha, const float *a, const INT *lda,
           float *b, const INT *ldb);

void cgemm_(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void chemm_(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void cher2k_(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb, const float *beta,
            Complex8 *c, const INT *ldc); 
void cherk_(const char *uplo, const char *trans, const INT *n, const INT *k,
           const float *alpha, const Complex8 *a, const INT *lda, const float *beta,
           Complex8 *c, const INT *ldc); 
void csymm_(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *b, const INT *ldb, const Complex8 *beta,
           Complex8 *c, const INT *ldc); 
void csyr2k_(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex8 *alpha, const Complex8 *a, const INT *lda,
            const Complex8 *b, const INT *ldb, const Complex8 *beta,
            Complex8 *c, const INT *ldc); 
void csyrk_(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex8 *alpha, const Complex8 *a, const INT *lda,
           const Complex8 *beta, Complex8 *c, const INT *ldc); 
void ctrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha, 
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *ldb); 
void ctrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex8 *alpha,
           const Complex8 *a, const INT *lda, Complex8 *b, const INT *ldb); 

void dgemm_(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void dsymm_(const char *side, const char *uplo, const INT *m, const INT *n,
           const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
           const double *beta, double *c, const INT *ldc);
void dsyr2k_(const char *uplo, const char *trans, const INT *n, const INT *k,
            const double *alpha, const double *a, const INT *lda, const double *b, const INT *ldb,
            const double *beta, double *c, const INT *ldc);
void dsyrk_(const char *uplo, const char *trans, const INT *n, const INT *k,
           const double *alpha, const double *a, const INT *lda, const double *beta,
           double *c, const INT *ldc);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const double *alpha, const double *a, const INT *lda,
           double *b, const INT *ldb);

void zgemm_(const char *transa, const char *transb, const INT *m, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void zhemm_(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void zher2k_(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const double *beta,
            Complex16 *c, const INT *ldc); 
void zherk_(const char *uplo, const char *trans, const INT *n, const INT *k,
           const double *alpha, const Complex16 *a, const INT *lda,
           const double *beta, Complex16 *c, const INT *ldc); 
void zsymm_(const char *side, const char *uplo, const INT *m, const INT *n,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *b, const INT *ldb, const Complex16 *beta,
           Complex16 *c, const INT *ldc); 
void zsyr2k_(const char *uplo, const char *trans, const INT *n, const INT *k,
            const Complex16 *alpha, const Complex16 *a, const INT *lda,
            const Complex16 *b, const INT *ldb, const Complex16 *beta,
            Complex16 *c, const INT *ldc); 
void zsyrk_(const char *uplo, const char *trans, const INT *n, const INT *k,
           const Complex16 *alpha, const Complex16 *a, const INT *lda,
           const Complex16 *beta, Complex16 *c, const INT *ldc); 
void ztrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha,
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 
void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
           const INT *m, const INT *n, const Complex16 *alpha, 
           const Complex16 *a, const INT *lda, Complex16 *b, const INT *ldb); 

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _BLAS_H_ */
