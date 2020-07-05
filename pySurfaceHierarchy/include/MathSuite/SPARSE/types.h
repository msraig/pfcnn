#ifndef _TYPES_H_
#define _TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* Complex type (single precision). */
typedef
struct _Complex8 {
    float    real;
    float    imag;
} Complex8;

/* Complex type (double precision). */
typedef
struct _Complex16 {
    double    real;
    double    imag;
} Complex16;

#ifdef MACHINE64
  #define INT long long
#else
  #define INT int
#endif

#ifdef __cplusplus
} 
#endif /* __cplusplus */

#endif /* _TYPES_H_ */
