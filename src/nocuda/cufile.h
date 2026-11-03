/*
 * nocuda/cufile.h - stand-in for <cufile.h> (see cuda.h). Only the error plumbing the
 * host code touches; the cuFile calls themselves go through heterodb_extra, which is
 * loaded with dlopen() and is simply absent here.
 */
#ifndef PGSTROM_NOCUDA_CUFILE_H
#define PGSTROM_NOCUDA_CUFILE_H

#define CUFILEOP_BASE_ERR		5000
typedef int CUfileOpError;

static inline const char *
cufileop_status_error(CUfileOpError status)
{
	(void) status;
	return "cuFile error (not available in a build without CUDA)";
}
#endif	/* PGSTROM_NOCUDA_CUFILE_H */
