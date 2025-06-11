/* EH_transfer.c */
%module EH_transfer
%include "typemaps.i"
%include "carrays.i"
%array_class(float,floatArray)
%{
extern void TFfit_hmpc(float omega0, float f_baryon, float hubble, float Tcmb,
		       int numk, float *k, float *tf_full);
%}
extern void TFfit_hmpc(float omega0, float f_baryon, float hubble, float Tcmb,
		       int numk, float *k, float *tf_full);
