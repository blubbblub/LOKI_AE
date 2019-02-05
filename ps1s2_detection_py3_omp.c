#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


/* Prototypes */
int stacking(long int nxyz, long int nsta, long int nsamples, int itp[nxyz][nsta], int its[nxyz][nsta], double stalta_p[nsta][nsamples], double stalta_s1[nsta][nsamples], double stalta_s2[nsta][nsamples], double corrmatrix[nxyz][nsamples], int nproc);
/* Python wrapper of the C function stacking */

static char module_docstring[]="Module for detection and location of seismic events P and S";
static char stacking_docstring[]="Detection and location through waveform stacking";


/* wrapper */

static PyObject *py_stacking(PyObject *self, PyObject *args){
   PyArrayObject *itp, *its, *stalta_p, *stalta_s1, *stalta_s2, *corrmatrix;
   long int nxyz, nsamples, nsta, nproc;
   npy_intp dims[2];
   /* checking the format of the arguments */

   if(!PyArg_ParseTuple(args, "OOOOOi", &itp, &its, &stalta_p, &stalta_s1, &stalta_s2, &nproc)){
      PyErr_SetString(PyExc_RuntimeError, "Invalid arguments for the C function stacking");
      return NULL;
   }

   /* Checking the contiguity of the arrays */

   if(!PyArray_Check(stalta_p) || !PyArray_ISCONTIGUOUS(stalta_p)){
      PyErr_SetString(PyExc_RuntimeError, "stalta_p is not a contiguous array");
      return NULL;
   }

   if(!PyArray_Check(stalta_s1) || !PyArray_ISCONTIGUOUS(stalta_s1)){
      PyErr_SetString(PyExc_RuntimeError, "stalta_s is not a contiguous array");
      return NULL;
   }

   if(!PyArray_Check(stalta_s2) || !PyArray_ISCONTIGUOUS(stalta_s2)){
      PyErr_SetString(PyExc_RuntimeError, "stalta_s is not a contiguous array");
      return NULL;
   }

   if(!PyArray_Check(itp) || !PyArray_ISCONTIGUOUS(itp)){
      PyErr_SetString(PyExc_RuntimeError, "tp is not a contiguous array");
      return NULL;
   }

   if(!PyArray_Check(its) || !PyArray_ISCONTIGUOUS(its)){
      PyErr_SetString(PyExc_RuntimeError, "ts is not a contiguous array");
      return NULL;
   }



   /* Checking that obs_data and stalta are the same type of array and with the same dimensions */

   if((PyArray_NDIM(stalta_p) != 2)){
      PyErr_SetString(PyExc_RuntimeError, "stalta_p is not a 2D array");
      return NULL;
   }

   if((PyArray_NDIM(stalta_s1) != 2)){
      PyErr_SetString(PyExc_RuntimeError, "stalta_s is not a 2D array");
      return NULL;
   }

   if((PyArray_NDIM(stalta_s2) != 2)){
      PyErr_SetString(PyExc_RuntimeError, "stalta_s is not a 2D array");
      return NULL;
   }

   if((PyArray_NDIM(itp) != 2)){
      PyErr_SetString(PyExc_RuntimeError, "tp is not a 2D array");
      return NULL;
   }

   if((PyArray_NDIM(its) != 2)){
      PyErr_SetString(PyExc_RuntimeError, "ts is not a 2D array");
      return NULL;
   }

   /* find the dimension of obs_data and stalta */
   nsta=(long int) PyArray_DIM(stalta_p, 0);
   nsamples=(long int) PyArray_DIM(stalta_p, 1);
   nxyz=(long int) PyArray_DIM(itp, 0);
   dims[0] = nxyz;
   dims[1] = nsamples;
   corrmatrix=(PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_DOUBLE);

   /*call stacking */
   if (0 != stacking(nxyz, nsta, nsamples, PyArray_DATA(itp), PyArray_DATA(its), PyArray_DATA(stalta_p), PyArray_DATA(stalta_s1), PyArray_DATA(stalta_s2), PyArray_DATA(corrmatrix), nproc)) {
      PyErr_SetString(PyExc_RuntimeError, "running stacking failed.");
      return NULL;
   }


   return PyArray_Return(corrmatrix);

}



/* module specifications and inizialization*/

static PyMethodDef module_methods[]={
 /* {method_name, Cfunction, argument_types, docstring} */
    {"stacking", py_stacking, METH_VARARGS, stacking_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef modps1s2_detection = {
       PyModuleDef_HEAD_INIT,
       "ps1s2_detection",
       module_docstring,
       -1,
       module_methods
};

PyMODINIT_FUNC PyInit_ps1s2_detection(void){
    PyObject *m;
    m = PyModule_Create(&modps1s2_detection);
    if (m==NULL)
       return NULL;
    import_array();
    return m;
};


int stacking(long int nxyz, long int nsta, long int nsamples, int itp[nxyz][nsta], int its[nxyz][nsta], double stalta_p[nsta][nsamples], double stalta_s1[nsta][nsamples] ,double stalta_s2[nsta][nsamples], double corrmatrix[nxyz][nsamples], int nproc){

    long int i, j, k;
    int ip, is;
    double stk0p, stk0s1, stk0s2;

    omp_set_num_threads(nproc);

    #pragma omp parallel for private(ip,is,stk0p,stk0s1, stk0s2,k,j)
    for(i=0;i<nxyz;i++){
       for(k=0;k<nsamples;k++){
           stk0p=0.;
           stk0s1=0.;
           stk0s2=0.;
           for(j=0;j<nsta;j++){
              ip=itp[i][j] + k;
              is=its[i][j] + k;
	      if (is < nsamples){
                  stk0p=stalta_p[j][ip] + stk0p;
                  stk0s1=stalta_s1[j][is]+ stk0s1;
                  stk0s2=stalta_s2[j][ip] + stk0s2;
	      } else {
	          stk0p=stk0p+stalta_p[j][nsamples-1];
                  stk0s1=stk0s1+stalta_s1[j][nsamples-1];
                  stk0s2=stk0s2+stalta_s2[j][nsamples-1];    
	      }
           }
       corrmatrix[i][k]=stk0p*stk0s1*stk0s2;
       }
    }
    return 0;
}

