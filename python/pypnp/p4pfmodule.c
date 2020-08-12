/**
 * @Author: Jan Brejcha <ibrejcha>
 * @Date:   2020-08-11T17:48:32+02:00
 * @Email:  ibrejcha@fit.vutbr.cz; brejchaja@gmail.com
 * @Project: LandscapeAR
 * @Last modified by:   ibrejcha
 * @Last modified time: 2020-08-12T11:44:27+02:00
 * @License: Copyright 2020 CPhoto@FIT, Brno University of Technology,
# Faculty of Information Technology,
# Božetěchova 2, 612 00, Brno, Czech Republic
#
# Redistribution and use in source code form, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# 3. Redistributions must be pursued only for non-commercial research
#    collaboration and demonstration purposes.
#
# 4. Where separate files retain their original licence terms
#    (e.g. MPL 2.0, Apache licence), these licence terms are announced, prevail
#    these terms and must be complied.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// include the prototype for original mex function
#include "p4pf.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>

/**
 * [fill2DPoint description]
 * @param  pt      [description]
 * @param  count   [description]
 * @param  arg_idx [description]
 * @param  in      [description]
 * @return         [description]
 */

int fill2DPoint(double *pt, npy_intp count, int arg_idx, char *in, npy_intp stride)
{
	if (count != 2)
	{
		char msg[128];
		sprintf(
			msg,
			"Wrong shape of an argument at index: %d, should be 2, but got: %ld",
			arg_idx, count
		);
		PyErr_SetString(PyExc_ValueError, msg);
		return -1;
	}
	while (count--)
	{
		*pt = *(double *)in;
		// printf("pt[%ld]: %f\n", count, *pt);
		pt++;
		in += stride;
	}
	return 0;
}

/*  wrapped p4p function */
static PyObject* pyp4pf(PyObject* self, PyObject* args)
{
  PyArrayObject *arrays[5];  /* holds input and output array */
  PyObject *ret;
  NpyIter *iter;
  npy_uint32 iterator_flags;

  NpyIter_IterNextFunc *iternext;

  //  parse single numpy array argument
  if (!PyArg_ParseTuple(args, "O!O!O!O!O!", &PyArray_Type, &arrays[0], &PyArray_Type, &arrays[1], &PyArray_Type, &arrays[2], &PyArray_Type, &arrays[3], &PyArray_Type, &arrays[4])) {
	  return NULL;
  }

  double glab_b[6];
  double pts_b[8];

  double *glab = &glab_b[0];
  double *pts = &pts_b[0];

  // Allocate the output
  npy_intp dims[2];
  dims[0] = 10;
  dims[1] = 10;
  ret = (PyObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  Py_INCREF(ret);
  PyArray_FILLWBYTE((PyArrayObject *) ret, 0);

  double *A = (double *)PyArray_GetPtr((PyArrayObject *)ret, (npy_intp[]){0, 0});

  // Set up and create the iterator
  iterator_flags = (NPY_ITER_ZEROSIZE_OK |
	  				NPY_ITER_READONLY |
					//
					// Enable buffering in case the input is not behaved
					// (native byte order or not aligned),
					// disabling may speed up some cases when it is known to
					// be unnecessary.
					//
					NPY_ITER_BUFFERED |
					// Manually handle innermost iteration for speed: /
					NPY_ITER_EXTERNAL_LOOP |
					NPY_ITER_GROWINNER);

  // Create the numpy iterator object:
  for (int arg_idx = 0; arg_idx < 5; ++arg_idx)
  {
	  iter = NpyIter_New(arrays[arg_idx], iterator_flags,
							  // Use input order for output and iteration
							  NPY_KEEPORDER,
							  // Allow only byte-swapping of input
							  NPY_NO_CASTING, NULL);


	  if (iter == NULL)
	  {
		  Py_DECREF(ret);
		  return NULL;
	  }

	  iternext = NpyIter_GetIterNext(iter, NULL);
	  if (iternext == NULL) {
		  NpyIter_Deallocate(iter);
		  Py_DECREF(ret);
		  return NULL;
	  }

	  if (NpyIter_GetIterSize(iter) == 0) {
		  //
		  // If there are no elements, the loop cannot be iterated.
		  // This check is necessary with NPY_ITER_ZEROSIZE_OK.
		  //
		  NpyIter_Deallocate(iter);
		  return ret;
	  }

	  // The location of the data pointer which the iterator may update
	  char **dataptr = NpyIter_GetDataPtrArray(iter);
	  // The location of the stride which the iterator may update
	  npy_intp *strideptr = NpyIter_GetInnerStrideArray(iter);
	  // The location of the inner loop size which the iterator may update
	  npy_intp *innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

	  // iterate over the arrays
	  do {
		npy_intp stride = strideptr[0];
		npy_intp count = *innersizeptr;
		// out is always contiguous, so use double
		char *in = dataptr[0];

		// The output is allocated and guaranteed contiguous (out++ works):
		assert(strideptr[1] == sizeof(double));
		//
		// For optimization it can make sense to add a check for
	 	// stride == sizeof(double) to allow the compiler to optimize for that.
		//

		if (arg_idx == 0)
		{
			if (count != 6)
			{
				PyErr_SetString(PyExc_ValueError, "Wrong shape of an argument at index: 0");
				return NULL;
			}
			while (count--)
			{
				*glab = *(double *)in;
				// printf("glab[%ld]: %f\n", count, *glab);
				glab++;
				in += stride;
			}
		}
		else
		{
			if (fill2DPoint(pts, count, arg_idx, in, stride) < 0)
			{
				return NULL;
			}
			pts += 2;
		}

	  } while (iternext(iter));

	  // Clean up and return the result
	  NpyIter_Deallocate(iter);
  }

  p4pfmex(&glab_b[0], &pts_b[0], &pts_b[2], &pts_b[4], &pts_b[6], A);

  return ret;
}


/*  define functions in module */
static PyMethodDef pyp4pfMethods[] =
{
   {"p4pf", pyp4pf, METH_VARARGS,
	   "Calculate perspective 4 point algorithm to estimate camera pose with \
	   unknown focal length."},
   {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
/* module initialization */
/* Python version 3*/
static struct PyModuleDef cModPyDem = {
  PyModuleDef_HEAD_INIT,
  "pyp4pf", "Some documentation",
  -1,
  pyp4pfMethods
};
PyMODINIT_FUNC PyInit_pyp4pf(void) {
  PyObject *module;
  module = PyModule_Create(&cModPyDem);
  if(module==NULL) return NULL;
  /* IMPORTANT: this must be called */
  import_array();
  if (PyErr_Occurred()) return NULL;
  return module;
}

#else
/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC initpyp4pf(void) {
  PyObject *module;
  module = Py_InitModule("pypnp", pyp4p4pfMethods);
  if(module==NULL) return;
  /* IMPORTANT: this must be called */
  import_array();
  return;
}

#endif
