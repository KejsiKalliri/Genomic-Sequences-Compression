#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

#include "../third_party/main_functions/bzlib.h"
#include "wrapper.h"


unsigned char* compress_block_bz2_c(
    const unsigned char* input,
    unsigned int input_len,
    int level,
    unsigned int* output_len
) {
    char* out_buf = NULL;
    unsigned int out_size;
    int rc;   // return code (bsj)

    if (output_len == NULL) {
        return NULL;
    }

    /* Conservative upper bound */
    out_size = input_len + (input_len / 100) + 600;  // para kompresimit krijohet buffer-i i outputit
    if (out_size < 1024) {
        out_size = 1024;
    }

    out_buf = (char*)malloc(out_size);  // rezervon memorien ne C
    if (out_buf == NULL) {
        return NULL;
    }

    rc = BZ2_bzBuffToBuffCompress(   // pjesa me e rendesishme eshte kjo, ketu thirret direkt funksioni i bzip2. Brenda bzip2 ndodhin fazat kryesore të compression core: BWT, MTF, RLE, Huffman Coding
        out_buf,
        &out_size,
        (char*)input,
        input_len,
        level,   /* 1..9 */
        0,       /* verbosity */
        30       /* workFactor */
    );

    if (rc != BZ_OK) {   // nese kompresimi deshton, memoria lirohet dhe Pythoni me vone merr error
        free(out_buf);
        return NULL;
    }

    *output_len = out_size;   // nese kompresimi funksionon, kthehet buffer-i i kompresuar dhe madhesia e reale e tij
    return (unsigned char*)out_buf;
}


unsigned char* decompress_block_bz2_c(  // funksioni i dekompresimit ben te kunderten
    const unsigned char* input,        // merr byte te kompresuar
    unsigned int input_len,           // gjatesine e tyre
    unsigned int expected_output_len, // madhesine qe pritet te kete outputi origjinal
    unsigned int* output_len
) {
    char* out_buf = NULL;
    unsigned int out_size;
    int rc;

    if (output_len == NULL) {
        return NULL;
    }

    out_size = expected_output_len;  // eshte e rendesishme qe expected_output_len te ruhet diku gjate kompresimit, sepse bzip2 duhet të dijë sa hapësirë të rezervojë për output-in e dekompresuar.
    out_buf = (char*)malloc(out_size);
    if (out_buf == NULL) {
        return NULL;
    }

    rc = BZ2_bzBuffToBuffDecompress(
        out_buf,
        &out_size,
        (char*)input,
        input_len,
        0, /* small */
        0  /* verbosity */
    );

    if (rc != BZ_OK) {
        free(out_buf);
        return NULL;
    }

    *output_len = out_size;
    return (unsigned char*)out_buf;
}


void free_bz2_buffer(void* p) {  // ne C memoria qe alokohet me malloc(...) duhet liruar me free(...)
    free(p);
}


/* ---------------- Python bindings ---------------- */

static PyObject* py_compress_block(PyObject* self, PyObject* args, PyObject* kwargs) {  // ky eshte funksioni qe ekspozohet ne Python si:  wrapper.compress_block(data, level=9)
    const unsigned char* input = NULL;
    Py_ssize_t input_len = 0;
    int level = 9;

    static char* kwlist[] = {"data", "level", NULL};

    unsigned char* out_buf = NULL;
    unsigned int out_len = 0;
    PyObject* result = NULL;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(   // kjo pjese lexon argumentet nga Python-i
            args,
            kwargs,
            "y#|i",   // y# do të thotë: merr një objekt bytes-like nga Python dhe jep pointer + length.
            kwlist,
            &input,
            &input_len,
            &level)) {
        return NULL;
    }

    if (level < 1 || level > 9) {
        PyErr_SetString(PyExc_ValueError, "level must be between 1 and 9");
        return NULL;
    }

    out_buf = compress_block_bz2_c(input, (unsigned int)input_len, level, &out_len);  // // pasi alokon vend ne memorie, therret kompresuesin e bzip2 dhe kthehet bufferi i kompresuar

    if (out_buf == NULL) {   // dmth qe kompresimi deshtoi
        PyErr_SetString(PyExc_RuntimeError, "BZ2_bzBuffToBuffCompress failed");
        return NULL;
    }

    result = PyBytes_FromStringAndSize((const char*)out_buf, (Py_ssize_t)out_len); // ne Python binding, pasi krijohen Python bytes
    free_bz2_buffer(out_buf);   // bufferi C lirohet
    return result;  // PyObject -> pra e kompreson dhe kthen bytes ne Python
}


static PyObject* py_decompress_block(PyObject* self, PyObject* args, PyObject* kwargs) {   // ne Python kjo ekspozohet si: wrapper.decompress_block(data, expected_output_len)
    const unsigned char* input = NULL;
    Py_ssize_t input_len = 0;
    unsigned int expected_output_len = 0;

    static char* kwlist[] = {"data", "expected_output_len", NULL};

    unsigned char* out_buf = NULL;
    unsigned int out_len = 0;
    PyObject* result = NULL;

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "y#I",
            kwlist,
            &input,
            &input_len,
            &expected_output_len)) {
        return NULL;
    }

    out_buf = decompress_block_bz2_c(input, (unsigned int)input_len, expected_output_len, &out_len); // krijohet vend ne memorie per outputin e dekompresuar, dekompresohet me bzip2 dhe kthehet bufferi i dekompresuar

    if (out_buf == NULL) {  // nqs dekompresimi deshton
        PyErr_SetString(PyExc_RuntimeError, "BZ2_bzBuffToBuffDecompress failed");
        return NULL;
    }

    result = PyBytes_FromStringAndSize((const char*)out_buf, (Py_ssize_t)out_len);
    free_bz2_buffer(out_buf);
    return result;
}


static PyMethodDef BZ2WrapperMethods[] = {  // kjo pjese i tregon Python-it cilat funksione do jenë të disponueshme.
    {
        "compress_block",   // thirret me kete emer
        (PyCFunction)py_compress_block,
        METH_VARARGS | METH_KEYWORDS,
        "Compress one block using libbzip2"
    },
    {
        "decompress_block",
        (PyCFunction)py_decompress_block,
        METH_VARARGS | METH_KEYWORDS,
        "Decompress one block using libbzip2"
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef bz2wrappermodule = {   // krijon modulin me emrin wrapper
    PyModuleDef_HEAD_INIT,
    "wrapper",
    "Thin C wrapper over libbzip2",
    -1,
    BZ2WrapperMethods    // ky modul ka keto funksione
};


PyMODINIT_FUNC PyInit_wrapper(void) {   // ky eshte funksioni qe Python therret kur ben import wrapper
    return PyModule_Create(&bz2wrappermodule);
}