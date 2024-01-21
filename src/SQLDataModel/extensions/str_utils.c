#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* str_utils(PyObject* self, PyObject* args) {
    const char *s;
    Py_ssize_t count, ii;
    char c;
    if (0 == PyArg_ParseTuple (args, "s#", &s, &count)) {
            return NULL;
    }
    for (ii = 0; ii < count; ii++) {
        c = s[ii];
        // check for invalid chars ', ", \, \n, maybe also: ’
        switch (c) {
            case '\'':
                Py_RETURN_FALSE;
            case '"':
                Py_RETURN_FALSE;
            case '\\':
                Py_RETURN_FALSE;
            case '\n':
                Py_RETURN_FALSE;                    
            default:
                continue;
        }
    }
    Py_RETURN_TRUE;            
}

static PyMethodDef MyExtensionMethods[] = {
    {"str_is_valid", str_utils, METH_VARARGS, "Print a hello message."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef str_utils_module = {
    PyModuleDef_HEAD_INIT,
    "str_utils",    /* name of module */
    NULL,           /* module documentation, may be NULL */
    -1,             /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    MyExtensionMethods
};

PyMODINIT_FUNC PyInit_str_utils(void) {
    return PyModule_Create(&str_utils_module);
}