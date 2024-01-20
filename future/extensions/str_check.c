#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* str_check(PyObject* self, PyObject* args) {
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
                // printf("failed on switch case 1 at c = %c\n", c);
                Py_RETURN_FALSE;
            case '"':
                // printf("failed on switch case 2 at c = %c\n", c);
                Py_RETURN_FALSE;
            case '\\':
                // printf("failed on switch case 3 at c = %c\n", c);
                Py_RETURN_FALSE;
            case '\n':
                // printf("failed on switch case 3 at c = %c\n", c);
                Py_RETURN_FALSE;                    
            default:
                // printf("default branch triggered for c = %c\n", c);
                continue;
        }
    }
    Py_RETURN_TRUE;            
}

static PyMethodDef MyExtensionMethods[] = {
    {"check", str_check, METH_VARARGS, "Print a hello message."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef str_check_module = {
    PyModuleDef_HEAD_INIT,
    "str_check",    /* name of module */
    NULL,           /* module documentation, may be NULL */
    -1,             /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    MyExtensionMethods
};

PyMODINIT_FUNC PyInit_str_check(void) {
    return PyModule_Create(&str_check_module);
}