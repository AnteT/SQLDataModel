#include <Python.h>

static PyObject *replace_single_quote(PyObject *self, PyObject *args) {
    const char *input_str;
    
    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "s", &input_str)) {
        return NULL;
    }

    int length = strlen(input_str);

    // Create a temporary buffer to store the modified string
    char *output_str = (char *)malloc((2 * length + 1) * sizeof(char));
    if (output_str == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
        return NULL;
    }

    int output_index = 0;

    // Iterate through each character in the input string
    for (int i = 0; i < length; ++i) {
        // Check if the current character is a single quote
        if (input_str[i] == '\'') {
            // Replace single quote with two single quotes in the output string
            output_str[output_index++] = '\'';
            output_str[output_index++] = '\'';
        } else {
            // Copy the character as is to the output string
            output_str[output_index++] = input_str[i];
        }
    }

    // Null-terminate the output string
    output_str[output_index] = '\0';

    // Create a Python string from the modified C string
    PyObject *result = Py_BuildValue("s", output_str);

    // Free the allocated memory
    free(output_str);

    return result;
}

static PyMethodDef methods[] = {
    {"replace_single_quote", replace_single_quote, METH_VARARGS, "Replace single quote with two single quotes"},
    {NULL, NULL, 0, NULL} // Sentinel
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "str_escape",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_str_escape(void) {
    return PyModule_Create(&module);
}