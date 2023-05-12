//
// Created by Gael Aglin on 2019-12-03.
//

#ifndef DL85_PY_SPLIT_PENALTY_WRAPPER_H
#define DL85_PY_SPLIT_PENALTY_WRAPPER_H

#include <Python.h>
#include "error_function.h" // cython helper file

class PySplitPenaltyWrapper {
public:
    // constructors and destructors mostly do reference counting
    PySplitPenaltyWrapper(PyObject* o): pyFunction(o) {
        Py_XINCREF(o);
    }

    PySplitPenaltyWrapper(const PySplitPenaltyWrapper& rhs): PySplitPenaltyWrapper(rhs.pyFunction) { // C++11 onwards only
    }

    PySplitPenaltyWrapper(PySplitPenaltyWrapper&& rhs): pyFunction(rhs.pyFunction) {
        rhs.pyFunction = nullptr;
    }

    // need no-arg constructor to stack allocate in Cython
    PySplitPenaltyWrapper(): PySplitPenaltyWrapper(nullptr) {
    }

    ~PySplitPenaltyWrapper() {
        Py_XDECREF(pyFunction);
    }

    PySplitPenaltyWrapper& operator=(const PySplitPenaltyWrapper& rhs) {
        PySplitPenaltyWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PySplitPenaltyWrapper& operator=(PySplitPenaltyWrapper&& rhs) {
        pyFunction = rhs.pyFunction;
        rhs.pyFunction = nullptr;
        return *this;
    }

    float operator()(int depth, int splits) {
        PyInit_error_function();
        float result = std::numeric_limits<float>::max();
        if (pyFunction != nullptr) { // nullptr check
            result = call_python_split_penalty_function(pyFunction, depth, splits); // note, no way of checking for errors until you return to Python
        }
        return result;
    }

    /*vector<float> operator()(RCover* ar) {
        int status = PyImport_AppendInittab("error_function", PyInit_error_function);
        if (status == -1) {
            vector<float> result;
            return result;
        }
        Py_Initialize();
        PyObject* module = PyImport_ImportModule("error_function");
        if (!module) {
            Py_Finalize();
            vector<float> result;
            return result;
        }

        vector<float> result;
        if (pyFunction) { // nullptr check
            result = *call_python_support_error_class_function(pyFunction, ar); // note, no way of checking for errors until you return to Python
        }

        Py_Finalize();
        return result;
    }*/

private:
    PyObject* pyFunction;
};

#endif //DL85_PY_SPLIT_PENALTY_WRAPPER_H
