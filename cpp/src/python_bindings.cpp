#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "math_reasoning/complexity_classifier.hpp"

namespace py = pybind11;
using namespace math_reasoning;

PYBIND11_MODULE(math_reasoning_cpp, m) {
    m.doc() = "C++ accelerated mathematical reasoning components";
    
    // ComplexityLevel enum
    py::enum_<ComplexityLevel>(m, "ComplexityLevel")
        .value("L0_EXPLICIT", ComplexityLevel::L0_EXPLICIT)
        .value("L1_SHALLOW", ComplexityLevel::L1_SHALLOW)
        .value("L2_MEDIUM", ComplexityLevel::L2_MEDIUM)
        .value("L3_DEEP", ComplexityLevel::L3_DEEP)
        .export_values();
    
    // SubLevel enum
    py::enum_<SubLevel>(m, "SubLevel")
        .value("L1_1", SubLevel::L1_1)
        .value("L1_2", SubLevel::L1_2)
        .value("L1_3", SubLevel::L1_3)
        .value("L2_1", SubLevel::L2_1)
        .value("L2_2", SubLevel::L2_2)
        .value("L2_3", SubLevel::L2_3)
        .value("L3_1", SubLevel::L3_1)
        .value("L3_2", SubLevel::L3_2)
        .value("L3_3", SubLevel::L3_3)
        .export_values();
    
    // ComplexityMetrics struct
    py::class_<ComplexityMetrics>(m, "ComplexityMetrics")
        .def(py::init<>())
        .def_readwrite("reasoning_depth", &ComplexityMetrics::reasoning_depth)
        .def_readwrite("knowledge_dependencies", &ComplexityMetrics::knowledge_dependencies)
        .def_readwrite("inference_steps", &ComplexityMetrics::inference_steps)
        .def_readwrite("variable_count", &ComplexityMetrics::variable_count)
        .def_readwrite("equation_count", &ComplexityMetrics::equation_count)
        .def_readwrite("constraint_count", &ComplexityMetrics::constraint_count)
        .def_readwrite("domain_switches", &ComplexityMetrics::domain_switches)
        .def_readwrite("abstraction_level", &ComplexityMetrics::abstraction_level)
        .def_readwrite("semantic_complexity", &ComplexityMetrics::semantic_complexity)
        .def_readwrite("computational_complexity", &ComplexityMetrics::computational_complexity);
    
    // ClassificationResult struct
    py::class_<ClassificationResult>(m, "ClassificationResult")
        .def(py::init<>())
        .def_readwrite("main_level", &ClassificationResult::main_level)
        .def_readwrite("sub_level", &ClassificationResult::sub_level)
        .def_readwrite("metrics", &ClassificationResult::metrics)
        .def_readwrite("confidence", &ClassificationResult::confidence)
        .def_readwrite("explanation", &ClassificationResult::explanation);
    
    // ComplexityClassifier class
    py::class_<ComplexityClassifier>(m, "ComplexityClassifier")
        .def(py::init<>())
        .def("classify", &ComplexityClassifier::classify,
             "Classify the complexity of a mathematical problem",
             py::arg("problem_text"))
        .def("calculate_metrics", &ComplexityClassifier::calculate_metrics,
             "Calculate complexity metrics for a problem",
             py::arg("problem_text"))
        .def("count_variables", &ComplexityClassifier::count_variables,
             "Count unique variables in the problem text",
             py::arg("text"))
        .def("count_equations", &ComplexityClassifier::count_equations,
             "Count equations in the problem text",
             py::arg("text"))
        .def("count_constraints", &ComplexityClassifier::count_constraints,
             "Count constraints in the problem text",
             py::arg("text"))
        .def("calculate_reasoning_depth", &ComplexityClassifier::calculate_reasoning_depth,
             "Calculate the depth of reasoning required",
             py::arg("text"))
        .def("estimate_inference_steps", &ComplexityClassifier::estimate_inference_steps,
             "Estimate the number of inference steps",
             py::arg("text"));
    
    // Version information
    m.attr("__version__") = "1.0.0";
}