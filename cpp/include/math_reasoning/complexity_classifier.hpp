#ifndef MATH_REASONING_COMPLEXITY_CLASSIFIER_HPP
#define MATH_REASONING_COMPLEXITY_CLASSIFIER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <regex>

namespace math_reasoning {

// Complexity levels enum
enum class ComplexityLevel {
    L0_EXPLICIT = 0,
    L1_SHALLOW = 1,
    L2_MEDIUM = 2,
    L3_DEEP = 3
};

// Sub-levels for fine-grained classification
enum class SubLevel {
    L1_1, L1_2, L1_3,
    L2_1, L2_2, L2_3,
    L3_1, L3_2, L3_3
};

// Structure to hold complexity metrics
struct ComplexityMetrics {
    int reasoning_depth;
    int knowledge_dependencies;
    int inference_steps;
    int variable_count;
    int equation_count;
    int constraint_count;
    int domain_switches;
    double abstraction_level;
    double semantic_complexity;
    double computational_complexity;
};

// Classification result
struct ClassificationResult {
    ComplexityLevel main_level;
    SubLevel sub_level;
    ComplexityMetrics metrics;
    double confidence;
    std::string explanation;
};

// Pattern type for implicit relation detection
struct Pattern {
    std::string name;
    std::regex regex_pattern;
    double weight;
};

class ComplexityClassifier {
public:
    ComplexityClassifier();
    ~ComplexityClassifier();
    
    // Main classification method
    ClassificationResult classify(const std::string& problem_text);
    
    // Component methods (public for testing)
    ComplexityMetrics calculate_metrics(const std::string& problem_text);
    int count_variables(const std::string& text);
    int count_equations(const std::string& text);
    int count_constraints(const std::string& text);
    int calculate_reasoning_depth(const std::string& text);
    int estimate_inference_steps(const std::string& text);
    
private:
    // Pattern collections
    std::unordered_map<std::string, std::vector<Pattern>> implicit_patterns_;
    std::unordered_map<std::string, std::vector<std::string>> reasoning_patterns_;
    std::unordered_map<std::string, double> complexity_indicators_;
    
    // Internal methods
    void load_patterns();
    ComplexityLevel determine_main_level(const ComplexityMetrics& metrics);
    SubLevel determine_sub_level(ComplexityLevel main_level, const ComplexityMetrics& metrics);
    double calculate_confidence(const ComplexityMetrics& metrics);
    std::string generate_explanation(ComplexityLevel level, SubLevel sub_level, const ComplexityMetrics& metrics);
    
    // Helper methods
    int count_pattern_matches(const std::string& text, const std::vector<Pattern>& patterns);
    std::vector<std::string> extract_words(const std::string& text);
    double calculate_abstraction_level(const std::string& text);
};

} // namespace math_reasoning

#endif // MATH_REASONING_COMPLEXITY_CLASSIFIER_HPP