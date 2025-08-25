#include "math_reasoning/complexity_classifier.hpp"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <iterator>

namespace math_reasoning {

ComplexityClassifier::ComplexityClassifier() {
    load_patterns();
}

ComplexityClassifier::~ComplexityClassifier() = default;

void ComplexityClassifier::load_patterns() {
    // Load implicit patterns
    implicit_patterns_["temporal"] = {
        {"after_time", std::regex(R"(after\s+(\d+)\s+(hours?|minutes?|days?))"), 1.2},
        {"time_later", std::regex(R"((\d+)\s+(hours?|minutes?|days?)\s+later)"), 1.2},
        {"same_time", std::regex(R"(at\s+the\s+same\s+time)"), 1.0},
        {"simultaneously", std::regex(R"(simultaneously)"), 1.0}
    };
    
    implicit_patterns_["proportional"] = {
        {"proportional_to", std::regex(R"(proportional\s+to)"), 1.5},
        {"varies_with", std::regex(R"(varies\s+(?:directly|inversely)\s+with)"), 1.5},
        {"times_as", std::regex(R"(times\s+as\s+(?:much|many))"), 1.3},
        {"ratio_of", std::regex(R"(ratio\s+of)"), 1.4}
    };
    
    implicit_patterns_["comparative"] = {
        {"more_than", std::regex(R"(more\s+than)"), 1.1},
        {"less_than", std::regex(R"(less\s+than)"), 1.1},
        {"multiple_of", std::regex(R"((?:twice|thrice|half)\s+(?:as|the))"), 1.3},
        {"compared_to", std::regex(R"(compared\s+to)"), 1.2}
    };
    
    // Load reasoning patterns
    reasoning_patterns_["algebraic"] = {
        "solve for", "equation", "variable", "unknown", "express"
    };
    
    reasoning_patterns_["geometric"] = {
        "angle", "triangle", "circle", "area", "perimeter", "volume"
    };
    
    reasoning_patterns_["probabilistic"] = {
        "probability", "chance", "likelihood", "random", "expected"
    };
    
    // Load complexity indicators
    complexity_indicators_["prove"] = 1.5;
    complexity_indicators_["demonstrate"] = 1.5;
    complexity_indicators_["derive"] = 1.4;
    complexity_indicators_["calculate"] = 1.0;
    complexity_indicators_["find"] = 1.0;
    complexity_indicators_["identify"] = 0.5;
}

ClassificationResult ComplexityClassifier::classify(const std::string& problem_text) {
    // Calculate metrics
    ComplexityMetrics metrics = calculate_metrics(problem_text);
    
    // Determine complexity levels
    ComplexityLevel main_level = determine_main_level(metrics);
    SubLevel sub_level = determine_sub_level(main_level, metrics);
    
    // Calculate confidence
    double confidence = calculate_confidence(metrics);
    
    // Generate explanation
    std::string explanation = generate_explanation(main_level, sub_level, metrics);
    
    return {
        main_level,
        sub_level,
        metrics,
        confidence,
        explanation
    };
}

ComplexityMetrics ComplexityClassifier::calculate_metrics(const std::string& problem_text) {
    ComplexityMetrics metrics;
    
    // Basic counts
    metrics.variable_count = count_variables(problem_text);
    metrics.equation_count = count_equations(problem_text);
    metrics.constraint_count = count_constraints(problem_text);
    
    // Reasoning metrics
    metrics.reasoning_depth = calculate_reasoning_depth(problem_text);
    metrics.inference_steps = estimate_inference_steps(problem_text);
    
    // Calculate other metrics
    metrics.knowledge_dependencies = 0; // Simplified for now
    metrics.domain_switches = 0; // Simplified for now
    metrics.abstraction_level = calculate_abstraction_level(problem_text);
    metrics.semantic_complexity = 0.5; // Default value
    
    // Computational complexity estimate
    metrics.computational_complexity = 
        (metrics.variable_count * 0.1 + 
         metrics.equation_count * 0.2 + 
         metrics.constraint_count * 0.15) / 3.0;
    
    return metrics;
}

int ComplexityClassifier::count_variables(const std::string& text) {
    // Pattern for single letter variables
    std::regex var_pattern(R"(\b[a-zA-Z]\b(?!\w))");
    std::sregex_iterator begin(text.begin(), text.end(), var_pattern);
    std::sregex_iterator end;
    
    std::unordered_map<std::string, int> var_set;
    for (auto it = begin; it != end; ++it) {
        var_set[it->str()] = 1;
    }
    
    return static_cast<int>(var_set.size());
}

int ComplexityClassifier::count_equations(const std::string& text) {
    int count = 0;
    
    // Count equals signs
    count += std::count(text.begin(), text.end(), '=');
    
    // Count inequality signs
    std::regex inequality_pattern(R"([<>≤≥])");
    std::sregex_iterator begin(text.begin(), text.end(), inequality_pattern);
    std::sregex_iterator end;
    count += std::distance(begin, end);
    
    return count;
}

int ComplexityClassifier::count_constraints(const std::string& text) {
    std::vector<std::string> constraint_keywords = {
        "constraint", "condition", "restriction",
        "must", "should", "cannot", "limited"
    };
    
    int count = 0;
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& keyword : constraint_keywords) {
        size_t pos = 0;
        while ((pos = lower_text.find(keyword, pos)) != std::string::npos) {
            count++;
            pos += keyword.length();
        }
    }
    
    return count;
}

int ComplexityClassifier::calculate_reasoning_depth(const std::string& text) {
    int implicit_count = 0;
    
    // Count implicit pattern matches
    for (const auto& [category, patterns] : implicit_patterns_) {
        implicit_count += count_pattern_matches(text, patterns);
    }
    
    // Map to reasoning depth
    if (implicit_count == 0) return 0;
    else if (implicit_count <= 2) return 1;
    else if (implicit_count <= 5) return 2;
    else return 3;
}

int ComplexityClassifier::estimate_inference_steps(const std::string& text) {
    std::vector<std::string> connectives = {
        "therefore", "thus", "hence", "so", "then", "implies"
    };
    
    int step_count = 1; // At least one step
    
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& connective : connectives) {
        size_t pos = 0;
        while ((pos = lower_text.find(connective, pos)) != std::string::npos) {
            step_count++;
            pos += connective.length();
        }
    }
    
    // Add steps for equations and constraints
    step_count += count_equations(text) / 2;
    step_count += count_constraints(text) / 3;
    
    return step_count;
}

int ComplexityClassifier::count_pattern_matches(const std::string& text, 
                                              const std::vector<Pattern>& patterns) {
    int count = 0;
    
    for (const auto& pattern : patterns) {
        std::sregex_iterator begin(text.begin(), text.end(), pattern.regex_pattern);
        std::sregex_iterator end;
        count += std::distance(begin, end);
    }
    
    return count;
}

ComplexityLevel ComplexityClassifier::determine_main_level(const ComplexityMetrics& metrics) {
    // L0: Explicit problems
    if (metrics.reasoning_depth == 0 && metrics.inference_steps <= 1) {
        return ComplexityLevel::L0_EXPLICIT;
    }
    // L1: Shallow implicit
    else if (metrics.reasoning_depth <= 1 && metrics.inference_steps <= 3) {
        return ComplexityLevel::L1_SHALLOW;
    }
    // L2: Medium implicit
    else if (metrics.reasoning_depth <= 3 && metrics.inference_steps <= 10) {
        return ComplexityLevel::L2_MEDIUM;
    }
    // L3: Deep implicit
    else {
        return ComplexityLevel::L3_DEEP;
    }
}

SubLevel ComplexityClassifier::determine_sub_level(ComplexityLevel main_level, 
                                                 const ComplexityMetrics& metrics) {
    switch (main_level) {
        case ComplexityLevel::L0_EXPLICIT:
            return SubLevel::L1_1; // Default for L0
            
        case ComplexityLevel::L1_SHALLOW:
            if (metrics.inference_steps <= 1) return SubLevel::L1_1;
            else if (metrics.inference_steps <= 2) return SubLevel::L1_2;
            else return SubLevel::L1_3;
            
        case ComplexityLevel::L2_MEDIUM:
            if (metrics.inference_steps <= 5) return SubLevel::L2_1;
            else if (metrics.inference_steps <= 7) return SubLevel::L2_2;
            else return SubLevel::L2_3;
            
        case ComplexityLevel::L3_DEEP:
            if (metrics.inference_steps <= 15) return SubLevel::L3_1;
            else if (metrics.inference_steps <= 20) return SubLevel::L3_2;
            else return SubLevel::L3_3;
            
        default:
            return SubLevel::L1_1;
    }
}

double ComplexityClassifier::calculate_confidence(const ComplexityMetrics& metrics) {
    double confidence = 0.7; // Base confidence
    
    // Adjust based on clarity of indicators
    if (metrics.reasoning_depth > 0) confidence += 0.1;
    if (metrics.inference_steps > 5) confidence += 0.1;
    if (metrics.abstraction_level > 0.7) confidence += 0.1;
    
    return std::min(confidence, 0.95);
}

std::string ComplexityClassifier::generate_explanation(ComplexityLevel level, 
                                                     SubLevel sub_level, 
                                                     const ComplexityMetrics& metrics) {
    std::stringstream ss;
    
    // Convert enums to strings
    std::string level_str = "L" + std::to_string(static_cast<int>(level));
    std::string sub_level_str = "L" + std::to_string(static_cast<int>(level)) + "." + 
                               std::to_string(static_cast<int>(sub_level) % 3 + 1);
    
    ss << "Problem classified as " << level_str << " (" << sub_level_str << ") based on:\n";
    ss << "- Reasoning depth: " << metrics.reasoning_depth << " levels\n";
    ss << "- Estimated inference steps: " << metrics.inference_steps << "\n";
    ss << "- Variables: " << metrics.variable_count << ", ";
    ss << "Equations: " << metrics.equation_count << ", ";
    ss << "Constraints: " << metrics.constraint_count << "\n";
    
    if (metrics.abstraction_level > 0.7) {
        ss << "- High level of abstraction detected\n";
    }
    
    return ss.str();
}

double ComplexityClassifier::calculate_abstraction_level(const std::string& text) {
    std::vector<std::string> abstraction_indicators = {
        "general", "arbitrary", "any", "all",
        "prove", "show that", "demonstrate"
    };
    
    double score = 0.0;
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    for (const auto& indicator : abstraction_indicators) {
        if (lower_text.find(indicator) != std::string::npos) {
            score += 0.2;
        }
    }
    
    return std::min(score, 1.0);
}

} // namespace math_reasoning