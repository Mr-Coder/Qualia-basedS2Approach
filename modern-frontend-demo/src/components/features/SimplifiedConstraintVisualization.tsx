import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui';
import { Button } from '@/components/ui';

// Simple component replacements
const Badge: React.FC<{children: React.ReactNode, className?: string, variant?: string}> = 
  ({ children, className = "", variant = "default" }) => (
    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
      variant === "outline" ? "border border-gray-300 bg-white text-gray-700" : 
      variant === "destructive" ? "bg-red-100 text-red-800" :
      "bg-blue-100 text-blue-800"
    } ${className}`}>
      {children}
    </span>
  );

const Alert: React.FC<{children: React.ReactNode, className?: string, variant?: string}> = 
  ({ children, className = "", variant = "default" }) => (
    <div className={`p-4 border rounded-lg ${
      variant === "destructive" ? "border-red-200 bg-red-50" : "border-blue-200 bg-blue-50"
    } ${className}`}>
      {children}
    </div>
  );

const AlertDescription: React.FC<{children: React.ReactNode}> = ({ children }) => (
  <div className="text-sm">{children}</div>
);

// Icons
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Calculator,
  Target,
  Clock,
  Brain
} from 'lucide-react';

interface ConstraintViolation {
  constraint_id: string;
  type: string;
  description: string;
  severity: number;
  entities: string[];
}

interface SimplifiedConstraintData {
  success: boolean;
  applicable_physics_laws: Array<{
    law_type: string;
    name: string;
    description: string;
    mathematical_form: string;
    applied: boolean;
  }>;
  generated_constraints: Array<{
    constraint_id: string;
    type: string;
    description: string;
    mathematical_expression: string;
    strength: number;
    entities: string[];
  }>;
  constraint_solution: {
    success: boolean;
    violations: ConstraintViolation[];
    solution_values: {[key: string]: any};
    confidence: number;
    confidence_adjustment: number;
  };
  constraint_guidance: string[];
  verification_steps: string[];
  reasoning_explanation: string;
  execution_time: number;
  network_metrics: {
    entities_count: number;
    constraints_count: number;
    laws_applied: number;
    satisfaction_rate: number;
  };
}

interface SimplifiedConstraintVisualizationProps {
  problemText?: string;
}

const SimplifiedConstraintVisualization: React.FC<SimplifiedConstraintVisualizationProps> = ({ 
  problemText = "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？" 
}) => {
  const [constraintData, setConstraintData] = useState<SimplifiedConstraintData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (problemText) {
      analyzeConstraints();
    }
  }, [problemText]);

  const analyzeConstraints = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/enhanced-physics-constraints', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem_text: problemText,
          enable_ortools: false,  // 简化版本不强调OR-Tools
          enable_extended_laws: false
        }),
      });

      if (!response.ok) {
        throw new Error(`API请求失败: ${response.status}`);
      }

      const data = await response.json();
      setConstraintData(data);
    } catch (error) {
      console.error('获取约束数据失败:', error);
      setError(error instanceof Error ? error.message : '未知错误');
      
      // 使用基础的演示数据
      setConstraintData(generateFallbackData());
    } finally {
      setIsLoading(false);
    }
  };

  const generateFallbackData = (): SimplifiedConstraintData => {
    return {
      success: true,
      applicable_physics_laws: [
        {
          law_type: "conservation_of_quantity",
          name: "数量守恒",
          description: "数学运算中数量应该守恒",
          mathematical_form: "总和 = 各部分之和",
          applied: true
        },
        {
          law_type: "non_negativity_law",
          name: "非负性约束",
          description: "物理数量不能为负",
          mathematical_form: "数量 ≥ 0",
          applied: true
        }
      ],
      generated_constraints: [
        {
          constraint_id: "basic_non_negative",
          type: "non_negative",
          description: "确保数值为非负",
          mathematical_expression: "x ≥ 0",
          strength: 1.0,
          entities: ["数量"]
        }
      ],
      constraint_solution: {
        success: true,
        violations: [],
        solution_values: { final_answer: 8 },
        confidence: 0.9,
        confidence_adjustment: 0.1
      },
      constraint_guidance: [
        "识别加法运算：需要将所有部分数量相加",
        "验证守恒约束：确保总和等于各部分之和"
      ],
      verification_steps: [
        "验证步骤1: 将各个部分数量相加，检查是否等于答案",
        "验证步骤2: 确认答案为正数且符合实际情况"
      ],
      reasoning_explanation: "基于问题类型(aggregation)，采用addition运算策略。",
      execution_time: 0.01,
      network_metrics: {
        entities_count: 3,
        constraints_count: 1,
        laws_applied: 2,
        satisfaction_rate: 1.0
      }
    };
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="flex items-center space-x-2">
          <Brain className="h-5 w-5 animate-spin" />
          <span>分析约束条件...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          约束分析失败: {error}
        </AlertDescription>
      </Alert>
    );
  }

  if (!constraintData) {
    return (
      <div className="text-center text-gray-500 p-8">
        暂无约束数据
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 总体状态 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            {constraintData.success ? (
              <CheckCircle className="h-5 w-5 text-green-500" />
            ) : (
              <XCircle className="h-5 w-5 text-red-500" />
            )}
            <span>约束验证结果</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {constraintData.network_metrics.laws_applied}
              </div>
              <div className="text-sm text-gray-600">应用定律</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {constraintData.network_metrics.constraints_count}
              </div>
              <div className="text-sm text-gray-600">生成约束</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {constraintData.constraint_solution.violations.length}
              </div>
              <div className="text-sm text-gray-600">约束违背</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {(constraintData.constraint_solution.confidence * 100).toFixed(0)}%
              </div>
              <div className="text-sm text-gray-600">置信度</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 应用的物理定律 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-5 w-5" />
            <span>应用的基础约束</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {constraintData.applicable_physics_laws.map((law, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-semibold">{law.name}</h4>
                  <Badge variant={law.applied ? "default" : "outline"}>
                    {law.applied ? "已应用" : "未应用"}
                  </Badge>
                </div>
                <p className="text-sm text-gray-600 mb-2">{law.description}</p>
                <div className="text-sm font-mono bg-gray-100 p-2 rounded">
                  {law.mathematical_form}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 约束指导和验证步骤 */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Brain className="h-5 w-5" />
              <span>约束指导</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {constraintData.constraint_guidance.map((guidance, index) => (
                <div key={index} className="flex items-start space-x-2">
                  <div className="w-5 h-5 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center text-xs font-bold mt-0.5">
                    {index + 1}
                  </div>
                  <div className="text-sm">{guidance}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5" />
              <span>验证步骤</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {constraintData.verification_steps.map((step, index) => (
                <div key={index} className="flex items-start space-x-2">
                  <div className="w-5 h-5 bg-green-100 text-green-800 rounded-full flex items-center justify-center text-xs font-bold mt-0.5">
                    ✓
                  </div>
                  <div className="text-sm">{step}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 约束违背情况 */}
      {constraintData.constraint_solution.violations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <span>约束违背</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {constraintData.constraint_solution.violations.map((violation, index) => (
                <Alert key={index} variant="destructive">
                  <AlertDescription>
                    <div className="font-semibold">{violation.constraint_id}</div>
                    <div>{violation.description}</div>
                    <div className="text-xs mt-1">
                      严重度: {(violation.severity * 100).toFixed(0)}% | 
                      影响实体: {violation.entities.join(', ')}
                    </div>
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* 推理解释 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Calculator className="h-5 w-5" />
            <span>推理解释</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="text-sm">{constraintData.reasoning_explanation}</p>
          </div>
          
          <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
            <div className="flex items-center space-x-2">
              <Clock className="h-4 w-4" />
              <span>执行时间: {(constraintData.execution_time * 1000).toFixed(1)}ms</span>
            </div>
            <div>
              置信度调整: {constraintData.constraint_solution.confidence_adjustment > 0 ? '+' : ''}
              {(constraintData.constraint_solution.confidence_adjustment * 100).toFixed(1)}%
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 重新分析按钮 */}
      <div className="text-center">
        <Button 
          onClick={analyzeConstraints}
          disabled={isLoading}
          className="px-6"
        >
          重新分析约束
        </Button>
      </div>
    </div>
  );
};

export default SimplifiedConstraintVisualization;