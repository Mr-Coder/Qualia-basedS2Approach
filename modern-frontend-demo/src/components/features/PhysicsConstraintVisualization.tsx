import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Alert, AlertDescription } from '@/components/ui/Alert';
import { Progress } from '@/components/ui/Progress';
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Zap, 
  Network, 
  Calculator,
  Eye,
  AlertTriangle,
  Target,
  Clock
} from 'lucide-react';

interface PhysicsLaw {
  law_type: string;
  name: string;
  description: string;
  mathematical_form: string;
  priority: number;
}

interface PhysicsConstraint {
  constraint_id: string;
  type: string;
  description: string;
  mathematical_expression: string;
  strength: number;
  entities: string[];
}

interface ConstraintViolation {
  constraint_id: string;
  type: string;
  severity: number;
  description: string;
  suggested_fix: string;
}

interface ConstraintSolution {
  success: boolean;
  satisfied_constraints: string[];
  violations: ConstraintViolation[];
  solution_values: Record<string, any>;
  confidence: number;
}

interface PhysicsConstraintData {
  success: boolean;
  applicable_physics_laws: PhysicsLaw[];
  generated_constraints: PhysicsConstraint[];
  constraint_solution: ConstraintSolution;
  physical_validation: {
    is_physically_consistent: boolean;
    consistency_score: number;
    law_validations: Array<{
      law_type: string;
      law_name: string;
      satisfied: boolean;
      confidence: number;
      validation_details: string;
    }>;
    global_consistency_checks: Array<{
      check_type: string;
      passed: boolean;
      details: string;
    }>;
  };
  physics_explanation: {
    physics_reasoning: Array<{
      law_name: string;
      description: string;
      mathematical_form: string;
      application_reason: string;
    }>;
    constraint_explanations: Array<{
      constraint_id: string;
      description: string;
      mathematical_expression: string;
      strength: number;
      justification: string;
    }>;
    law_applications: Array<{
      law_type: string;
      application_context: string;
      expected_outcome: string;
    }>;
    solution_justification: string;
  };
  execution_time: number;
  network_metrics: {
    entities_count: number;
    constraints_count: number;
    laws_applied: number;
    satisfaction_rate: number;
  };
}

interface PhysicsConstraintVisualizationProps {
  constraintData: PhysicsConstraintData | null;
  isLoading?: boolean;
}

const PhysicsConstraintVisualization: React.FC<PhysicsConstraintVisualizationProps> = ({
  constraintData,
  isLoading = false
}) => {
  const [selectedConstraint, setSelectedConstraint] = useState<string | null>(null);
  const [selectedLaw, setSelectedLaw] = useState<string | null>(null);

  const getStatusIcon = (success: boolean) => {
    return success ? (
      <CheckCircle className="h-5 w-5 text-green-500" />
    ) : (
      <XCircle className="h-5 w-5 text-red-500" />
    );
  };

  const getSeverityColor = (severity: number) => {
    if (severity >= 0.8) return 'bg-red-500';
    if (severity >= 0.5) return 'bg-orange-500';
    return 'bg-yellow-500';
  };

  const getPriorityColor = (priority: number) => {
    if (priority >= 0.9) return 'bg-purple-500';
    if (priority >= 0.8) return 'bg-blue-500';
    return 'bg-green-500';
  };

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            物理约束网络分析中...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center p-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!constraintData) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            物理约束网络
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              暂无约束数据。请先解决数学问题以生成物理约束分析。
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Network className="h-5 w-5" />
          物理约束传播网络
          {getStatusIcon(constraintData.success)}
        </CardTitle>
        <div className="flex gap-4 text-sm text-gray-600">
          <span className="flex items-center gap-1">
            <Target className="h-4 w-4" />
            满足率: {(constraintData.network_metrics.satisfaction_rate * 100).toFixed(1)}%
          </span>
          <span className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            执行时间: {constraintData.execution_time.toFixed(3)}s
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">概览</TabsTrigger>
            <TabsTrigger value="laws">物理定律</TabsTrigger>
            <TabsTrigger value="constraints">约束条件</TabsTrigger>
            <TabsTrigger value="violations">违规检测</TabsTrigger>
            <TabsTrigger value="explanation">推理解释</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">应用定律</p>
                      <p className="text-2xl font-bold">{constraintData.network_metrics.laws_applied}</p>
                    </div>
                    <Zap className="h-8 w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">生成约束</p>
                      <p className="text-2xl font-bold">{constraintData.network_metrics.constraints_count}</p>
                    </div>
                    <Network className="h-8 w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">满足约束</p>
                      <p className="text-2xl font-bold">{constraintData.constraint_solution.satisfied_constraints.length}</p>
                    </div>
                    <CheckCircle className="h-8 w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">约束违背</p>
                      <p className="text-2xl font-bold text-red-500">{constraintData.constraint_solution.violations.length}</p>
                    </div>
                    <AlertTriangle className="h-8 w-8 text-red-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">约束满足率</span>
                  <span className="text-sm text-gray-600">
                    {(constraintData.network_metrics.satisfaction_rate * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={constraintData.network_metrics.satisfaction_rate * 100} className="h-2" />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">物理一致性</span>
                  <span className="text-sm text-gray-600">
                    {(constraintData.physical_validation.consistency_score * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={constraintData.physical_validation.consistency_score * 100} className="h-2" />
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium">求解置信度</span>
                  <span className="text-sm text-gray-600">
                    {(constraintData.constraint_solution.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={constraintData.constraint_solution.confidence * 100} className="h-2" />
              </div>
            </div>
          </TabsContent>

          <TabsContent value="laws" className="mt-4">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold mb-4">应用的物理定律 ({constraintData.applicable_physics_laws.length})</h3>
              
              {constraintData.applicable_physics_laws.map((law, index) => (
                <Card 
                  key={index} 
                  className={`cursor-pointer transition-all duration-200 ${
                    selectedLaw === law.law_type ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => setSelectedLaw(selectedLaw === law.law_type ? null : law.law_type)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">{law.name}</CardTitle>
                      <Badge 
                        className={`${getPriorityColor(law.priority)} text-white`}
                      >
                        优先级: {law.priority}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-600 mb-3">{law.description}</p>
                    <div className="bg-gray-50 p-3 rounded-lg">
                      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">数学表达</span>
                      <p className="text-sm font-mono mt-1">{law.mathematical_form}</p>
                    </div>
                    
                    {selectedLaw === law.law_type && (
                      <div className="mt-4 pt-4 border-t">
                        <h4 className="font-semibold mb-2">验证结果</h4>
                        {constraintData.physical_validation.law_validations
                          .filter(validation => validation.law_type === law.law_type)
                          .map((validation, idx) => (
                            <div key={idx} className="flex items-center gap-2">
                              {validation.satisfied ? (
                                <CheckCircle className="h-4 w-4 text-green-500" />
                              ) : (
                                <XCircle className="h-4 w-4 text-red-500" />
                              )}
                              <span className="text-sm">{validation.validation_details}</span>
                              <Badge variant="outline">
                                置信度: {(validation.confidence * 100).toFixed(1)}%
                              </Badge>
                            </div>
                          ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="constraints" className="mt-4">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold mb-4">生成的约束条件 ({constraintData.generated_constraints.length})</h3>
              
              {constraintData.generated_constraints.map((constraint, index) => (
                <Card 
                  key={index}
                  className={`cursor-pointer transition-all duration-200 ${
                    selectedConstraint === constraint.constraint_id ? 'ring-2 ring-blue-500' : ''
                  }`}
                  onClick={() => setSelectedConstraint(
                    selectedConstraint === constraint.constraint_id ? null : constraint.constraint_id
                  )}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">{constraint.description}</CardTitle>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{constraint.type}</Badge>
                        <Badge className="bg-blue-500 text-white">
                          强度: {constraint.strength}
                        </Badge>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-gray-50 p-3 rounded-lg mb-3">
                      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">数学表达式</span>
                      <p className="text-sm font-mono mt-1">{constraint.mathematical_expression}</p>
                    </div>
                    
                    <div className="flex flex-wrap gap-1">
                      <span className="text-xs text-gray-500">涉及实体:</span>
                      {constraint.entities.map((entity, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          {entity}
                        </Badge>
                      ))}
                    </div>

                    {selectedConstraint === constraint.constraint_id && (
                      <div className="mt-4 pt-4 border-t">
                        <h4 className="font-semibold mb-2">约束状态</h4>
                        {constraintData.constraint_solution.satisfied_constraints.includes(constraint.constraint_id) ? (
                          <div className="flex items-center gap-2 text-green-600">
                            <CheckCircle className="h-4 w-4" />
                            <span className="text-sm">约束已满足</span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2 text-red-600">
                            <XCircle className="h-4 w-4" />
                            <span className="text-sm">约束未满足或存在冲突</span>
                          </div>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="violations" className="mt-4">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold mb-4">约束违规检测</h3>
              
              {constraintData.constraint_solution.violations.length === 0 ? (
                <Alert>
                  <CheckCircle className="h-4 w-4" />
                  <AlertDescription>
                    🎉 所有约束都已满足，没有发现违规情况！
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="space-y-3">
                  {constraintData.constraint_solution.violations.map((violation, index) => (
                    <Alert key={index} className="border-l-4 border-red-500">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="font-semibold">约束ID: {violation.constraint_id}</span>
                            <Badge 
                              className={`${getSeverityColor(violation.severity)} text-white`}
                            >
                              严重度: {(violation.severity * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <p className="text-sm"><strong>违规类型:</strong> {violation.type}</p>
                          <p className="text-sm"><strong>问题描述:</strong> {violation.description}</p>
                          <div className="bg-blue-50 p-3 rounded-lg">
                            <p className="text-sm font-semibold text-blue-800">建议修复:</p>
                            <p className="text-sm text-blue-700">{violation.suggested_fix}</p>
                          </div>
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              )}

              <div className="mt-6">
                <h4 className="font-semibold mb-3">全局一致性检查</h4>
                <div className="space-y-2">
                  {constraintData.physical_validation.global_consistency_checks.map((check, index) => (
                    <div key={index} className="flex items-center gap-2 p-3 rounded-lg bg-gray-50">
                      {check.passed ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <div>
                        <span className="text-sm font-medium">{check.check_type}</span>
                        <p className="text-xs text-gray-600">{check.details}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="explanation" className="mt-4">
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-4">物理推理解释</h3>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-sm">{constraintData.physics_explanation.solution_justification}</p>
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-3">定律应用分析</h4>
                <div className="space-y-3">
                  {constraintData.physics_explanation.physics_reasoning.map((reasoning, index) => (
                    <Card key={index}>
                      <CardContent className="p-4">
                        <h5 className="font-medium mb-2">{reasoning.law_name}</h5>
                        <p className="text-sm text-gray-600 mb-2">{reasoning.description}</p>
                        <div className="bg-gray-50 p-2 rounded text-xs font-mono mb-2">
                          {reasoning.mathematical_form}
                        </div>
                        <p className="text-sm text-blue-600">{reasoning.application_reason}</p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="font-semibold mb-3">约束生成依据</h4>
                <div className="space-y-2">
                  {constraintData.physics_explanation.constraint_explanations.map((explanation, index) => (
                    <div key={index} className="p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">{explanation.constraint_id}</span>
                        <Badge variant="outline">强度: {explanation.strength}</Badge>
                      </div>
                      <p className="text-sm text-gray-600 mb-1">{explanation.description}</p>
                      <p className="text-xs font-mono text-gray-500 mb-1">{explanation.mathematical_expression}</p>
                      <p className="text-xs text-blue-600">{explanation.justification}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default PhysicsConstraintVisualization;