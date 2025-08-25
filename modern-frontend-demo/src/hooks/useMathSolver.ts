import { useState } from 'react'
import { useToast } from '@chakra-ui/react'

export interface MathEntity {
  id: string
  text: string
  type: 'number' | 'variable' | 'operator' | 'object' | 'person'
  value?: number
  confidence: number
}

export interface MathRelation {
  id: string
  source: string
  target: string
  type: 'arithmetic' | 'comparison' | 'ownership' | 'temporal'
  description: string
  confidence: number
}

export interface ReasoningStep {
  id: string
  step: number
  type: 'entity_recognition' | 'relation_discovery' | 'equation_building' | 'solving' | 'validation'
  description: string
  confidence: number
  timestamp: number
  details?: {
    entities?: MathEntity[]
    relations?: MathRelation[]
    equations?: string[]
    calculations?: string[]
  }
}

export interface SolutionResult {
  answer: string | number
  confidence: number
  explanation: string
  steps: ReasoningStep[]
  entities: MathEntity[]
  relations: MathRelation[]
  complexity: {
    level: 'L0' | 'L1' | 'L2' | 'L3'
    sublevel: string
    reasoning_depth: number
  }
}

export interface ProblemSolveRequest {
  problem: string
  options?: {
    show_steps: boolean
    include_visualization: boolean
    complexity_analysis: boolean
  }
}

export const useMathSolver = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [currentProblem, setCurrentProblem] = useState<string>('')
  const [result, setResult] = useState<SolutionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const toast = useToast()

  const solveProblem = async (problem: string, options?: ProblemSolveRequest['options']) => {
    if (!problem.trim()) {
      toast({
        title: "Empty Problem",
        description: "Please enter a mathematical problem to solve",
        status: "warning",
        duration: 3000,
        isClosable: true,
      })
      return
    }

    setIsLoading(true)
    setError(null)
    setCurrentProblem(problem)

    try {
      const response = await fetch('http://localhost:5001/api/solve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          problem,
          options: {
            show_steps: true,
            include_visualization: true,
            complexity_analysis: true,
            ...options,
          },
        }),
      })

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }

      // Transform API response to match our interface
      const transformedResult: SolutionResult = {
        answer: data.final_answer || data.answer || 'No solution found',
        confidence: data.confidence || 0.8,
        explanation: data.explanation || 'Solution found using mathematical reasoning',
        steps: data.reasoning_steps?.map((step: any, index: number) => ({
          id: `step-${index}`,
          step: index + 1,
          type: step.type || 'solving',
          description: step.description || step.step_description || `Step ${index + 1}`,
          confidence: step.confidence || 0.8,
          timestamp: Date.now() + index * 100,
          details: step.details
        })) || [],
        entities: data.entities?.map((entity: any, index: number) => ({
          id: entity.id || `entity-${index}`,
          text: entity.text || entity.name || '',
          type: entity.type || 'number',
          value: entity.value,
          confidence: entity.confidence || 0.8
        })) || [],
        relations: data.relations?.map((relation: any, index: number) => ({
          id: relation.id || `relation-${index}`,
          source: relation.source || '',
          target: relation.target || '',
          type: relation.type || 'arithmetic',
          description: relation.description || '',
          confidence: relation.confidence || 0.8
        })) || [],
        complexity: {
          level: data.complexity?.level || 'L1',
          sublevel: data.complexity?.sublevel || 'L1.1',
          reasoning_depth: data.complexity?.reasoning_depth || 1
        }
      }

      setResult(transformedResult)
      
      toast({
        title: "Problem Solved!",
        description: "Mathematical reasoning completed successfully",
        status: "success",
        duration: 3000,
        isClosable: true,
      })

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred'
      setError(errorMessage)
      
      toast({
        title: "Solving Failed",
        description: errorMessage,
        status: "error",
        duration: 5000,
        isClosable: true,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const clearResult = () => {
    setResult(null)
    setError(null)
    setCurrentProblem('')
  }

  const retryProblem = () => {
    if (currentProblem) {
      solveProblem(currentProblem)
    }
  }

  return {
    solveProblem,
    clearResult,
    retryProblem,
    isLoading,
    result,
    error,
    currentProblem,
  }
}