import React from 'react'
import {
  Box,
  VStack,
  HStack,
  Text,
  Badge,
  Card,
  CardBody,
  CardHeader,
  Heading,
  Progress,
  Alert,
  AlertIcon,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  useColorModeValue,
  Divider,
  Icon,
  CircularProgress,
  CircularProgressLabel,
} from '@chakra-ui/react'
import { 
  CheckCircle, 
  Target, 
  TrendingUp,
  Brain,
  Award,
  Clock
} from 'lucide-react'
import { SolutionResult } from '../../hooks/useMathSolver'

interface SolutionDisplayProps {
  result: SolutionResult | null
  isLoading?: boolean
}

const SolutionDisplay: React.FC<SolutionDisplayProps> = ({
  result,
  isLoading = false
}) => {
  const cardBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('reasoning.200', 'reasoning.600')

  const getComplexityColor = (level: string) => {
    switch (level) {
      case 'L0': return 'green'
      case 'L1': return 'blue'
      case 'L2': return 'orange'
      case 'L3': return 'red'
      default: return 'gray'
    }
  }

  const formatAnswer = (answer: string | number) => {
    if (typeof answer === 'number') {
      return answer.toLocaleString()
    }
    return answer
  }

  if (isLoading) {
    return (
      <Card bg={cardBg} borderColor={borderColor} borderWidth="2px" variant="reasoning">
        <CardBody>
          <VStack spacing={6} py={8}>
            <CircularProgress isIndeterminate color="reasoning.500" size="60px" />
            <VStack spacing={2}>
              <Text fontWeight="semibold" color="reasoning.600">
                Solving Problem...
              </Text>
              <Text fontSize="sm" color="gray.600" textAlign="center">
                Applying mathematical reasoning algorithms
              </Text>
            </VStack>
          </VStack>
        </CardBody>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card bg={cardBg} borderColor="gray.200" borderWidth="1px">
        <CardBody>
          <VStack spacing={4} py={8} color="gray.500">
            <Icon as={Target} boxSize={12} />
            <VStack spacing={2}>
              <Text fontWeight="semibold">No Solution Yet</Text>
              <Text fontSize="sm" textAlign="center">
                Enter a mathematical problem to see the solution here
              </Text>
            </VStack>
          </VStack>
        </CardBody>
      </Card>
    )
  }

  return (
    <Card bg={cardBg} borderColor={borderColor} borderWidth="2px" variant="reasoning">
      <CardHeader pb={4}>
        <HStack spacing={3}>
          <Box p={2} bg="reasoning.500" borderRadius="lg" color="white">
            <CheckCircle size={20} />
          </Box>
          <Heading size="md" color="reasoning.700">
            Solution & Analysis
          </Heading>
        </HStack>
      </CardHeader>

      <CardBody pt={0}>
        <VStack spacing={6} align="stretch">
          {/* Main Answer */}
          <Alert status="success" borderRadius="lg" bg="green.50" borderColor="green.200">
            <AlertIcon as={Award} color="green.500" />
            <Box>
              <Text fontWeight="bold" color="green.800" mb={1}>
                Final Answer
              </Text>
              <Text fontSize="xl" fontWeight="bold" color="green.900">
                {formatAnswer(result.answer)}
              </Text>
            </Box>
          </Alert>

          {/* Confidence & Complexity Stats */}
          <HStack spacing={4}>
            <Stat flex={1}>
              <StatLabel color="gray.600">Confidence</StatLabel>
              <HStack>
                <CircularProgress 
                  value={result.confidence * 100} 
                  color="reasoning.500" 
                  size="40px"
                  thickness="8px"
                >
                  <CircularProgressLabel fontSize="xs">
                    {Math.round(result.confidence * 100)}%
                  </CircularProgressLabel>
                </CircularProgress>
                <StatHelpText mb={0}>
                  {result.confidence > 0.8 ? 'High' : result.confidence > 0.6 ? 'Medium' : 'Low'}
                </StatHelpText>
              </HStack>
            </Stat>

            <Divider orientation="vertical" h="60px" />

            <Stat flex={1}>
              <StatLabel color="gray.600">Complexity</StatLabel>
              <HStack spacing={2}>
                <Badge 
                  colorScheme={getComplexityColor(result.complexity.level)}
                  variant="solid"
                  fontSize="sm"
                  px={3}
                  py={1}
                  borderRadius="full"
                >
                  {result.complexity.level}.{result.complexity.sublevel}
                </Badge>
              </HStack>
              <StatHelpText>
                Depth: {result.complexity.reasoning_depth}
              </StatHelpText>
            </Stat>
          </HStack>

          <Divider />

          {/* Explanation */}
          <Box>
            <HStack mb={3} spacing={2}>
              <Brain size={18} color="var(--chakra-colors-reasoning-500)" />
              <Text fontWeight="semibold" color="reasoning.600">
                Explanation
              </Text>
            </HStack>
            <Text color="gray.700" lineHeight="1.6">
              {result.explanation}
            </Text>
          </Box>

          {/* Analysis Summary */}
          <Box>
            <HStack mb={3} spacing={2}>
              <TrendingUp size={18} color="var(--chakra-colors-reasoning-500)" />
              <Text fontWeight="semibold" color="reasoning.600">
                Analysis Summary
              </Text>
            </HStack>
            
            <VStack spacing={3} align="stretch">
              <HStack justify="space-between">
                <Text fontSize="sm" color="gray.600">Mathematical Entities:</Text>
                <Badge variant="outline" colorScheme="semantic.entity">
                  {result.entities.length} found
                </Badge>
              </HStack>
              
              <HStack justify="space-between">
                <Text fontSize="sm" color="gray.600">Relationships:</Text>
                <Badge variant="outline" colorScheme="semantic.relation">
                  {result.relations.length} discovered
                </Badge>
              </HStack>
              
              <HStack justify="space-between">
                <Text fontSize="sm" color="gray.600">Reasoning Steps:</Text>
                <Badge variant="outline" colorScheme="reasoning">
                  {result.steps.length} executed
                </Badge>
              </HStack>
            </VStack>
          </Box>

          {/* Processing Time */}
          {result.steps.length > 0 && (
            <Box>
              <HStack mb={3} spacing={2}>
                <Clock size={18} color="var(--chakra-colors-gray-500)" />
                <Text fontWeight="semibold" color="gray.600">
                  Processing Metrics
                </Text>
              </HStack>
              
              <VStack spacing={2} align="stretch">
                <HStack justify="space-between">
                  <Text fontSize="sm" color="gray.600">Steps Completed:</Text>
                  <Text fontSize="sm" fontWeight="semibold">
                    {result.steps.length}
                  </Text>
                </HStack>
                
                <Box>
                  <HStack justify="space-between" mb={1}>
                    <Text fontSize="sm" color="gray.600">Average Confidence:</Text>
                    <Text fontSize="sm" fontWeight="semibold">
                      {Math.round(
                        result.steps.reduce((acc, step) => acc + step.confidence, 0) / 
                        result.steps.length * 100
                      )}%
                    </Text>
                  </HStack>
                  <Progress 
                    value={
                      result.steps.reduce((acc, step) => acc + step.confidence, 0) / 
                      result.steps.length * 100
                    }
                    colorScheme="reasoning"
                    size="sm"
                    borderRadius="full"
                  />
                </Box>
              </VStack>
            </Box>
          )}
        </VStack>
      </CardBody>
    </Card>
  )
}

export default SolutionDisplay