import React, { useState } from 'react'
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
  Timeline,
  TimelineItem,
  TimelineConnector,
  TimelineContent,
  TimelineIcon,
  TimelineSeparator,
  Button,
  Collapse,
  IconButton,
  Tooltip,
  useColorModeValue,
  Alert,
  AlertIcon,
} from '@chakra-ui/react'
import { 
  ChevronDown, 
  ChevronRight, 
  Eye, 
  EyeOff,
  Play,
  CheckCircle,
  Circle,
  Zap,
  Brain,
  Target,
  Lightbulb
} from 'lucide-react'
import { ReasoningStep } from '../../hooks/useMathSolver'

interface StepByStepPanelProps {
  steps: ReasoningStep[]
  isLoading?: boolean
  currentStep?: number
}

const StepByStepPanel: React.FC<StepByStepPanelProps> = ({
  steps,
  isLoading = false,
  currentStep = -1
}) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())
  const [showDetails, setShowDetails] = useState(true)
  
  const cardBg = useColorModeValue('white', 'gray.800')
  const timelineBg = useColorModeValue('gray.50', 'gray.700')

  const toggleStep = (stepId: string) => {
    const newExpanded = new Set(expandedSteps)
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId)
    } else {
      newExpanded.add(stepId)
    }
    setExpandedSteps(newExpanded)
  }

  const getStepIcon = (type: ReasoningStep['type'], isActive: boolean) => {
    const iconProps = { size: 16 }
    
    switch (type) {
      case 'entity_recognition':
        return <Eye {...iconProps} />
      case 'relation_discovery':
        return <Zap {...iconProps} />
      case 'equation_building':
        return <Target {...iconProps} />
      case 'solving':
        return <Brain {...iconProps} />
      case 'validation':
        return <CheckCircle {...iconProps} />
      default:
        return <Circle {...iconProps} />
    }
  }

  const getStepColor = (type: ReasoningStep['type']) => {
    switch (type) {
      case 'entity_recognition': return 'blue'
      case 'relation_discovery': return 'purple'
      case 'equation_building': return 'orange'
      case 'solving': return 'green'
      case 'validation': return 'teal'
      default: return 'gray'
    }
  }

  const getStepTitle = (type: ReasoningStep['type']) => {
    switch (type) {
      case 'entity_recognition': return 'Entity Recognition'
      case 'relation_discovery': return 'Relation Discovery'
      case 'equation_building': return 'Equation Building'
      case 'solving': return 'Problem Solving'
      case 'validation': return 'Solution Validation'
      default: return 'Processing Step'
    }
  }

  if (isLoading) {
    return (
      <Card bg={cardBg}>
        <CardHeader>
          <Heading size="md" color="math.700">
            Reasoning Steps
          </Heading>
        </CardHeader>
        <CardBody>
          <VStack spacing={4}>
            {[1, 2, 3].map((i) => (
              <HStack key={i} w="100%" spacing={4} opacity={0.6}>
                <Box w={8} h={8} bg="gray.200" borderRadius="full" />
                <VStack align="start" spacing={1} flex={1}>
                  <Box h={4} bg="gray.200" borderRadius="md" w="60%" />
                  <Box h={3} bg="gray.100" borderRadius="md" w="80%" />
                </VStack>
              </HStack>
            ))}
          </VStack>
        </CardBody>
      </Card>
    )
  }

  if (steps.length === 0) {
    return (
      <Card bg={cardBg}>
        <CardHeader>
          <HStack justify="space-between">
            <Heading size="md" color="math.700">
              Reasoning Steps
            </Heading>
          </HStack>
        </CardHeader>
        <CardBody>
          <VStack spacing={4} py={8} color="gray.500">
            <Lightbulb size={48} />
            <VStack spacing={2}>
              <Text fontWeight="semibold">No Steps Available</Text>
              <Text fontSize="sm" textAlign="center">
                Solve a problem to see the step-by-step reasoning process
              </Text>
            </VStack>
          </VStack>
        </CardBody>
      </Card>
    )
  }

  return (
    <Card bg={cardBg}>
      <CardHeader pb={4}>
        <HStack justify="space-between">
          <Heading size="md" color="math.700">
            Reasoning Steps ({steps.length})
          </Heading>
          <HStack spacing={2}>
            <Tooltip label={showDetails ? "Hide Details" : "Show Details"}>
              <IconButton
                aria-label="Toggle details"
                icon={showDetails ? <EyeOff size={18} /> : <Eye size={18} />}
                size="sm"
                variant="ghost"
                onClick={() => setShowDetails(!showDetails)}
              />
            </Tooltip>
          </HStack>
        </HStack>
      </CardHeader>

      <CardBody pt={0}>
        <VStack spacing={0} align="stretch">
          {steps.map((step, index) => {
            const isExpanded = expandedSteps.has(step.id)
            const isActive = index === currentStep
            const isCompleted = index < currentStep || currentStep === -1
            const stepColor = getStepColor(step.type)

            return (
              <Box key={step.id} position="relative">
                {/* Timeline connector */}
                {index < steps.length - 1 && (
                  <Box
                    position="absolute"
                    left="16px"
                    top="40px"
                    bottom="-8px"
                    w="2px"
                    bg={isCompleted ? `${stepColor}.300` : 'gray.200'}
                    zIndex={0}
                  />
                )}

                <HStack spacing={4} py={3} align="start">
                  {/* Step Icon */}
                  <Box
                    w={8}
                    h={8}
                    borderRadius="full"
                    bg={isCompleted ? `${stepColor}.500` : isActive ? `${stepColor}.200` : 'gray.200'}
                    color={isCompleted ? 'white' : isActive ? `${stepColor}.600` : 'gray.500'}
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    zIndex={1}
                    border="3px solid"
                    borderColor={cardBg}
                  >
                    {getStepIcon(step.type, isActive)}
                  </Box>

                  {/* Step Content */}
                  <VStack align="start" spacing={2} flex={1} minW={0}>
                    <HStack w="100%" justify="space-between" align="start">
                      <VStack align="start" spacing={1} flex={1} minW={0}>
                        <HStack spacing={2} wrap="wrap">
                          <Badge colorScheme={stepColor} variant="subtle">
                            Step {step.step}
                          </Badge>
                          <Badge variant="outline" fontSize="xs">
                            {getStepTitle(step.type)}
                          </Badge>
                          <Badge 
                            colorScheme={step.confidence > 0.8 ? 'green' : step.confidence > 0.6 ? 'yellow' : 'red'}
                            variant="subtle"
                            fontSize="xs"
                          >
                            {Math.round(step.confidence * 100)}%
                          </Badge>
                        </HStack>
                        
                        <Text fontWeight="medium" color="gray.800" fontSize="sm">
                          {step.description}
                        </Text>
                      </VStack>

                      {step.details && (
                        <IconButton
                          aria-label="Toggle step details"
                          icon={isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                          size="sm"
                          variant="ghost"
                          onClick={() => toggleStep(step.id)}
                        />
                      )}
                    </HStack>

                    {/* Step Details */}
                    {step.details && (
                      <Collapse in={isExpanded && showDetails} animateOpacity>
                        <Box 
                          p={4} 
                          bg={timelineBg} 
                          borderRadius="lg" 
                          borderLeft="4px solid"
                          borderLeftColor={`${stepColor}.400`}
                          w="100%"
                        >
                          <VStack align="start" spacing={3}>
                            {step.details.entities && step.details.entities.length > 0 && (
                              <Box>
                                <Text fontSize="sm" fontWeight="semibold" color="gray.700" mb={2}>
                                  Entities Found:
                                </Text>
                                <HStack spacing={2} wrap="wrap">
                                  {step.details.entities.map((entity, i) => (
                                    <Badge key={i} variant="entity">
                                      {entity.text || entity.id}
                                    </Badge>
                                  ))}
                                </HStack>
                              </Box>
                            )}

                            {step.details.relations && step.details.relations.length > 0 && (
                              <Box>
                                <Text fontSize="sm" fontWeight="semibold" color="gray.700" mb={2}>
                                  Relations Discovered:
                                </Text>
                                <VStack align="start" spacing={1}>
                                  {step.details.relations.map((relation, i) => (
                                    <Text key={i} fontSize="xs" color="gray.600">
                                      • {relation.description || `${relation.source} → ${relation.target}`}
                                    </Text>
                                  ))}
                                </VStack>
                              </Box>
                            )}

                            {step.details.equations && step.details.equations.length > 0 && (
                              <Box>
                                <Text fontSize="sm" fontWeight="semibold" color="gray.700" mb={2}>
                                  Equations:
                                </Text>
                                <VStack align="start" spacing={1}>
                                  {step.details.equations.map((equation, i) => (
                                    <Text key={i} fontSize="sm" fontFamily="mono" bg="white" p={2} borderRadius="md">
                                      {equation}
                                    </Text>
                                  ))}
                                </VStack>
                              </Box>
                            )}

                            {step.details.calculations && step.details.calculations.length > 0 && (
                              <Box>
                                <Text fontSize="sm" fontWeight="semibold" color="gray.700" mb={2}>
                                  Calculations:
                                </Text>
                                <VStack align="start" spacing={1}>
                                  {step.details.calculations.map((calc, i) => (
                                    <Text key={i} fontSize="sm" color="gray.600">
                                      • {calc}
                                    </Text>
                                  ))}
                                </VStack>
                              </Box>
                            )}
                          </VStack>
                        </Box>
                      </Collapse>
                    )}
                  </VStack>
                </HStack>
              </Box>
            )
          })}
        </VStack>

        {/* Summary */}
        {steps.length > 0 && (
          <Alert status="info" mt={6} borderRadius="lg" bg="blue.50" borderColor="blue.200">
            <AlertIcon color="blue.500" />
            <Box>
              <Text fontWeight="semibold" color="blue.800" mb={1}>
                Reasoning Complete
              </Text>
              <Text fontSize="sm" color="blue.700">
                Processed {steps.length} steps with average confidence of{' '}
                {Math.round(steps.reduce((acc, step) => acc + step.confidence, 0) / steps.length * 100)}%
              </Text>
            </Box>
          </Alert>
        )}
      </CardBody>
    </Card>
  )
}

export default StepByStepPanel