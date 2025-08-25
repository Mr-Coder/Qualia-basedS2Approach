import React, { useState } from 'react'
import {
  Box,
  VStack,
  HStack,
  Textarea,
  Button,
  Text,
  Badge,
  IconButton,
  Tooltip,
  Card,
  CardBody,
  CardHeader,
  Heading,
  useColorModeValue,
  Divider,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
} from '@chakra-ui/react'
import { 
  Send, 
  RotateCcw, 
  BookOpen, 
  Calculator, 
  Zap,
  RefreshCw 
} from 'lucide-react'

interface MathProblemInputProps {
  onSolve: (problem: string) => void
  isLoading?: boolean
  error?: string | null
}

const EXAMPLE_PROBLEMS = [
  {
    text: "If John has 5 apples and gives 2 to Mary, how many apples does he have left?",
    complexity: "L1",
    type: "Arithmetic"
  },
  {
    text: "A train travels 120 km in 2 hours. What is its average speed?",
    complexity: "L2", 
    type: "Word Problem"
  },
  {
    text: "Find the derivative of f(x) = x² + 3x - 2",
    complexity: "L3",
    type: "Calculus"
  },
  {
    text: "A projectile is launched at 45° with initial velocity 20 m/s. Find the maximum height.",
    complexity: "L3",
    type: "Physics"
  }
]

const MathProblemInput: React.FC<MathProblemInputProps> = ({
  onSolve,
  isLoading = false,
  error = null
}) => {
  const [problem, setProblem] = useState('')
  const [selectedExample, setSelectedExample] = useState<number | null>(null)
  
  const cardBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('math.200', 'math.600')

  const handleSolve = () => {
    if (problem.trim()) {
      onSolve(problem.trim())
    }
  }

  const handleExampleSelect = (example: typeof EXAMPLE_PROBLEMS[0], index: number) => {
    setProblem(example.text)
    setSelectedExample(index)
  }

  const handleClear = () => {
    setProblem('')
    setSelectedExample(null)
  }

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'L1': return 'green'
      case 'L2': return 'yellow'
      case 'L3': return 'red'
      default: return 'gray'
    }
  }

  return (
    <Card bg={cardBg} borderColor={borderColor} borderWidth="2px" variant="math">
      <CardHeader pb={4}>
        <HStack spacing={3}>
          <Box p={2} bg="math.500" borderRadius="lg" color="white">
            <Calculator size={20} />
          </Box>
          <Heading size="md" color="math.700">
            Mathematical Problem Input
          </Heading>
        </HStack>
      </CardHeader>

      <CardBody pt={0}>
        <VStack spacing={6} align="stretch">
          {/* Error Display */}
          {error && (
            <Alert status="error" borderRadius="lg">
              <AlertIcon />
              <Box>
                <AlertTitle>Solving Error!</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Box>
            </Alert>
          )}

          {/* Main Input Area */}
          <Box>
            <Text mb={3} fontWeight="semibold" color="gray.700">
              Enter your mathematical problem:
            </Text>
            <Textarea
              value={problem}
              onChange={(e) => setProblem(e.target.value)}
              placeholder="Type your mathematical problem here... 
Examples:
• Solve x + 5 = 10
• If a car travels 60 km/h for 2 hours, how far does it go?
• Find the area of a circle with radius 5 cm"
              variant="math"
              size="lg"
              minH="120px"
              resize="vertical"
              disabled={isLoading}
            />
          </Box>

          {/* Action Buttons */}
          <HStack spacing={3}>
            <Button
              leftIcon={<Send size={18} />}
              variant="math"
              size="lg"
              onClick={handleSolve}
              isLoading={isLoading}
              loadingText="Solving..."
              disabled={!problem.trim()}
              flex={1}
            >
              Solve Problem
            </Button>
            
            <Tooltip label="Clear input">
              <IconButton
                aria-label="Clear input"
                icon={<RotateCcw size={18} />}
                variant="outline"
                size="lg"
                onClick={handleClear}
                disabled={isLoading}
              />
            </Tooltip>

            <Tooltip label="Refresh">
              <IconButton
                aria-label="Refresh"
                icon={<RefreshCw size={18} />}
                variant="outline"
                size="lg"
                onClick={() => window.location.reload()}
              />
            </Tooltip>
          </HStack>

          <Divider />

          {/* Example Problems */}
          <Box>
            <HStack mb={4} spacing={2}>
              <BookOpen size={18} color="var(--chakra-colors-math-500)" />
              <Text fontWeight="semibold" color="math.600">
                Try Example Problems:
              </Text>
            </HStack>
            
            <VStack spacing={3} align="stretch">
              {EXAMPLE_PROBLEMS.map((example, index) => (
                <Card
                  key={index}
                  size="sm"
                  variant={selectedExample === index ? "elevated" : "outline"}
                  cursor="pointer"
                  onClick={() => handleExampleSelect(example, index)}
                  _hover={{
                    transform: 'translateY(-2px)',
                    boxShadow: 'lg',
                  }}
                  transition="all 0.2s"
                  bg={selectedExample === index ? 'math.50' : 'transparent'}
                  borderColor={selectedExample === index ? 'math.300' : 'gray.200'}
                >
                  <CardBody p={4}>
                    <VStack align="start" spacing={2}>
                      <HStack spacing={2} w="100%">
                        <Badge 
                          colorScheme={getComplexityColor(example.complexity)}
                          variant="solid"
                          borderRadius="full"
                        >
                          {example.complexity}
                        </Badge>
                        <Badge variant="outline" borderRadius="full">
                          {example.type}
                        </Badge>
                        {selectedExample === index && (
                          <Badge colorScheme="math" variant="solid" ml="auto">
                            <Zap size={12} />
                          </Badge>
                        )}
                      </HStack>
                      <Text fontSize="sm" color="gray.700" lineHeight="1.5">
                        {example.text}
                      </Text>
                    </VStack>
                  </CardBody>
                </Card>
              ))}
            </VStack>
          </Box>
        </VStack>
      </CardBody>
    </Card>
  )
}

export default MathProblemInput