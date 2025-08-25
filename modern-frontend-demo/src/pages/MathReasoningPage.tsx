import React from 'react'
import {
  Box,
  Container,
  Grid,
  GridItem,
  VStack,
  HStack,
  Heading,
  Text,
  Badge,
  Card,
  CardBody,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  useColorModeValue,
  Icon,
  Flex,
  Spacer,
  Button,
  Alert,
  AlertIcon,
} from '@chakra-ui/react'
import { 
  Calculator, 
  Brain, 
  Zap,
  GitBranch,
  BarChart3,
  Settings,
  Info
} from 'lucide-react'
import { MathProblemInput, SolutionDisplay, StepByStepPanel } from '../components/chakra'
import { useMathSolver } from '../hooks/useMathSolver'

const MathReasoningPage: React.FC = () => {
  const { solveProblem, result, isLoading, error, clearResult } = useMathSolver()
  const headerBg = useColorModeValue('white', 'gray.800')
  const pageeBg = useColorModeValue('gray.50', 'gray.900')

  return (
    <Box bg={pageeBg} minH="100vh">
      {/* Header */}
      <Box bg={headerBg} borderBottom="1px solid" borderColor="gray.200" mb={6}>
        <Container maxW="7xl" py={6}>
          <Flex align="center">
            <HStack spacing={4}>
              <Box p={3} bg="gradient" borderRadius="lg" bgGradient="linear(135deg, math.500, reasoning.500)">
                <Brain size={32} color="white" />
              </Box>
              <VStack align="start" spacing={1}>
                <Heading size="lg" color="gray.800">
                  Mathematical Reasoning System
                </Heading>
                <Text color="gray.600">
                  COT-DIR Enhanced Problem Solving with Chakra UI
                </Text>
              </VStack>
            </HStack>
            <Spacer />
            <HStack spacing={2}>
              <Badge colorScheme="math" variant="solid" px={3} py={1} borderRadius="full">
                Story 6.1
              </Badge>
              <Badge colorScheme="green" variant="outline" px={3} py={1} borderRadius="full">
                Active
              </Badge>
            </HStack>
          </Flex>
        </Container>
      </Box>

      {/* Main Content */}
      <Container maxW="7xl" pb={8}>
        <Grid 
          templateColumns={{ base: "1fr", lg: "400px 1fr 350px" }}
          gap={6}
          h={{ base: "auto", lg: "calc(100vh - 180px)" }}
        >
          {/* Left Panel - Problem Input */}
          <GridItem>
            <Box h="100%" overflowY="auto">
              <MathProblemInput 
                onSolve={solveProblem}
                isLoading={isLoading}
                error={error}
              />
            </Box>
          </GridItem>

          {/* Main Panel - Analysis & Visualization */}
          <GridItem>
            <Card h="100%" bg={headerBg}>
              <CardBody p={0}>
                <Tabs h="100%" display="flex" flexDir="column">
                  <Box px={6} pt={6} pb={2}>
                    <TabList>
                      <Tab>
                        <HStack spacing={2}>
                          <Zap size={16} />
                          <Text>Step-by-Step</Text>
                        </HStack>
                      </Tab>
                      <Tab>
                        <HStack spacing={2}>
                          <GitBranch size={16} />
                          <Text>Entity Relations</Text>
                        </HStack>
                      </Tab>
                      <Tab>
                        <HStack spacing={2}>
                          <BarChart3 size={16} />
                          <Text>Visualization</Text>
                        </HStack>
                      </Tab>
                      <Tab>
                        <HStack spacing={2}>
                          <Settings size={16} />
                          <Text>Analysis</Text>
                        </HStack>
                      </Tab>
                    </TabList>
                  </Box>

                  <TabPanels flex={1} overflow="hidden">
                    {/* Step-by-Step Tab */}
                    <TabPanel h="100%" p={6} pt={4} overflow="auto">
                      <StepByStepPanel 
                        steps={result?.steps || []}
                        isLoading={isLoading}
                      />
                    </TabPanel>

                    {/* Entity Relations Tab */}
                    <TabPanel h="100%" p={6} pt={4} overflow="auto">
                      <VStack spacing={6} align="stretch" h="100%">
                        {result ? (
                          <>
                            {/* Entities Section */}
                            <Card variant="elevated">
                              <CardBody>
                                <HStack mb={4} spacing={2}>
                                  <Calculator size={18} color="var(--chakra-colors-semantic-entity)" />
                                  <Heading size="sm">Mathematical Entities</Heading>
                                  <Badge variant="outline">{result.entities.length}</Badge>
                                </HStack>
                                <VStack align="stretch" spacing={3}>
                                  {result.entities.length > 0 ? (
                                    result.entities.map((entity, index) => (
                                      <HStack key={index} justify="space-between" p={3} bg="gray.50" borderRadius="md">
                                        <VStack align="start" spacing={1}>
                                          <Text fontWeight="semibold">{entity.text}</Text>
                                          <HStack spacing={2}>
                                            <Badge variant="entity" size="sm">{entity.type}</Badge>
                                            {entity.value && (
                                              <Badge colorScheme="blue" variant="outline" size="sm">
                                                Value: {entity.value}
                                              </Badge>
                                            )}
                                          </HStack>
                                        </VStack>
                                        <Badge 
                                          colorScheme={entity.confidence > 0.8 ? 'green' : entity.confidence > 0.6 ? 'yellow' : 'red'}
                                          variant="subtle"
                                        >
                                          {Math.round(entity.confidence * 100)}%
                                        </Badge>
                                      </HStack>
                                    ))
                                  ) : (
                                    <Text color="gray.500" textAlign="center" py={4}>
                                      No entities found
                                    </Text>
                                  )}
                                </VStack>
                              </CardBody>
                            </Card>

                            {/* Relations Section */}
                            <Card variant="elevated">
                              <CardBody>
                                <HStack mb={4} spacing={2}>
                                  <GitBranch size={18} color="var(--chakra-colors-semantic-relation)" />
                                  <Heading size="sm">Mathematical Relations</Heading>
                                  <Badge variant="outline">{result.relations.length}</Badge>
                                </HStack>
                                <VStack align="stretch" spacing={3}>
                                  {result.relations.length > 0 ? (
                                    result.relations.map((relation, index) => (
                                      <Box key={index} p={3} bg="gray.50" borderRadius="md">
                                        <VStack align="start" spacing={2}>
                                          <HStack spacing={2}>
                                            <Badge variant="relation" size="sm">{relation.type}</Badge>
                                            <Badge 
                                              colorScheme={relation.confidence > 0.8 ? 'green' : relation.confidence > 0.6 ? 'yellow' : 'red'}
                                              variant="subtle"
                                              size="sm"
                                            >
                                              {Math.round(relation.confidence * 100)}%
                                            </Badge>
                                          </HStack>
                                          <Text fontSize="sm">{relation.description}</Text>
                                          {relation.source && relation.target && (
                                            <Text fontSize="xs" color="gray.600">
                                              {relation.source} → {relation.target}
                                            </Text>
                                          )}
                                        </VStack>
                                      </Box>
                                    ))
                                  ) : (
                                    <Text color="gray.500" textAlign="center" py={4}>
                                      No relations discovered
                                    </Text>
                                  )}
                                </VStack>
                              </CardBody>
                            </Card>
                          </>
                        ) : (
                          <VStack spacing={4} py={12} color="gray.500">
                            <GitBranch size={48} />
                            <VStack spacing={2}>
                              <Text fontWeight="semibold">No Analysis Available</Text>
                              <Text fontSize="sm" textAlign="center">
                                Solve a problem to see entity and relation analysis
                              </Text>
                            </VStack>
                          </VStack>
                        )}
                      </VStack>
                    </TabPanel>

                    {/* Visualization Tab */}
                    <TabPanel h="100%" p={6} pt={4} overflow="auto">
                      <VStack spacing={4} py={12} color="gray.500">
                        <BarChart3 size={48} />
                        <VStack spacing={2}>
                          <Text fontWeight="semibold">Visualization Coming Soon</Text>
                          <Text fontSize="sm" textAlign="center">
                            Interactive graphs and charts will be available here
                          </Text>
                        </VStack>
                      </VStack>
                    </TabPanel>

                    {/* Analysis Tab */}
                    <TabPanel h="100%" p={6} pt={4} overflow="auto">
                      <VStack spacing={6} align="stretch">
                        {result ? (
                          <>
                            {/* System Info */}
                            <Alert status="info" borderRadius="lg">
                              <AlertIcon />
                              <VStack align="start" spacing={1}>
                                <Text fontWeight="bold">Analysis Complete</Text>
                                <Text fontSize="sm">
                                  Problem solved using COT-DIR methodology with {result.steps.length} reasoning steps
                                </Text>
                              </VStack>
                            </Alert>

                            {/* Detailed Metrics */}
                            <Card variant="elevated">
                              <CardBody>
                                <Heading size="sm" mb={4}>Performance Metrics</Heading>
                                <VStack align="stretch" spacing={4}>
                                  <HStack justify="space-between">
                                    <Text>Overall Confidence:</Text>
                                    <Badge 
                                      colorScheme={result.confidence > 0.8 ? 'green' : result.confidence > 0.6 ? 'yellow' : 'red'}
                                      variant="solid"
                                    >
                                      {Math.round(result.confidence * 100)}%
                                    </Badge>
                                  </HStack>
                                  
                                  <HStack justify="space-between">
                                    <Text>Complexity Level:</Text>
                                    <Badge colorScheme="purple" variant="solid">
                                      {result.complexity.level}.{result.complexity.sublevel}
                                    </Badge>
                                  </HStack>
                                  
                                  <HStack justify="space-between">
                                    <Text>Reasoning Depth:</Text>
                                    <Badge colorScheme="blue" variant="outline">
                                      {result.complexity.reasoning_depth} levels
                                    </Badge>
                                  </HStack>
                                </VStack>
                              </CardBody>
                            </Card>
                          </>
                        ) : (
                          <VStack spacing={4} py={12} color="gray.500">
                            <Settings size={48} />
                            <VStack spacing={2}>
                              <Text fontWeight="semibold">No Analysis Data</Text>
                              <Text fontSize="sm" textAlign="center">
                                Detailed analysis metrics will appear here after solving a problem
                              </Text>
                            </VStack>
                          </VStack>
                        )}
                      </VStack>
                    </TabPanel>
                  </TabPanels>
                </Tabs>
              </CardBody>
            </Card>
          </GridItem>

          {/* Right Panel - Solution Display */}
          <GridItem>
            <Box h="100%" overflowY="auto">
              <VStack spacing={4} align="stretch" h="100%">
                <SolutionDisplay 
                  result={result}
                  isLoading={isLoading}
                />
                
                {result && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={clearResult}
                    leftIcon={<Icon as={Settings} />}
                  >
                    Clear Results
                  </Button>
                )}
              </VStack>
            </Box>
          </GridItem>
        </Grid>
      </Container>
    </Box>
  )
}

export default MathReasoningPage