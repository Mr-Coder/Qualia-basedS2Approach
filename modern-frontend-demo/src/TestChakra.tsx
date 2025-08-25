import React from 'react'
import { ChakraProvider, Box, Heading, Text, Button, VStack } from '@chakra-ui/react'

const TestChakra: React.FC = () => {
  return (
    <ChakraProvider>
      <Box p={8} bg="gray.50" minH="100vh">
        <VStack spacing={4} align="center">
          <Heading>Chakra UI 测试页面</Heading>
          <Text>如果你能看到这个页面，说明Chakra UI正在工作！</Text>
          <Button colorScheme="blue" onClick={() => alert('按钮点击成功！')}>
            测试按钮
          </Button>
        </VStack>
      </Box>
    </ChakraProvider>
  )
}

export default TestChakra