import React from 'react'
import { ChakraProvider, ColorModeScript } from '@chakra-ui/react'
import theme from './theme'
import MathReasoningPage from './pages/MathReasoningPage'

// Error Fallback Component
const ErrorFallback: React.FC<{ error: Error }> = ({ error }) => (
  <div style={{ 
    padding: '2rem', 
    textAlign: 'center', 
    backgroundColor: '#fed7d7', 
    color: '#c53030',
    borderRadius: '8px',
    margin: '2rem'
  }}>
    <h2>Something went wrong:</h2>
    <pre style={{ marginTop: '1rem', fontSize: '0.875rem' }}>{error.message}</pre>
  </div>
)

const ChakraApp: React.FC = () => {
  return (
    <>
      <ColorModeScript initialColorMode={theme.config.initialColorMode} />
      <ChakraProvider theme={theme}>
        <MathReasoningPage />
      </ChakraProvider>
    </>
  )
}

export default ChakraApp