import React from 'react'
import ReactDOM from 'react-dom/client'
import ChakraApp from './ChakraApp'
import TestChakra from './TestChakra'
import App from './App'
import './styles/globals.css'
// import './utils/cacheDebugger' // 禁用缓存调试器

// Use ChakraApp for new Chakra UI interface, fallback to original App
const isDevelopment = true // Set to false to use original App
const useTestPage = false // Set to true to test Chakra UI

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    {useTestPage ? <TestChakra /> : (isDevelopment ? <ChakraApp /> : <App />)}
  </React.StrictMode>
)