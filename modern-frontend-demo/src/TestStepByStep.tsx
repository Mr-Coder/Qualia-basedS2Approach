import React from 'react'
import StepByStepPhysicalReasoning from './components/features/StepByStepPhysicalReasoning'

const TestStepByStep: React.FC = () => {
  return (
    <div style={{ padding: '20px', background: '#f5f5f5', minHeight: '100vh' }}>
      <h1 style={{ textAlign: 'center', color: '#333', marginBottom: '30px' }}>
        测试分步物性推理组件
      </h1>
      <StepByStepPhysicalReasoning />
    </div>
  )
}

export default TestStepByStep