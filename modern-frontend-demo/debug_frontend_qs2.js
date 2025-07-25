// 前端QS²数据获取调试脚本
// 在浏览器控制台中运行

console.log('🔍 开始调试前端QS²数据获取...')

// 1. 测试算法API
async function testAlgorithmAPI() {
  console.log('📡 测试算法API端点...')
  
  try {
    const response = await fetch('/api/algorithm/execution/latest')
    const data = await response.json()
    
    console.log('✅ 算法API响应:', data)
    console.log('🔍 QS²特征检查:', {
      is_qs2_enhanced: data.final_result?.is_qs2_enhanced,
      algorithm_type: data.final_result?.algorithm_type,
      stage_count: data.stages?.length,
      has_qualia_stages: data.stages?.some(s => s.stage_name?.includes('语义结构构建')),
      has_qs2_algorithm_state: data.stages?.some(s => s.algorithm_state?.is_qs2_enhanced)
    })
    
    return data
  } catch (error) {
    console.error('❌ 算法API测试失败:', error)
    return null
  }
}

// 2. 测试QS²专用API
async function testQS2API() {
  console.log('🧠 测试QS²专用API端点...')
  
  try {
    const [relationsResponse, qualiaResponse] = await Promise.all([
      fetch('/api/qs2/relations'),
      fetch('/api/qs2/qualia-structures')
    ])
    
    const relationsData = await relationsResponse.json()
    const qualiaData = await qualiaResponse.json()
    
    console.log('✅ QS²关系API响应:', relationsData)
    console.log('✅ Qualia结构API响应:', qualiaData)
    
    return { relationsData, qualiaData }
  } catch (error) {
    console.error('❌ QS²专用API测试失败:', error)
    return null
  }
}

// 3. 检查localStorage中的数据
function checkLocalStorage() {
  console.log('💾 检查localStorage中的QS²相关数据...')
  
  const keys = Object.keys(localStorage).filter(key => 
    key.includes('qs2') || key.includes('algorithm') || key.includes('qualia')
  )
  
  console.log('🔍 找到相关键值:', keys)
  keys.forEach(key => {
    console.log(`📦 ${key}:`, localStorage.getItem(key))
  })
}

// 4. 检查React组件状态（如果可能）
function checkReactState() {
  console.log('⚛️ 尝试检查React组件状态...')
  
  // 尝试从DOM中获取React fiber
  const diagramElement = document.querySelector('[data-testid="entity-relationship-diagram"]') || 
                         document.querySelector('.entity-relationship-diagram')
  
  if (diagramElement) {
    console.log('✅ 找到关系图组件元素:', diagramElement)
    
    // 检查元素上的React属性
    const keys = Object.keys(diagramElement)
    const reactKey = keys.find(key => key.startsWith('__reactInternalInstance') || key.startsWith('__reactFiber'))
    
    if (reactKey) {
      console.log('✅ 找到React fiber键:', reactKey)
      // 注意：这在生产环境中可能不工作
    }
  } else {
    console.log('❌ 未找到关系图组件元素')
  }
}

// 5. 模拟前端QS²检测逻辑
function simulateQS2Detection(algorithmData) {
  console.log('🧪 模拟QS²检测逻辑...')
  
  if (!algorithmData) {
    console.log('❌ 无算法数据可供检测')
    return false
  }
  
  // 检查QS²标志
  const hasQS2Flags = algorithmData.final_result?.is_qs2_enhanced || 
                      algorithmData.final_result?.algorithm_type === 'QS2_Enhanced'
  
  // 检查QS²特有的阶段
  const hasQS2Stages = algorithmData.stages?.some(stage => 
    stage.stage_name?.includes('语义结构构建') ||
    stage.stage_name?.includes('兼容性计算') ||
    stage.decisions?.some(d => d.method === 'qualia_based')
  )
  
  // 检查算法状态中的QS²标识
  const hasQS2AlgorithmState = algorithmData.stages?.some(stage =>
    stage.algorithm_state?.is_qs2_enhanced === true ||
    stage.algorithm_state?.algorithm_type === 'QS2_Enhanced'
  )
  
  const isQS2Enhanced = hasQS2Flags || hasQS2Stages || hasQS2AlgorithmState
  
  console.log('🔍 QS²检测结果:', {
    hasQS2Flags,
    hasQS2Stages, 
    hasQS2AlgorithmState,
    isQS2Enhanced
  })
  
  return isQS2Enhanced
}

// 主调试函数
async function debugQS2Frontend() {
  console.log('🚀 开始完整的QS²前端调试...')
  console.log('='.repeat(50))
  
  // 测试API
  const algorithmData = await testAlgorithmAPI()
  console.log('-'.repeat(30))
  
  const qs2ApiData = await testQS2API()
  console.log('-'.repeat(30))
  
  // 检查存储
  checkLocalStorage()
  console.log('-'.repeat(30))
  
  // 检查React状态
  checkReactState()
  console.log('-'.repeat(30))
  
  // 模拟检测
  const detectionResult = simulateQS2Detection(algorithmData)
  console.log('-'.repeat(30))
  
  // 总结
  console.log('📊 调试总结:')
  console.log('• 算法API可用:', !!algorithmData)
  console.log('• QS²专用API可用:', !!qs2ApiData)
  console.log('• QS²检测结果:', detectionResult)
  
  if (detectionResult) {
    console.log('✅ QS²应该被检测到并激活')
  } else {
    console.log('❌ QS²未被正确检测到')
  }
  
  console.log('='.repeat(50))
  
  return {
    algorithmData,
    qs2ApiData,
    detectionResult
  }
}

// 运行调试
debugQS2Frontend()

// 导出调试函数供手动调用
window.debugQS2 = {
  testAlgorithmAPI,
  testQS2API,
  checkLocalStorage,
  checkReactState,
  simulateQS2Detection,
  debugQS2Frontend
}

console.log('🔧 调试工具已添加到 window.debugQS2')
console.log('💡 你可以在控制台中调用 debugQS2.debugQS2Frontend() 重新运行完整调试')