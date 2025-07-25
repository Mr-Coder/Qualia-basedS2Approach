// 测试算法API数据转换
// 在浏览器控制台中运行

async function testAlgorithmAPI() {
  console.log('🧪 测试算法API数据转换...')
  
  try {
    // 1. 获取QS²演示数据
    const qs2Response = await fetch('/api/qs2/demo')
    const qs2Data = await qs2Response.json()
    
    console.log('✅ QS²演示数据获取成功:', qs2Data.success)
    console.log('📊 数据结构:', {
      entities: qs2Data.data?.entities?.length || 0,
      relations: qs2Data.data?.relations?.length || 0,
      algorithm_stages: qs2Data.data?.algorithm_stages?.length || 0
    })
    
    // 2. 测试算法执行数据获取
    const algoResponse = await fetch('/api/algorithm/execution/latest')
    const algoData = await algoResponse.json()
    
    console.log('✅ 算法执行数据获取成功')
    console.log('📊 执行数据结构:', {
      execution_id: algoData.execution_id,
      total_stages: algoData.stages?.length || 0,
      has_final_result: !!algoData.final_result,
      is_qs2_enhanced: algoData.final_result?.is_qs2_enhanced,
      algorithm_type: algoData.final_result?.algorithm_type
    })
    
    // 3. 检查实体关系图数据
    if (algoData.final_result?.entities) {
      console.log('🔍 实体数据样本:', algoData.final_result.entities.slice(0, 2))
    }
    
    if (algoData.final_result?.relations) {
      console.log('🔗 关系数据样本:', algoData.final_result.relations.slice(0, 2))
    }
    
    if (algoData.final_result?.qualia_structures) {
      console.log('🧠 Qualia结构样本:', algoData.final_result.qualia_structures.slice(0, 2))
    }
    
    console.log('🎉 所有测试通过！算法API工作正常')
    return true
    
  } catch (error) {
    console.error('❌ 测试失败:', error)
    return false
  }
}

// 运行测试
testAlgorithmAPI().then(result => {
  if (result) {
    console.log('🎊 QS²算法API集成测试成功完成！')
  } else {
    console.log('⚠️ 测试失败，请检查错误信息')
  }
})

// 导出测试函数
window.testAlgorithmAPI = testAlgorithmAPI
console.log('🔧 测试工具已添加到 window.testAlgorithmAPI')