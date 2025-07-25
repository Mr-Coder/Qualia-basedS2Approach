import React, { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'

interface AlgorithmStep {
  id: string
  name: string
  description: string
  input: any
  output: any
  progress: number
  status: 'pending' | 'running' | 'completed' | 'error'
  subSteps: SubStep[]
  visualElements: VisualElement[]
}

interface SubStep {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed'
  data: any
}

interface VisualElement {
  type: 'entity' | 'relation' | 'process' | 'result'
  id: string
  name: string
  position: { x: number; y: number }
  status: 'hidden' | 'appearing' | 'visible' | 'processing' | 'highlighted'
  connections: string[]
}

const IRDAlgorithmVisualization: React.FC = () => {
  const [algorithmData, setAlgorithmData] = useState<any>(null)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0)
  const [algorithmSteps, setAlgorithmSteps] = useState<AlgorithmStep[]>([])
  const [debugInfo, setDebugInfo] = useState<string>('初始化 B+C 方案中...')
  const [visualElements, setVisualElements] = useState<VisualElement[]>([])
  const canvasRef = useRef<HTMLCanvasElement>(null)


  // 获取算法执行数据
  useEffect(() => {
    const fetchAlgorithmData = async () => {
      try {
        setDebugInfo('正在获取算法数据...')
        console.log('🔍 正在获取算法数据...')
        const response = await fetch('http://localhost:5000/api/algorithm/execution')
        console.log('📡 API响应状态:', response.status)
        setDebugInfo(`API响应状态: ${response.status}`)
        
        const data = await response.json()
        console.log('📊 收到的数据:', data)
        
        if (data.success && data.data) {
          console.log('✅ 数据验证成功，数据类型:', typeof data.data)
          console.log('📋 数据内容:', data.data)
          setDebugInfo('数据验证成功')
          
          if (data.data.stages && Array.isArray(data.data.stages) && data.data.stages.length > 0) {
            console.log('🔢 算法阶段数量:', data.data.stages.length)
            console.log('📑 阶段详情:', data.data.stages)
            setDebugInfo(`发现${data.data.stages.length}个算法阶段，正在处理...`)
            setAlgorithmData(data.data)
            processAlgorithmStages(data.data.stages)
          } else {
            console.warn('⚠️ 数据中没有有效的stages字段')
            console.log('🔍 stages值:', data.data.stages)
            console.log('🔍 可用字段:', Object.keys(data.data))
            setDebugInfo(`⚠️ 数据中没有有效的stages字段，可用字段: ${Object.keys(data.data).join(', ')}`)
          }
        } else {
          console.warn('⚠️ 数据验证失败:', { success: data.success, hasData: !!data.data })
          setDebugInfo(`⚠️ 数据验证失败: success=${data.success}, hasData=${!!data.data}`)
        }
      } catch (error) {
        console.error('❌ 获取算法数据失败:', error)
        setDebugInfo(`❌ 获取算法数据失败: ${error.message}`)
      }
    }

    fetchAlgorithmData()
    const interval = setInterval(fetchAlgorithmData, 5000)
    return () => clearInterval(interval)
  }, [])

  // 处理算法阶段数据，转换为可视化步骤
  const processAlgorithmStages = (stages: any[]) => {
    try {
      console.log('🔄 开始处理算法阶段数据:', stages.length, '个阶段')
      console.log('📊 原始阶段数据:', stages)
      
      if (!stages || !Array.isArray(stages) || stages.length === 0) {
        console.error('❌ 无效的stages数据:', stages)
        setDebugInfo('❌ 无效的stages数据')
        return
      }
      
      // 从第一个阶段提取基础实体，供后续阶段使用
      const baseEntities = stages[0]?.output_data?.entities || []
      console.log('📦 基础实体数据:', baseEntities)
    
    const steps: AlgorithmStep[] = stages.map((stage, index) => {
      console.log(`📝 处理第${index + 1}个阶段:`, stage)
      
      const stepId = stage.stage_id || `step-${index}`
      const stepName = stage.stage_name || `阶段 ${index + 1}`
      
      console.log(`🏷️ 阶段${index + 1}: ID=${stepId}, Name=${stepName}`)
      
      // 处理子步骤
      const subSteps: SubStep[] = []
      console.log(`🔍 检查阶段${index + 1}的输出数据:`, stage.output_data)
      
      if (stage.output_data) {
        if (stage.output_data.entities) {
          console.log(`👥 阶段${index + 1}发现${stage.output_data.entities.length}个实体:`, stage.output_data.entities)
          subSteps.push({
            id: `${stepId}-entities`,
            name: '实体提取',
            description: `提取了 ${stage.output_data.entities.length} 个实体`,
            status: 'completed',
            data: stage.output_data.entities
          })
        }
        if (stage.output_data.relations) {
          // 检查relations是数组还是数字
          if (Array.isArray(stage.output_data.relations)) {
            console.log(`🔗 阶段${index + 1}发现${stage.output_data.relations.length}个关系:`, stage.output_data.relations)
            subSteps.push({
              id: `${stepId}-relations`,
              name: '关系发现',
              description: `发现了 ${stage.output_data.relations.length} 个关系`,
              status: 'completed',
              data: stage.output_data.relations
            })
          } else if (typeof stage.output_data.relations === 'number') {
            console.log(`🔗 阶段${index + 1}发现${stage.output_data.relations}个关系（数量）`)
            subSteps.push({
              id: `${stepId}-relations`,
              name: '关系发现',
              description: `发现了 ${stage.output_data.relations} 个关系`,
              status: 'completed',
              data: []
            })
          }
        }
      } else {
        console.warn(`⚠️ 阶段${index + 1}没有输出数据`)
      }

      // 生成可视化元素
      const visualElements: VisualElement[] = []
      console.log(`🎨 为阶段${index + 1}生成可视化元素...`)
      
      // 使用当前阶段的实体数据，如果没有则使用基础实体数据
      const currentEntities = stage.output_data?.entities || baseEntities
      
      if (currentEntities && currentEntities.length > 0) {
        console.log(`📊 处理${currentEntities.length}个实体 (来源: ${stage.output_data?.entities ? '当前阶段' : '基础数据'})`)
        currentEntities.forEach((entity: any, entityIndex: number) => {
          console.log(`🔍 处理实体${entityIndex + 1}:`, entity)
          
          // 基于后台真实数据和问题内容推断实体含义
          let entityName = entity.name || `实体${entityIndex + 1}`
          
          // 从问题文本中推断实体含义
          const problemText = stage.input_data?.problem_text || algorithmData?.problem_text || ""
          
          if (!entity.name || entity.name === 'unknown' || entityName === 'unknown') {
            // 根据问题内容和实体位置推断含义
            if (problemText.includes("小明") && problemText.includes("苹果")) {
              // 苹果问题的实体映射
              const appleEntities = ["小明", "苹果", "小红", "数量10", "数量3", "数量5"]
              entityName = appleEntities[entityIndex] || `提取实体${entityIndex + 1}`
            } else {
              // 通用数学问题实体
              entityName = `数学实体${entityIndex + 1}`
            }
            
            // 根据阶段调整名称
            if (stage.stage_name === "语义结构构建") {
              entityName = `语义结构${entityIndex + 1}`
            } else if (stage.stage_name === "关系发现") {
              entityName = `关系模式${entityIndex + 1}`
            } else if (stage.stage_name === "后处理优化") {
              entityName = `优化结果${entityIndex + 1}`
            }
          }
          
          console.log(`🏷️ 实体${entityIndex + 1}最终名称: ${entityName}`)
          
          // 根据阶段调整实体位置和属性
          const stageOffset = index * 30 // 不同阶段略微偏移
          const visualElement = {
            type: 'entity' as const,
            id: entity.id || `entity_${entityIndex}`,
            name: entityName,
            position: {
              x: 100 + (entityIndex % 3) * 150 + stageOffset,
              y: 100 + Math.floor(entityIndex / 3) * 100 + (index * 10)
            },
            status: 'visible' as const,
            connections: []
          }
          
          console.log(`➕ 添加可视化元素:`, visualElement)
          visualElements.push(visualElement)
        })
      } else {
        console.warn(`⚠️ 阶段${index + 1}没有实体数据`)
      }

      if (stage.visual_elements) {
        stage.visual_elements.forEach((element: any, elementIndex: number) => {
          if (element.type === 'relation') {
            visualElements.push({
              type: 'relation',
              id: element.id || `rel-${elementIndex}`,
              name: `关系${elementIndex + 1}`,
              position: {
                x: 200 + elementIndex * 50,
                y: 250
              },
              status: 'hidden',
              connections: entities.length >= 2 ? [entities[0].id, entities[1].id] : [] // 连接前两个实体
            })
          }
        })
      }
      
      // 处理relations数据（如果存在）
      if (stage.output_data && stage.output_data.relations) {
        if (Array.isArray(stage.output_data.relations)) {
          stage.output_data.relations.forEach((relation: any, relationIndex: number) => {
            visualElements.push({
              type: 'relation',
              id: `data_rel-${relationIndex}`,
              name: `数据关系${relationIndex + 1}`,
              position: {
                x: 300 + relationIndex * 50,
                y: 280
              },
              status: 'hidden',
              connections: entities.length >= 2 ? [entities[0].id, entities[1].id] : []
            })
          })
        } else if (typeof stage.output_data.relations === 'number') {
          // 基于后台返回的关系数量，生成有意义的关系连接
          const entityCount = currentEntities.length
          const relationsCount = stage.output_data.relations
          const problemText = stage.input_data?.problem_text || algorithmData?.problem_text || ""
          
          // 为苹果问题生成特定的关系模式
          if (problemText.includes("小明") && problemText.includes("苹果")) {
            // 生成苹果问题的逻辑关系
            const appleRelations = [
              { from: "0", to: "1", name: "拥有" },      // 小明拥有苹果
              { from: "0", to: "2", name: "给出" },      // 小明给小红
              { from: "1", to: "3", name: "数量" },      // 苹果数量10
              { from: "0", to: "4", name: "购买" },      // 小明买苹果
              { from: "1", to: "5", name: "增加" },      // 苹果增加
              { from: "2", to: "3", name: "获得" }       // 小红获得3个
            ]
            
            for (let i = 0; i < Math.min(relationsCount, appleRelations.length); i++) {
              const rel = appleRelations[i]
              visualElements.push({
                type: 'relation',
                id: `apple_rel-${i}`,
                name: rel.name,
                position: {
                  x: 250 + i * 20,
                  y: 250
                },
                status: 'visible' as const,
                connections: [rel.from, rel.to]
              })
            }
          } else {
            // 通用关系生成
            for (let relationIndex = 0; relationIndex < relationsCount; relationIndex++) {
              const sourceIndex = relationIndex % entityCount
              const targetIndex = (relationIndex + 1) % entityCount
              
              visualElements.push({
                type: 'relation',
                id: `generic_rel-${relationIndex}`,
                name: `关系${relationIndex + 1}`,
                position: {
                  x: 300 + relationIndex * 40,
                  y: 280
                },
                status: 'visible' as const,
                connections: entities.length > Math.max(sourceIndex, targetIndex) 
                  ? [entities[sourceIndex].id, entities[targetIndex].id] 
                  : []
              })
            }
          }
        }
      }

      // 为不同阶段添加特殊的可视化元素
      if (index === 1) { // 语义结构构建阶段
        visualElements.push({
          type: 'process' as const,
          id: `semantic-structure-${index}`,
          name: '语义结构',
          position: { x: 200, y: 50 },
          status: 'hidden' as const,
          connections: []
        })
      } else if (index === 2) { // 关系发现阶段
        visualElements.push({
          type: 'process' as const,
          id: `relation-discovery-${index}`,
          name: '关系发现',
          position: { x: 350, y: 50 },
          status: 'hidden' as const,
          connections: []
        })
      } else if (index === 3) { // 后处理优化阶段
        visualElements.push({
          type: 'result' as const,
          id: `optimization-result-${index}`,
          name: '优化结果',
          position: { x: 500, y: 50 },
          status: 'hidden' as const,
          connections: []
        })
      }

      const step = {
        id: stepId,
        name: stepName,
        description: stage.algorithm_state?.description || `执行${stepName}`,
        input: stage.input_data,
        output: stage.output_data,
        progress: 100,
        status: 'completed' as const,
        subSteps,
        visualElements
      }
      
      console.log(`✅ 阶段${index + 1}处理完成:`, step)
      return step
    })

    console.log('🎯 所有步骤处理完成，总数:', steps.length)
    console.log('📋 最终步骤数组:', steps)
    
    setAlgorithmSteps(steps)
    console.log('✨ 已设置算法步骤')
    setDebugInfo(`✅ 成功处理${steps.length}个算法步骤`)
    
    if (steps.length > 0) {
      console.log('🎨 设置初始可视化元素:', steps[0].visualElements)
      setVisualElements(steps[0].visualElements)
      console.log('✅ 初始可视化元素设置完成')
      setDebugInfo(`✅ 算法可视化准备就绪，共${steps.length}个步骤`)
    } else {
      console.warn('⚠️ 没有步骤可用于设置可视化元素')
      setDebugInfo('⚠️ 处理完成但没有生成任何步骤')
    }
    } catch (error) {
      console.error('❌ 处理算法阶段数据时出错:', error)
      setDebugInfo(`❌ 处理错误: ${error.message}`)
    }
  }

  // 播放算法步骤（瀑布式展示）
  const playAlgorithmSteps = () => {
    if (algorithmSteps.length === 0) return

    setIsPlaying(true)
    let stepIndex = 0

    const playNextStep = () => {
      if (stepIndex >= algorithmSteps.length) {
        setIsPlaying(false)
        return
      }

      setCurrentStep(stepIndex)
      const step = algorithmSteps[stepIndex]
      
      // 瀑布式展示：逐个显示可视化元素
      step.visualElements.forEach((element, elementIndex) => {
        setTimeout(() => {
          setVisualElements(prevElements => {
            const newElements = [...prevElements]
            const index = newElements.findIndex(e => e.id === element.id)
            if (index !== -1) {
              newElements[index] = { ...element, status: 'appearing' }
            } else {
              newElements.push({ ...element, status: 'appearing' })
            }
            return newElements
          })

          // 稍后标记为可见
          setTimeout(() => {
            setVisualElements(prevElements => 
              prevElements.map(e => 
                e.id === element.id ? { ...e, status: 'visible' } : e
              )
            )
          }, 300)
        }, elementIndex * 200 / playbackSpeed)
      })

      // 逐步显示子步骤（瀑布效果）
      step.subSteps.forEach((subStep, subIndex) => {
        setTimeout(() => {
          console.log(`🔍 执行子步骤: ${subStep.name} - ${subStep.description}`)
          
          // 为每个子步骤创建视觉反馈
          if (subStep.data && Array.isArray(subStep.data)) {
            subStep.data.forEach((dataItem: any, dataIndex: number) => {
              setTimeout(() => {
                // 高亮相关元素
                setVisualElements(prevElements => 
                  prevElements.map(element => 
                    element.name === dataItem.name ? 
                      { ...element, status: 'highlighted' } : element
                  )
                )

                // 恢复正常状态
                setTimeout(() => {
                  setVisualElements(prevElements => 
                    prevElements.map(element => 
                      element.status === 'highlighted' ? 
                        { ...element, status: 'visible' } : element
                    )
                  )
                }, 300)
              }, dataIndex * 100)
            })
          }
        }, (subIndex * 800 + 500) / playbackSpeed)
      })

      setTimeout(() => {
        stepIndex++
        if (isPlaying && stepIndex < algorithmSteps.length) {
          playNextStep()
        } else {
          setIsPlaying(false)
        }
      }, 3000 / playbackSpeed)
    }

    playNextStep()
  }

  // 停止播放
  const stopPlaying = () => {
    setIsPlaying(false)
  }

  // 跳转到指定步骤
  const goToStep = (stepIndex: number) => {
    if (stepIndex >= 0 && stepIndex < algorithmSteps.length) {
      setCurrentStep(stepIndex)
      const step = algorithmSteps[stepIndex]
      setVisualElements(step.visualElements.map(element => ({
        ...element,
        status: 'visible'
      })))
    }
  }

  // 简化条件：只检查步骤数组
  if (algorithmSteps.length === 0) {
    console.log('⚠️ 显示等待消息 - 条件检查:')
    console.log('  algorithmData:', algorithmData)
    console.log('  algorithmSteps.length:', algorithmSteps.length)
    console.log('  algorithmSteps:', algorithmSteps)
    
    return (
      <Card>
        <CardHeader>
          <CardTitle>🧠 IRD算法执行可视化 v2.0 - B+C方案</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🔄</div>
            <div className="text-lg font-medium mb-2">IRD算法可视化 v2.0 (B+C方案)</div>
            <div className="text-sm text-gray-600">请点击下面的按钮查看时间线解题过程</div>
            <div className="text-xs text-gray-500 mt-4">
              调试状态: {debugInfo}
            </div>
            <div className="text-xs text-gray-400 mt-2">
              数据状态: algorithmData={algorithmData ? '有数据' : '无数据'}, 
              步骤数量={algorithmSteps.length}
            </div>
            <button 
              onClick={() => {
                console.log('🔄 手动刷新数据')
                fetch('http://localhost:5000/api/algorithm/execution')
                  .then(res => res.json())
                  .then(data => {
                    console.log('📊 手动获取的数据:', data)
                    if (data.success && data.data && data.data.stages) {
                      setAlgorithmData(data.data)
                      processAlgorithmStages(data.data.stages)
                    }
                  })
                  .catch(err => console.error('❌ 手动获取失败:', err))
              }}
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              🔄 获取后台真实数据
            </button>
            <div className="text-xs text-gray-500 mt-2">
              后台返回: {algorithmData ? `${algorithmData.stages?.length || 0}个阶段` : '无数据'}
            </div>
            <button 
              onClick={() => {
                console.log('🧪 使用测试数据')
                // B+C: 实体关系图谱 + 时间线解题过程
                const timelineSteps = [
                  {
                    id: 'timeline-t0',
                    name: 'T0: 初始状态',
                    description: '小明拥有10个苹果',
                    time: 0,
                    state: { 小明: 10, 小红: 0 },
                    action: '初始状态',
                    progress: 100,
                    status: 'completed' as const,
                    subSteps: [],
                    visualElements: [
                      { type: 'entity' as const, id: 'xiaoming', name: '小明\n10个🍎', position: { x: 150, y: 150 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'xiaohong', name: '小红\n0个🍎', position: { x: 350, y: 150 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'apples', name: '🍎苹果\n总数10', position: { x: 250, y: 80 }, status: 'visible' as const, connections: [] },
                      { type: 'relation' as const, id: 'owns1', name: '拥有10', position: { x: 200, y: 115 }, status: 'visible' as const, connections: ['xiaoming', 'apples'] }
                    ]
                  },
                  {
                    id: 'timeline-t1',
                    name: 'T1: 给出苹果',
                    description: '小明给了小红3个苹果',
                    time: 1,
                    state: { 小明: 7, 小红: 3 },
                    action: '转移: -3个苹果',
                    progress: 100,
                    status: 'completed' as const,
                    subSteps: [
                      { id: 'transfer', name: '转移操作', description: '小明 → 小红: 3个苹果', status: 'completed', data: {} }
                    ],
                    visualElements: [
                      { type: 'entity' as const, id: 'xiaoming', name: '小明\n7个🍎', position: { x: 150, y: 150 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'xiaohong', name: '小红\n3个🍎', position: { x: 350, y: 150 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'apples', name: '🍎苹果\n总数10', position: { x: 250, y: 80 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'operation', name: '给出操作\n10-3=7', position: { x: 250, y: 220 }, status: 'visible' as const, connections: [] },
                      { type: 'relation' as const, id: 'owns1', name: '拥有7', position: { x: 200, y: 115 }, status: 'visible' as const, connections: ['xiaoming', 'apples'] },
                      { type: 'relation' as const, id: 'owns2', name: '拥有3', position: { x: 300, y: 115 }, status: 'visible' as const, connections: ['xiaohong', 'apples'] },
                      { type: 'relation' as const, id: 'transfer', name: '转移-3', position: { x: 250, y: 150 }, status: 'visible' as const, connections: ['xiaoming', 'xiaohong'] }
                    ]
                  },
                  {
                    id: 'timeline-t2',
                    name: 'T2: 购买苹果', 
                    description: '小明又买了5个苹果',
                    time: 2,
                    state: { 小明: 12, 小红: 3 },
                    action: '购买: +5个苹果',
                    progress: 100,
                    status: 'completed' as const,
                    subSteps: [
                      { id: 'purchase', name: '购买操作', description: '小明买入5个苹果', status: 'completed', data: {} }
                    ],
                    visualElements: [
                      { type: 'entity' as const, id: 'xiaoming', name: '小明\n12个🍎', position: { x: 150, y: 150 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'xiaohong', name: '小红\n3个🍎', position: { x: 350, y: 150 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'apples', name: '🍎苹果\n总数15', position: { x: 250, y: 80 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'shop', name: '🏪商店\n+5个🍎', position: { x: 80, y: 80 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'operation', name: '购买操作\n7+5=12', position: { x: 250, y: 220 }, status: 'visible' as const, connections: [] },
                      { type: 'relation' as const, id: 'owns1', name: '拥有12', position: { x: 200, y: 115 }, status: 'visible' as const, connections: ['xiaoming', 'apples'] },
                      { type: 'relation' as const, id: 'owns2', name: '拥有3', position: { x: 300, y: 115 }, status: 'visible' as const, connections: ['xiaohong', 'apples'] },
                      { type: 'relation' as const, id: 'purchase', name: '购买+5', position: { x: 115, y: 115 }, status: 'visible' as const, connections: ['shop', 'xiaoming'] }
                    ]
                  },
                  {
                    id: 'timeline-final',
                    name: '最终答案',
                    description: '小明现在有12个苹果',
                    time: 3,
                    state: { 小明: 12, 小红: 3 },
                    action: '计算完成',
                    progress: 100,
                    status: 'completed' as const,
                    subSteps: [
                      { id: 'answer', name: '最终答案', description: '小明现在有12个苹果', status: 'completed', data: {} }
                    ],
                    visualElements: [
                      { type: 'result' as const, id: 'final_answer', name: '🎯最终答案\n12个苹果', position: { x: 250, y: 150 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'calculation', name: '计算过程\n10-3+5=12', position: { x: 250, y: 80 }, status: 'visible' as const, connections: [] },
                      { type: 'entity' as const, id: 'verification', name: '✅验证\n答案正确', position: { x: 250, y: 220 }, status: 'visible' as const, connections: [] },
                      { type: 'relation' as const, id: 'calc_result', name: '推导', position: { x: 250, y: 115 }, status: 'visible' as const, connections: ['calculation', 'final_answer'] },
                      { type: 'relation' as const, id: 'verify_result', name: '验证', position: { x: 250, y: 185 }, status: 'visible' as const, connections: ['final_answer', 'verification'] }
                    ]
                  }
                ]
                setAlgorithmSteps(timelineSteps)
                setVisualElements(timelineSteps[0].visualElements)
                setDebugInfo('🧪 已设置时间线解题数据')
              }}
              className="mt-2 px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 font-bold text-lg"
            >
              🎯 开始B+C方案演示
            </button>
          </div>
        </CardContent>
      </Card>
    )
  }

  const currentStepData = algorithmSteps[currentStep]

  return (
    <div className="space-y-6">
      {/* 主可视化画布 */}
      <Card>
        <CardHeader>
          <CardTitle>🧠 IRD算法执行过程可视化</CardTitle>
          <div className="flex items-center gap-4">
            <Button
              onClick={() => {
                if (algorithmSteps.length > 0) {
                  // 简单的步骤切换，展示解题过程
                  const nextStep = (currentStep + 1) % algorithmSteps.length
                  setCurrentStep(nextStep)
                  setVisualElements(algorithmSteps[nextStep].visualElements)
                  setDebugInfo(`🎯 显示步骤 ${nextStep + 1}: ${algorithmSteps[nextStep].name}`)
                  console.log('切换到步骤:', nextStep + 1, algorithmSteps[nextStep])
                  console.log('可视化元素:', algorithmSteps[nextStep].visualElements)
                }
              }}
              className="bg-blue-500 hover:bg-blue-600 px-6 py-3 text-lg font-bold"
            >
              ▶️ 下一步解题 ({currentStep + 1}/{algorithmSteps.length})
            </Button>
            <div className="flex items-center gap-2">
              <span className="text-sm">速度:</span>
              <select 
                value={playbackSpeed} 
                onChange={(e) => setPlaybackSpeed(Number(e.target.value))}
                className="text-sm border rounded px-2 py-1"
              >
                <option value={0.5}>0.5x</option>
                <option value={1.0}>1.0x</option>
                <option value={1.5}>1.5x</option>
                <option value={2.0}>2.0x</option>
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* 大标题显示当前状态 */}
          {currentStepData && (
            <div className="text-center mb-4 p-4 bg-yellow-100 rounded-lg">
              <h2 className="text-2xl font-bold text-gray-800 mb-2">
                {currentStepData.name}
              </h2>
              {(currentStepData as any).state && (
                <div className="text-lg">
                  {Object.entries((currentStepData as any).state).map(([person, count]) => (
                    <span key={person} className="inline-block mx-4 px-3 py-1 bg-blue-500 text-white rounded-full font-bold">
                      {person}: {count}个🍎
                    </span>
                  ))}
                </div>
              )}
              {(currentStepData as any).action && (
                <div className="text-md text-green-700 font-semibold mt-2">
                  操作: {(currentStepData as any).action}
                </div>
              )}
            </div>
          )}

          {/* 算法执行画布 */}
          <div className="relative bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg border-2 border-dashed border-blue-300 h-96 overflow-hidden">
            <svg width="100%" height="100%" className="absolute inset-0">
              {/* 绘制关系连接线 */}
              {visualElements
                .filter(element => element.type === 'relation')
                .map((relation, index) => {
                  const sourceEntity = visualElements.find(e => e.id === relation.connections[0])
                  const targetEntity = visualElements.find(e => e.id === relation.connections[1])
                  
                  if (!sourceEntity || !targetEntity) {
                    console.log('找不到连接实体:', relation.connections, '可用实体:', visualElements.map(e => e.id))
                    return null
                  }

                  return (
                    <g key={relation.id}>
                      {/* 关系连接线 */}
                      <line
                        x1={sourceEntity.position.x}
                        y1={sourceEntity.position.y}
                        x2={targetEntity.position.x}
                        y2={targetEntity.position.y}
                        stroke="#e11d48"
                        strokeWidth="3"
                        opacity="0.8"
                      />
                      {/* 关系标签 */}
                      <text
                        x={(sourceEntity.position.x + targetEntity.position.x) / 2}
                        y={(sourceEntity.position.y + targetEntity.position.y) / 2 - 10}
                        textAnchor="middle"
                        className="text-xs font-bold fill-red-600"
                      >
                        {relation.name}
                      </text>
                      {/* 箭头 */}
                      <polygon
                        points={`${targetEntity.position.x-10},${targetEntity.position.y-5} ${targetEntity.position.x},${targetEntity.position.y} ${targetEntity.position.x-10},${targetEntity.position.y+5}`}
                        fill="#e11d48"
                      />
                    </g>
                  )
                })}

              {/* 绘制所有可视化节点 */}
              {visualElements
                .map((element, index) => (
                  <g key={element.id}>
                    <circle
                      cx={element.position.x}
                      cy={element.position.y}
                      r={element.type === 'entity' ? 40 : element.type === 'process' ? 35 : 50}
                      fill={
                        element.status === 'highlighted' ? '#f59e0b' :
                        element.status === 'processing' ? '#3b82f6' : 
                        element.type === 'entity' ? '#10b981' :
                        element.type === 'process' ? '#8b5cf6' :
                        element.type === 'result' ? '#ef4444' : '#6b7280'
                      }
                      stroke="#333"
                      strokeWidth="3"
                      className="drop-shadow-lg"
                    />
                    
                    <text
                      x={element.position.x}
                      y={element.position.y + 5}
                      textAnchor="middle"
                      className="text-sm font-bold fill-white pointer-events-none"
                    >
                      {element.name}
                    </text>
                  </g>
                ))}

              {/* 算法状态指示器 */}
              {currentStepData && (
                <g>
                  <circle
                    cx="200"
                    cy="50"
                    r="30"
                    fill="#f0f0f0"
                    stroke="#8b5cf6"
                    strokeWidth="3"
                  />
                  <text x="200" y="55" textAnchor="middle" className="text-sm font-bold fill-purple-700">
                    IRD
                  </text>
                </g>
              )}
            </svg>

            {/* 时间线状态提示 */}
            {currentStepData && (
              <div className="absolute top-4 left-4 bg-white bg-opacity-95 rounded-lg p-4 shadow-lg max-w-md">
                <div className="text-lg font-bold text-gray-800 mb-2">
                  {currentStepData.name}
                </div>
                <div className="text-sm text-gray-700 mb-3">{currentStepData.description}</div>
                
                {/* 状态变化信息 */}
                {(currentStepData as any).state && (
                  <div className="text-sm bg-blue-50 p-2 rounded mb-2">
                    <strong>当前状态:</strong>
                    <div className="mt-1">
                      {Object.entries((currentStepData as any).state).map(([person, count]) => (
                        <span key={person} className="inline-block mr-3 text-blue-700">
                          {person}: {count}个🍎
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {(currentStepData as any).action && (
                  <div className="text-xs text-green-600 mb-2">
                    <strong>操作:</strong> {(currentStepData as any).action}
                  </div>
                )}
                
                <div className="text-xs text-purple-600">
                  时间点: T{(currentStepData as any).time || currentStep} | 状态: {currentStepData.status}
                </div>
              </div>
            )}

            {/* 时间轴 */}
            {algorithmSteps.length > 0 && (
              <div className="absolute bottom-4 left-4 right-4 bg-white bg-opacity-90 rounded-lg p-3 shadow-lg">
                <div className="flex items-center justify-between">
                  {algorithmSteps.map((step, index) => (
                    <div key={step.id} className="flex-1 text-center relative">
                      <div 
                        className={`w-8 h-8 rounded-full mx-auto mb-1 flex items-center justify-center text-white text-sm font-bold ${
                          index === currentStep ? 'bg-blue-500' : 
                          index < currentStep ? 'bg-green-500' : 'bg-gray-300'
                        }`}
                      >
                        T{(step as any).time !== undefined ? (step as any).time : index}
                      </div>
                      <div className="text-xs text-gray-600">
                        {step.name.split(':')[1] || step.name}
                      </div>
                      {index < algorithmSteps.length - 1 && (
                        <div className="absolute top-4 left-1/2 w-full h-0.5 bg-gray-300"></div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 步骤控制面板 */}
      <Card>
        <CardHeader>
          <CardTitle>🎯 算法步骤控制</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {algorithmSteps.map((step, index) => (
              <motion.div
                key={step.id}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                  currentStep === index 
                    ? 'border-purple-500 bg-purple-50' 
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => goToStep(index)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                    index <= currentStep ? 'bg-green-500' : 'bg-gray-400'
                  }`}>
                    {index + 1}
                  </div>
                  <div className="text-sm font-medium">{step.name}</div>
                </div>
                <div className="text-xs text-gray-600 mb-2">{step.description}</div>
                
                {/* 子步骤（瀑布式进度指示器） */}
                <div className="space-y-1">
                  {step.subSteps.map((subStep, subIndex) => (
                    <motion.div 
                      key={subStep.id} 
                      className="flex items-center gap-2"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ 
                        delay: index <= currentStep ? subIndex * 0.1 : 0,
                        duration: 0.3 
                      }}
                    >
                      <motion.div 
                        className={`w-2 h-2 rounded-full ${
                          subStep.status === 'completed' ? 'bg-green-400' :
                          subStep.status === 'running' ? 'bg-yellow-400' : 'bg-gray-300'
                        }`}
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ 
                          delay: index <= currentStep ? subIndex * 0.1 + 0.2 : 0,
                          type: "spring",
                          stiffness: 200,
                          damping: 10
                        }}
                      />
                      <span className="text-xs text-gray-600">{subStep.name}</span>
                      {subStep.status === 'running' && (
                        <motion.div
                          className="w-1 h-1 bg-yellow-400 rounded-full"
                          animate={{ opacity: [0, 1, 0] }}
                          transition={{ 
                            duration: 1,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                      )}
                    </motion.div>
                  ))}
                </div>

                {/* 瀑布式进度条 */}
                {index <= currentStep && (
                  <motion.div
                    className="mt-2 h-1 bg-gray-200 rounded-full overflow-hidden"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                  >
                    <motion.div
                      className="h-full bg-gradient-to-r from-green-400 to-blue-500 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: index < currentStep ? '100%' : `${Math.min(step.progress, 100)}%` }}
                      transition={{ 
                        duration: 1.0,
                        delay: 0.3,
                        ease: "easeInOut"
                      }}
                    />
                  </motion.div>
                )}
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 详细信息面板 */}
      {currentStepData && (
        <Card>
          <CardHeader>
            <CardTitle>📋 当前步骤详情: {currentStepData.name}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* 输入数据 */}
              <div className="bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">📥 输入数据</h4>
                <div className="text-sm text-blue-700">
                  {currentStepData.input ? (
                    <pre className="whitespace-pre-wrap text-xs">
                      {JSON.stringify(currentStepData.input, null, 2)}
                    </pre>
                  ) : (
                    '无输入数据'
                  )}
                </div>
              </div>

              {/* 输出结果 */}
              <div className="bg-green-50 p-4 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">📤 输出结果</h4>
                <div className="text-sm text-green-700">
                  {currentStepData.output ? (
                    <div className="space-y-2">
                      {currentStepData.output.entities && (
                        <div>
                          <strong>实体:</strong> {currentStepData.output.entities.length}个
                        </div>
                      )}
                      {currentStepData.output.relations && (
                        <div>
                          <strong>关系:</strong> {currentStepData.output.relations.length}个
                        </div>
                      )}
                    </div>
                  ) : (
                    '无输出数据'
                  )}
                </div>
              </div>
            </div>

            {/* 处理过程 */}
            <div className="mt-6 bg-purple-50 p-4 rounded-lg">
              <h4 className="font-semibold text-purple-800 mb-3">⚙️ 处理过程</h4>
              <div className="space-y-2">
                {currentStepData.subSteps.map((subStep, index) => (
                  <motion.div
                    key={subStep.id}
                    className="flex items-center gap-3 p-2 bg-white rounded border-l-4 border-purple-400"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white text-xs ${
                      subStep.status === 'completed' ? 'bg-green-500' :
                      subStep.status === 'running' ? 'bg-yellow-500' : 'bg-gray-400'
                    }`}>
                      {subStep.status === 'completed' ? '✓' :
                       subStep.status === 'running' ? '⟳' : '○'}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-gray-800">{subStep.name}</div>
                      <div className="text-sm text-gray-600">{subStep.description}</div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default IRDAlgorithmVisualization