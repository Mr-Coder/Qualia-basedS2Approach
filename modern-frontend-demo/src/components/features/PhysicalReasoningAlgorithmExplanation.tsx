import React, { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { ArrowRight, Lightbulb, Network, Target, AlertCircle, BookOpen, ChevronRight } from 'lucide-react'

const PhysicalReasoningAlgorithmExplanation: React.FC = () => {
  const [expandedSection, setExpandedSection] = useState<string | null>('concept')

  const sections = [
    {
      id: 'concept',
      title: '核心概念',
      icon: <Lightbulb className="h-5 w-5" />,
      content: (
        <div className="space-y-4">
          <p className="text-gray-700">
            <strong>问题：</strong>数学题目中的已知量和未知量常常是"断开"的，没有直接的连接。
          </p>
          <p className="text-gray-700">
            <strong>解决方案：</strong>引入物性关系（物理世界的基本规律）来建立桥梁。
          </p>
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-blue-900">
              物性关系包括：守恒定律、因果关系、时空连续性等自然界的基本规律
            </p>
          </div>
        </div>
      )
    },
    {
      id: 'algorithm',
      title: '算法步骤',
      icon: <Network className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div className="space-y-4">
            {[
              {
                step: 1,
                title: '题目分解',
                desc: '提取phrase和entity，获得直陈关系',
                detail: '识别题目中的人物、物体、数量等实体，以及它们之间的直接关系'
              },
              {
                step: 2,
                title: '构建物性Graph',
                desc: '直陈关系 + 物性entity → 物性Graph',
                detail: '在原有关系基础上，加入物理概念节点（如"守恒"、"转换"等）'
              },
              {
                step: 3,
                title: '物性关系推理',
                desc: '通过物性规律扩展Graph',
                detail: '应用守恒定律、因果关系等规则，在Graph中添加新的连接'
              },
              {
                step: 4,
                title: '寻找通路',
                desc: '在扩展后的Graph中找到已知→未知的路径',
                detail: '使用图搜索算法，找到从已知量到未知量的连接路径'
              },
              {
                step: 5,
                title: '精简推理',
                desc: '只保留找到的路径，忽略其他无关推理',
                detail: '专注于解题路径，去除冗余信息，得出最终答案'
              }
            ].map(({ step, title, desc, detail }) => (
              <div key={step} className="flex items-start space-x-4">
                <div className="flex-shrink-0 w-10 h-10 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                  {step}
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-gray-900">{title}</h4>
                  <p className="text-gray-700">{desc}</p>
                  <p className="text-sm text-gray-600 mt-1">{detail}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'rules',
      title: '推理规则',
      icon: <BookOpen className="h-5 w-5" />,
      content: (
        <div className="space-y-4">
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
            <h4 className="font-semibold text-yellow-900 mb-2">三条核心推理规则</h4>
            <div className="space-y-3">
              <div>
                <strong className="text-yellow-900">规则1：守恒扩展</strong>
                <p className="text-sm text-yellow-800 mt-1">
                  当识别到"总量"概念时，自动建立部分与整体的守恒关系
                </p>
                <p className="text-xs text-yellow-700 mt-1">
                  应用时机：看到"一共"、"总共"、"全部"等关键词
                </p>
              </div>
              <div>
                <strong className="text-yellow-900">规则2：因果链接</strong>
                <p className="text-sm text-yellow-800 mt-1">
                  当存在动作或变化时，建立原因→结果的因果关系
                </p>
                <p className="text-xs text-yellow-700 mt-1">
                  应用时机：看到"导致"、"所以"、"因为"、动词等
                </p>
              </div>
              <div>
                <strong className="text-yellow-900">规则3：时空连续</strong>
                <p className="text-sm text-yellow-800 mt-1">
                  当涉及时间或空间变化时，建立连续性关系
                </p>
                <p className="text-xs text-yellow-700 mt-1">
                  应用时机：看到"每小时"、"移动"、"变化"等
                </p>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'example',
      title: '算法示例',
      icon: <Target className="h-5 w-5" />,
      content: (
        <div className="space-y-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="font-semibold mb-3">题目：小明有5个苹果，妈妈又给了他3个，一共有几个？</h4>
            
            <div className="space-y-3 text-sm">
              <div className="flex items-start">
                <span className="font-semibold text-purple-600 mr-2">Step 1:</span>
                <div>
                  <p>提取entities: 小明、苹果、5个、3个、妈妈</p>
                  <p className="text-gray-600">直陈关系: 小明→拥有→5个苹果</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <span className="font-semibold text-purple-600 mr-2">Step 2:</span>
                <div>
                  <p>加入物性entity: "总量"概念节点</p>
                  <p className="text-gray-600">形成初始Graph</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <span className="font-semibold text-purple-600 mr-2">Step 3:</span>
                <div>
                  <p>应用规则1（守恒扩展）</p>
                  <p className="text-gray-600">建立: 5个 + 3个 → 守恒 → 总量</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <span className="font-semibold text-purple-600 mr-2">Step 4:</span>
                <div>
                  <p>找到路径: 已知(5,3) → 守恒关系 → 未知(总量)</p>
                  <p className="text-gray-600">路径明确，可以求解</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <span className="font-semibold text-purple-600 mr-2">Step 5:</span>
                <div>
                  <p>沿路径推理: 5 + 3 = 8</p>
                  <p className="text-gray-600">忽略其他无关信息，得出答案</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'key-insight',
      title: '关键洞察',
      icon: <AlertCircle className="h-5 w-5" />,
      content: (
        <div className="space-y-4">
          <div className="bg-purple-50 p-4 rounded-lg">
            <h4 className="font-semibold text-purple-900 mb-2">算法的核心价值</h4>
            <ul className="space-y-2 text-purple-800">
              <li className="flex items-start">
                <ChevronRight className="h-4 w-4 mt-0.5 mr-2 flex-shrink-0" />
                <span>不需要复杂的数学推导，只需要识别物性关系模式</span>
              </li>
              <li className="flex items-start">
                <ChevronRight className="h-4 w-4 mt-0.5 mr-2 flex-shrink-0" />
                <span>通过图的连通性保证一定能找到解题路径</span>
              </li>
              <li className="flex items-start">
                <ChevronRight className="h-4 w-4 mt-0.5 mr-2 flex-shrink-0" />
                <span>推理过程可解释、可追踪</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg">
            <p className="text-green-900 font-medium">
              记住：我们要的不是所有可能的推理，而是找到一条从已知到未知的通路，
              然后沿着这条路径推理即可。其余的推理都可以"忘掉"。
            </p>
          </div>
        </div>
      )
    }
  ]

  return (
    <Card className="max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookOpen className="h-6 w-6 text-purple-600" />
          基于物性关系的推理算法讲解
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="bg-blue-50 border-l-4 border-blue-400 p-4 mb-6">
            <p className="text-blue-900">
              <strong>算法精髓：</strong>通过引入物性关系，在断开的已知和未知之间建立连接通路，
              然后沿着找到的路径进行推理，忽略无关的推理分支。
            </p>
          </div>

          {sections.map((section) => (
            <div key={section.id} className="border rounded-lg overflow-hidden">
              <button
                className="w-full px-6 py-4 bg-gray-50 hover:bg-gray-100 transition-colors flex items-center justify-between"
                onClick={() => setExpandedSection(expandedSection === section.id ? null : section.id)}
              >
                <div className="flex items-center gap-3">
                  <div className="text-purple-600">{section.icon}</div>
                  <h3 className="font-semibold text-gray-900">{section.title}</h3>
                </div>
                <ChevronRight
                  className={`h-5 w-5 text-gray-500 transition-transform ${
                    expandedSection === section.id ? 'rotate-90' : ''
                  }`}
                />
              </button>
              
              {expandedSection === section.id && (
                <div className="px-6 py-4 border-t bg-white">
                  {section.content}
                </div>
              )}
            </div>
          ))}

          <div className="mt-8 p-6 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg">
            <h3 className="font-semibold text-gray-900 mb-3">算法总结公式</h3>
            <div className="bg-white p-4 rounded-lg font-mono text-sm">
              <p>题目 → Phrase + Entity → 直陈关系 → +物性Entity → 物性Graph</p>
              <p className="mt-2">→ 物性推理扩展 → 找通路(已知→未知) → 沿路径推理 → 答案</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export default PhysicalReasoningAlgorithmExplanation