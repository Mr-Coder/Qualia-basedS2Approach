import React, { useState } from 'react'
import Layout from '@/components/layout/Layout'
import ProblemSolver from '@/components/features/ProblemSolver'
import HistoryPanel from '@/components/features/HistoryPanel'
import KnowledgeMap from '@/components/features/KnowledgeMap'
import LearningGuide from '@/components/features/LearningGuide'
import ErrorAnalysis from '@/components/features/ErrorAnalysis'
import StrategyAnalysis from '@/components/features/StrategyAnalysis'
import EntityRelationshipDiagram from '@/components/features/EntityRelationshipDiagram'
import { useProblemStore } from '@/stores/problemStore'

type TabType = 'solver' | 'knowledge' | 'learning' | 'error' | 'strategy' | 'diagram'

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('solver')
  const { solveResult } = useProblemStore()

  const renderContent = () => {
    switch (activeTab) {
      case 'solver':
        return (
          <div className="space-y-8">
            <ProblemSolver />
            <div className="max-w-md mx-auto">
              <HistoryPanel />
            </div>
          </div>
        )
      case 'knowledge':
        return <KnowledgeMap />
      case 'learning':
        return <LearningGuide />
      case 'error':
        return <ErrorAnalysis />
      case 'strategy':
        return <StrategyAnalysis />
      case 'diagram':
        return <EntityRelationshipDiagram 
          entities={solveResult?.entities || []}
          relationships={solveResult?.relationships || []}
          physicalConstraints={solveResult?.physicalConstraints || []}
          physicalProperties={solveResult?.physicalProperties}
          deepRelations={solveResult?.deepRelations || []}
          implicitConstraints={solveResult?.implicitConstraints || []}
          visualizationConfig={solveResult?.visualizationConfig}
          width={800}
          height={600}
        />
      default:
        return <ProblemSolver />
    }
  }

  return (
    <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
      {renderContent()}
    </Layout>
  )
}

export default App