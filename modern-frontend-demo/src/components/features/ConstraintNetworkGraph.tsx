import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Network, ZoomIn, ZoomOut, RotateCcw, Eye } from 'lucide-react';

interface ConstraintNode {
  id: string;
  label: string;
  type: 'entity' | 'constraint' | 'law';
  x: number;
  y: number;
  color: string;
  size: number;
  metadata?: any;
}

interface ConstraintEdge {
  source: string;
  target: string;
  type: 'involves' | 'applies' | 'violates' | 'satisfies';
  strength: number;
  color: string;
}

interface ConstraintNetworkData {
  entities: Array<{
    entity_id: string;
    name: string;
    entity_type: string;
  }>;
  constraints: Array<{
    constraint_id: string;
    description: string;
    type: string;
    strength: number;
    entities: string[];
  }>;
  laws: Array<{
    law_type: string;
    name: string;
    priority: number;
  }>;
  violations: Array<{
    constraint_id: string;
    severity: number;
  }>;
  satisfied_constraints: string[];
}

interface ConstraintNetworkGraphProps {
  networkData: ConstraintNetworkData | null;
  width?: number;
  height?: number;
}

const ConstraintNetworkGraph: React.FC<ConstraintNetworkGraphProps> = ({
  networkData,
  width = 800,
  height = 600
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [nodes, setNodes] = useState<ConstraintNode[]>([]);
  const [edges, setEdges] = useState<ConstraintEdge[]>([]);
  const [selectedNode, setSelectedNode] = useState<ConstraintNode | null>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });

  useEffect(() => {
    if (!networkData) return;

    const newNodes: ConstraintNode[] = [];
    const newEdges: ConstraintEdge[] = [];

    // 创建实体节点
    networkData.entities.forEach((entity, index) => {
      const angle = (index / networkData.entities.length) * 2 * Math.PI;
      const radius = 150;
      newNodes.push({
        id: entity.entity_id,
        label: entity.name,
        type: 'entity',
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        color: '#3B82F6', // blue
        size: 20,
        metadata: entity
      });
    });

    // 创建约束节点
    networkData.constraints.forEach((constraint, index) => {
      const angle = (index / networkData.constraints.length) * 2 * Math.PI;
      const radius = 250;
      const isViolated = networkData.violations.some(v => v.constraint_id === constraint.constraint_id);
      const isSatisfied = networkData.satisfied_constraints.includes(constraint.constraint_id);
      
      newNodes.push({
        id: constraint.constraint_id,
        label: constraint.description.substring(0, 20) + '...',
        type: 'constraint',
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        color: isViolated ? '#EF4444' : isSatisfied ? '#10B981' : '#F59E0B', // red/green/yellow
        size: 15 + constraint.strength * 10,
        metadata: constraint
      });

      // 创建约束到实体的边
      constraint.entities.forEach(entityId => {
        newEdges.push({
          source: constraint.constraint_id,
          target: entityId,
          type: 'involves',
          strength: constraint.strength,
          color: isViolated ? '#EF4444' : '#6B7280'
        });
      });
    });

    // 创建定律节点
    networkData.laws.forEach((law, index) => {
      const angle = (index / networkData.laws.length) * 2 * Math.PI;
      const radius = 100;
      newNodes.push({
        id: law.law_type,
        label: law.name,
        type: 'law',
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        color: '#8B5CF6', // purple
        size: 25 + law.priority * 15,
        metadata: law
      });

      // 连接定律到相关约束
      networkData.constraints.forEach(constraint => {
        if (constraint.type.includes(law.law_type) || 
            law.law_type.includes(constraint.type.toLowerCase())) {
          newEdges.push({
            source: law.law_type,
            target: constraint.constraint_id,
            type: 'applies',
            strength: law.priority,
            color: '#8B5CF6'
          });
        }
      });
    });

    setNodes(newNodes);
    setEdges(newEdges);
  }, [networkData, width, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 清空画布
    ctx.clearRect(0, 0, width, height);
    
    // 应用缩放和偏移
    ctx.save();
    ctx.scale(zoom, zoom);
    ctx.translate(offset.x, offset.y);

    // 绘制边
    edges.forEach(edge => {
      const sourceNode = nodes.find(n => n.id === edge.source);
      const targetNode = nodes.find(n => n.id === edge.target);
      
      if (sourceNode && targetNode) {
        ctx.beginPath();
        ctx.moveTo(sourceNode.x, sourceNode.y);
        ctx.lineTo(targetNode.x, targetNode.y);
        ctx.strokeStyle = edge.color;
        ctx.lineWidth = Math.max(1, edge.strength * 3);
        ctx.globalAlpha = 0.6;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });

    // 绘制节点
    nodes.forEach(node => {
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI);
      ctx.fillStyle = node.color;
      ctx.fill();
      
      // 选中节点的高亮
      if (selectedNode && selectedNode.id === node.id) {
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.stroke();
      }

      // 绘制标签
      ctx.fillStyle = '#000000';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(node.label, node.x, node.y + node.size + 15);
    });

    ctx.restore();
  }, [nodes, edges, selectedNode, zoom, offset, width, height]);

  const handleMouseDown = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / zoom - offset.x;
    const y = (event.clientY - rect.top) / zoom - offset.y;

    // 检查是否点击了节点
    const clickedNode = nodes.find(node => {
      const distance = Math.sqrt((x - node.x) ** 2 + (y - node.y) ** 2);
      return distance <= node.size;
    });

    if (clickedNode) {
      setSelectedNode(clickedNode);
    } else {
      setSelectedNode(null);
      setIsDragging(true);
      setLastMousePos({ x: event.clientX, y: event.clientY });
    }
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    const deltaX = event.clientX - lastMousePos.x;
    const deltaY = event.clientY - lastMousePos.y;

    setOffset(prev => ({
      x: prev.x + deltaX / zoom,
      y: prev.y + deltaY / zoom
    }));

    setLastMousePos({ x: event.clientX, y: event.clientY });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 3));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.3));
  };

  const handleReset = () => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setSelectedNode(null);
  };

  if (!networkData) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            约束网络图
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-gray-500">
            暂无网络数据
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            约束网络图
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={handleZoomIn}>
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleZoomOut}>
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="flex gap-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-blue-500"></div>
            <span>实体</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span>已满足约束</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span>违规约束</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded-full bg-purple-500"></div>
            <span>物理定律</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            className="border rounded-lg cursor-move"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
          
          {selectedNode && (
            <div className="absolute top-4 right-4 bg-white border rounded-lg p-3 shadow-lg max-w-xs">
              <div className="flex items-center gap-2 mb-2">
                <div 
                  className="w-4 h-4 rounded-full" 
                  style={{ backgroundColor: selectedNode.color }}
                ></div>
                <Badge variant="outline">
                  {selectedNode.type === 'entity' ? '实体' : 
                   selectedNode.type === 'constraint' ? '约束' : '定律'}
                </Badge>
              </div>
              <h4 className="font-semibold mb-1">{selectedNode.label}</h4>
              {selectedNode.metadata && (
                <div className="text-sm text-gray-600">
                  {selectedNode.type === 'constraint' && (
                    <>
                      <p>类型: {selectedNode.metadata.type}</p>
                      <p>强度: {selectedNode.metadata.strength}</p>
                      <p>涉及实体: {selectedNode.metadata.entities.length}个</p>
                    </>
                  )}
                  {selectedNode.type === 'law' && (
                    <p>优先级: {selectedNode.metadata.priority}</p>
                  )}
                  {selectedNode.type === 'entity' && (
                    <p>类型: {selectedNode.metadata.entity_type}</p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
        
        <div className="mt-4 text-sm text-gray-600">
          <p>💡 提示: 点击节点查看详情，拖拽可移动视图，使用缩放按钮调整视图大小</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default ConstraintNetworkGraph;