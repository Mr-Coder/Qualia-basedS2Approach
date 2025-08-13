export type DrawingTool = 'pen' | 'pencil' | 'highlighter' | 'eraser';

export interface Point {
  x: number;
  y: number;
  timestamp: number;
  pressure?: number;
}

export interface DrawingPath {
  id: string;
  points: Point[];
  tool: DrawingTool;
  color: string;
  strokeWidth: number;
  pathData: string;
}

export interface WhiteboardState {
  paths: DrawingPath[];
  currentTool: DrawingTool;
  currentColor: string;
  currentStrokeWidth: number;
  isDrawingEnabled: boolean;
}

export interface WhiteboardSync {
  type: 'add_path' | 'remove_path' | 'clear' | 'undo' | 'redo';
  path?: DrawingPath;
  pathId?: string;
  timestamp: number;
  userId: string;
}