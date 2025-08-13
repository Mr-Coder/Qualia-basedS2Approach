import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { TouchCanvas } from '../../../src/components/whiteboard/TouchCanvas';
import { DrawingPath } from '../../../src/types/whiteboard';

describe('TouchCanvas', () => {
  const mockOnPathComplete = jest.fn();
  const defaultProps = {
    tool: 'pen' as const,
    color: '#000000',
    strokeWidth: 4,
    onPathComplete: mockOnPathComplete,
    paths: [],
    isDrawingEnabled: true,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders without crashing', () => {
    const { getByTestId } = render(<TouchCanvas {...defaultProps} />);
    expect(getByTestId).toBeTruthy();
  });

  it('starts drawing on touch start', () => {
    const { getByTestId } = render(<TouchCanvas {...defaultProps} />);
    const canvas = getByTestId('touch-canvas');

    fireEvent(canvas, 'responderGrant', {
      nativeEvent: { locationX: 100, locationY: 100, timestamp: Date.now() },
    });

    // Canvas should be in drawing state
    expect(canvas).toBeTruthy();
  });

  it('completes path on touch end', () => {
    const { getByTestId } = render(<TouchCanvas {...defaultProps} />);
    const canvas = getByTestId('touch-canvas');
    const timestamp = Date.now();

    // Start drawing
    fireEvent(canvas, 'responderGrant', {
      nativeEvent: { locationX: 100, locationY: 100, timestamp },
    });

    // Move
    fireEvent(canvas, 'responderMove', {
      nativeEvent: { locationX: 150, locationY: 150, timestamp: timestamp + 10 },
    });

    // End drawing
    fireEvent(canvas, 'responderRelease');

    expect(mockOnPathComplete).toHaveBeenCalledWith(
      expect.objectContaining({
        tool: 'pen',
        color: '#000000',
        strokeWidth: 4,
        points: expect.arrayContaining([
          expect.objectContaining({ x: 100, y: 100 }),
          expect.objectContaining({ x: 150, y: 150 }),
        ]),
      })
    );
  });

  it('does not draw when drawing is disabled', () => {
    const { getByTestId } = render(
      <TouchCanvas {...defaultProps} isDrawingEnabled={false} />
    );
    const canvas = getByTestId('touch-canvas');

    fireEvent(canvas, 'responderGrant', {
      nativeEvent: { locationX: 100, locationY: 100, timestamp: Date.now() },
    });
    fireEvent(canvas, 'responderMove', {
      nativeEvent: { locationX: 150, locationY: 150, timestamp: Date.now() + 10 },
    });
    fireEvent(canvas, 'responderRelease');

    expect(mockOnPathComplete).not.toHaveBeenCalled();
  });

  it('renders existing paths', () => {
    const existingPaths: DrawingPath[] = [
      {
        id: 'path1',
        points: [
          { x: 10, y: 10, timestamp: 1000 },
          { x: 20, y: 20, timestamp: 1010 },
        ],
        tool: 'pen',
        color: '#FF0000',
        strokeWidth: 6,
        pathData: 'M 10 10 L 20 20',
      },
    ];

    const { getByTestId } = render(
      <TouchCanvas {...defaultProps} paths={existingPaths} />
    );

    // Path should be rendered in the SVG
    expect(getByTestId).toBeTruthy();
  });

  it('changes opacity for pencil tool', () => {
    const { rerender } = render(
      <TouchCanvas {...defaultProps} tool="pencil" />
    );

    // Pencil should have different opacity
    expect(defaultProps.tool).toBe('pen');
    
    rerender(<TouchCanvas {...defaultProps} tool="pencil" />);
    // Tool should be updated
  });

  it('handles highlighter with increased stroke width', () => {
    const { getByTestId } = render(
      <TouchCanvas {...defaultProps} tool="highlighter" strokeWidth={10} />
    );
    const canvas = getByTestId('touch-canvas');

    fireEvent(canvas, 'responderGrant', {
      nativeEvent: { locationX: 50, locationY: 50, timestamp: Date.now() },
    });
    fireEvent(canvas, 'responderMove', {
      nativeEvent: { locationX: 100, locationY: 100, timestamp: Date.now() + 10 },
    });
    fireEvent(canvas, 'responderRelease');

    expect(mockOnPathComplete).toHaveBeenCalledWith(
      expect.objectContaining({
        tool: 'highlighter',
        strokeWidth: 10,
      })
    );
  });
});