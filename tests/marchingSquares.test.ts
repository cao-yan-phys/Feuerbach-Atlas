import { describe, expect, it } from 'vitest'
import { marchingSquares } from '../src/render/marchingSquares'
import { diskSegmentPaths, projectiveBoundaryIntersection, projectiveCycleBoundaryPoints, projectiveSegmentPaths } from '../src/render/svgViewport'

describe('marching squares', () => {
  it('traces an implicit circle without a second branch', () => {
    const lines = marchingSquares(([x, y]) => x * x + y * y - 0.36, {
      minX: -1,
      maxX: 1,
      minY: -1,
      maxY: 1
    }, 100, 100)
    const points = lines.flat()
    expect(points.length).toBeGreaterThan(100)
    expect(Math.max(...points.map(([x, y]) => Math.abs(x * x + y * y - 0.36)))).toBeLessThan(1e-3)
  })

  it('uses the asymptotic decider for an ambiguous cell', () => {
    const lines = marchingSquares(([x, y]) => x * y + 0.2, {
      minX: -1,
      maxX: 1,
      minY: -1,
      maxY: 1
    }, 2, 2)
    expect(lines).toHaveLength(2)
    expect(lines.every((line) => line.length >= 2)).toBe(true)
  })

  it('snaps projective contours to the exact null boundary', () => {
    const point = projectiveBoundaryIntersection([0, 0.98], [0, 0.8], 1)
    expect(point).not.toBeNull()
    expect(point![0]).toBeCloseTo(0, 12)
    expect(point![1]).toBeCloseTo(1, 12)
  })

  it('clips a projective geodesic at each null-boundary crossing', () => {
    const paths = projectiveSegmentPaths([-2, 2.2], [2, 2.2])
    expect(paths).toHaveLength(2)
    expect(paths.flat().every(([x, y]) => 1 + x * x - y * y >= -1e-10)).toBe(true)
    expect(paths.flat().some(([x, y]) => Math.abs(1 + x * x - y * y) < 1e-10)).toBe(true)
  })

  it('solves each projective cycle endpoint on the null boundary', () => {
    const points = projectiveCycleBoundaryPoints([0, 1, 0])
    expect(points).toHaveLength(2)
    expect(points[0]![0]).toBeCloseTo(0, 12)
    expect(points[0]![1]).toBeCloseTo(1, 12)
    expect(points[1]![0]).toBeCloseTo(0, 12)
    expect(points[1]![1]).toBeCloseTo(-1, 12)
    expect(points.every(([x, y]) => Math.abs(1 + x * x - y * y) < 1e-12)).toBe(true)
  })

  it('clips hyperbolic chart paths to the exact unit-circle boundary', () => {
    const paths = diskSegmentPaths([-2, 0], [2, 0])
    expect(paths).toEqual([[[-1, 0], [1, 0]]])
  })
})
