import { describe, expect, it } from 'vitest'
import { boostTriangle, rotateTriangle, transformTriangle } from '../src/math/affine'

describe('Euclidean triangle transforms', () => {
  it('rotates and scales about the centroid before translating', () => {
    const transformed = transformTriangle([[0, 0], [3, 0], [0, 3]], {
      translate: [5, -2],
      rotation: Math.PI / 2,
      scale: 2
    })
    const centroid = transformed.reduce(([x, y], [u, v]) => [x + u / 3, y + v / 3], [0, 0])
    expect(centroid[0]).toBeCloseTo(6, 12)
    expect(centroid[1]).toBeCloseTo(-1, 12)
    expect(Math.hypot(transformed[1][0] - transformed[0][0], transformed[1][1] - transformed[0][1])).toBeCloseTo(6, 12)
  })

  it('keeps a zero rotation and unit scale as a translation', () => {
    const transformed = transformTriangle([[0, 0], [1, 0], [0, 1]], {
      translate: [2, -3],
      rotation: 0,
      scale: 1
    })
    expect(transformed[0]![0]).toBeCloseTo(2, 12)
    expect(transformed[0]![1]).toBeCloseTo(-3, 12)
    expect(transformed[1]![0]).toBeCloseTo(3, 12)
    expect(transformed[1]![1]).toBeCloseTo(-3, 12)
    expect(transformed[2]![0]).toBeCloseTo(2, 12)
    expect(transformed[2]![1]).toBeCloseTo(-2, 12)
  })

  it('applies a Minkowski boost about its specified origin', () => {
    const origin: [number, number] = [2, -1]
    const transformed = boostTriangle([[3, -1], [2, 0], [4, 1]], origin, Math.log(2))
    expect(transformed[0]![0]).toBeCloseTo(3.25, 12)
    expect(transformed[0]![1]).toBeCloseTo(-0.25, 12)
    expect(transformed[1]![0]).toBeCloseTo(2.75, 12)
    expect(transformed[1]![1]).toBeCloseTo(0.25, 12)
    const interval = (first: [number, number], second: [number, number]) => {
      const dx = first[0] - second[0]
      const dt = first[1] - second[1]
      return dx * dx - dt * dt
    }
    expect(interval(transformed[1]!, transformed[2]!)).toBeCloseTo(interval([2, 0], [4, 1]), 12)
  })

  it('rotates about its specified origin', () => {
    const transformed = rotateTriangle([[3, -1], [2, 0], [4, 1]], [2, -1], Math.PI / 2)
    expect(transformed[0]![0]).toBeCloseTo(2, 12)
    expect(transformed[0]![1]).toBeCloseTo(0, 12)
    expect(transformed[1]![0]).toBeCloseTo(1, 12)
    expect(transformed[1]![1]).toBeCloseTo(-1, 12)
    expect(transformed[2]![0]).toBeCloseTo(0, 12)
    expect(transformed[2]![1]).toBeCloseTo(1, 12)
  })
})
