import { describe, expect, it } from 'vitest'
import { boostTriangle, rotateTriangle, transformTriangle, transvectHyperbolicTriangle, transvectLorentzianTriangle, transvectSphereTriangle } from '../src/math/affine'

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

  it('preserves both projective Lorentzian chart domains', () => {
    const deSitter: [[number, number], [number, number], [number, number]] = [[0.2, 0.6], [0.8, 0.3], [-0.5, -0.7]]
    const antiDeSitter: [[number, number], [number, number], [number, number]] = [[0.8, 0.1], [-0.4, 0.6], [0.3, -0.9]]
    const deSitterValue = ([x, y]: [number, number]) => 1 + x * x - y * y
    const antiDeSitterValue = ([x, y]: [number, number]) => 1 + y * y - x * x

    const deSitterBoosted = boostTriangle(deSitter, [0, 0], 0.1)
    const antiDeSitterBoosted = boostTriangle(antiDeSitter, [0, 0], -0.1)

    deSitter.forEach((vertex, index) => {
      expect(deSitterValue(deSitterBoosted[index]!)).toBeCloseTo(deSitterValue(vertex), 12)
      expect(deSitterValue(deSitterBoosted[index]!)).toBeGreaterThan(0)
    })
    antiDeSitter.forEach((vertex, index) => {
      expect(antiDeSitterValue(antiDeSitterBoosted[index]!)).toBeCloseTo(antiDeSitterValue(vertex), 12)
      expect(antiDeSitterValue(antiDeSitterBoosted[index]!)).toBeGreaterThan(0)
    })
  })

  it('applies the noncompact Lorentzian transvections in their principal charts', () => {
    const deSitter: [[number, number], [number, number], [number, number]] = [[0.2, 0.6], [0.8, 0.3], [-0.5, -0.7]]
    const antiDeSitter: [[number, number], [number, number], [number, number]] = [[0.8, 0.1], [-0.4, 0.6], [0.3, -0.9]]
    const deSitterValue = ([x, y]: [number, number]) => 1 + x * x - y * y
    const antiDeSitterValue = ([x, y]: [number, number]) => 1 + y * y - x * x
    const deSitterTransformed = transvectLorentzianTriangle(deSitter, 'desitter', 0.35)!
    const antiDeSitterTransformed = transvectLorentzianTriangle(antiDeSitter, 'ads', -0.45)!

    deSitterTransformed.forEach((vertex) => expect(deSitterValue(vertex)).toBeGreaterThan(0))
    antiDeSitterTransformed.forEach((vertex) => expect(antiDeSitterValue(vertex)).toBeGreaterThan(0))
    const deSitterRestored = transvectLorentzianTriangle(deSitterTransformed, 'desitter', -0.35)!
    const antiDeSitterRestored = transvectLorentzianTriangle(antiDeSitterTransformed, 'ads', 0.45)!
    deSitter.forEach((vertex, index) => {
      expect(deSitterRestored[index]![0]).toBeCloseTo(vertex[0], 12)
      expect(deSitterRestored[index]![1]).toBeCloseTo(vertex[1], 12)
    })
    antiDeSitter.forEach((vertex, index) => {
      expect(antiDeSitterRestored[index]![0]).toBeCloseTo(vertex[0], 12)
      expect(antiDeSitterRestored[index]![1]).toBeCloseTo(vertex[1], 12)
    })
    expect(transvectLorentzianTriangle(deSitter, 'desitter', 1)).toBeNull()
    expect(transvectLorentzianTriangle(antiDeSitter, 'ads', -1)).toBeNull()
  })

  it('preserves the Poincaré disk under a hyperbolic transvection', () => {
    const vertices: [[number, number], [number, number], [number, number]] = [[0.2, 0.3], [-0.4, 0.1], [0.1, -0.5]]
    const transformed = transvectHyperbolicTriangle(vertices, 0.72)!
    transformed.forEach(([x, y]) => expect(x * x + y * y).toBeLessThan(1))
    const restored = transvectHyperbolicTriangle(transformed, -0.72)!
    vertices.forEach((vertex, index) => {
      expect(restored[index]![0]).toBeCloseTo(vertex[0], 12)
      expect(restored[index]![1]).toBeCloseTo(vertex[1], 12)
    })
  })

  it('keeps spherical transvection continuous across the stereographic infinity', () => {
    const vertices: [[number, number], [number, number], [number, number]] = [[0.2, 0.3], [-0.4, 0.1], [0.1, -0.5]]
    const angle = Math.PI + 0.35
    const transformed = transvectSphereTriangle(vertices, angle)!
    const restored = transvectSphereTriangle(transformed, -angle)!
    vertices.forEach((vertex, index) => {
      expect(restored[index]![0]).toBeCloseTo(vertex[0], 11)
      expect(restored[index]![1]).toBeCloseTo(vertex[1], 11)
    })
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
