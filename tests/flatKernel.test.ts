import { describe, expect, it } from 'vitest'
import { buildFlatState, classifyMinkowskiTriangle, euclideanCircumcircle, flatCycleContact, minkowskiCircumcycle } from '../src/math/flatKernel'

const maxResidual = (values: Record<string, number>) => Math.max(...Object.values(values))

describe('flat kernel', () => {
  it('constructs the Euclidean nine-point configuration', () => {
    const state = buildFlatState(1, [[-1.2, -0.6], [1.2, -0.4], [-0.2, 1.1]])
    expect(state.ninePoint).not.toBeNull()
    expect(state.circumcircle).not.toBeNull()
    expect(state.centroid![0]).toBeCloseTo(-1 / 15, 12)
    expect(state.centroid![1]).toBeCloseTo(1 / 30, 12)
    expect(state.tangentCycles).toHaveLength(4)
    expect(maxResidual(state.residuals)).toBeLessThan(1e-10)
    expect(state.contacts.every((contact) => contact.point !== null)).toBe(true)
    expect(state.contacts.every((contact) => Math.abs(contact.discriminant ?? Infinity) < 1e-8)).toBe(true)
  })

  it('places the Euclidean circumcenter at equal distance from all three vertices', () => {
    const vertices: [[number, number], [number, number], [number, number]] = [[-1.2, -0.6], [1.2, -0.4], [-0.2, 1.1]]
    const center = buildFlatState(1, vertices).circumcenter
    expect(center).not.toBeNull()
    const distances = vertices.map(([x, y]) => (x - center![0]) ** 2 + (y - center![1]) ** 2)
    expect(Math.max(...distances) - Math.min(...distances)).toBeLessThan(1e-10)
  })

  it('classifies all Minkowski preset types', () => {
    expect(classifyMinkowskiTriangle([[-1.8, -0.15], [0.35, 0.25], [1.2, -0.35]]).kind).toBe('spacelike')
    expect(classifyMinkowskiTriangle([[-0.15, -1.8], [0.25, 0.35], [-0.35, 1.2]]).kind).toBe('timelike')
    expect(classifyMinkowskiTriangle([[-1, 0], [1, 0], [0, 2]]).kind).toBe('mixed')
    expect(classifyMinkowskiTriangle([[-1, 0], [1, 0], [0, 1]]).kind).toBe('null')
  })

  it('keeps non-mixed Minkowski cycles real', () => {
    const state = buildFlatState(-1, [[-1.8, -0.15], [0.35, 0.25], [1.2, -0.35]])
    expect(state.kind).toBe('spacelike')
    expect(state.sideLambdas.filter((value) => value < 0)).toHaveLength(1)
    expect(state.tangentCycles).toHaveLength(4)
    expect(maxResidual(state.residuals)).toBeLessThan(1e-9)
    expect(state.contacts.every((contact) => contact.point !== null)).toBe(true)
    expect(state.contacts.every((contact) => Math.abs(contact.discriminant ?? Infinity) < 1e-8)).toBe(true)
  })

  it('constructs a Minkowski circumcycle through all three vertices', () => {
    const vertices: [[number, number], [number, number], [number, number]] = [[-1.8, -0.15], [0.35, 0.25], [1.2, -0.35]]
    const cycle = buildFlatState(-1, vertices).circumcircle
    expect(cycle).not.toBeNull()
    const values = vertices.map(([x, y]) => (x - cycle!.center[0]) ** 2 - (y - cycle!.center[1]) ** 2)
    expect(Math.max(...values) - Math.min(...values)).toBeLessThan(1e-10)
  })

  it('relates the circumcycle and Euler cycle by the two nine-point homotheties', () => {
    const cases: Array<[1 | -1, [[number, number], [number, number], [number, number]]]> = [
      [1 as const, [[-1.2, -0.6], [1.2, -0.4], [-0.2, 1.1]] as [[number, number], [number, number], [number, number]]],
      [-1 as const, [[-1.8, -0.15], [0.35, 0.25], [1.2, -0.35]] as [[number, number], [number, number], [number, number]]]
    ]

    cases.forEach(([sigma, vertices]) => {
      const state = buildFlatState(sigma, vertices)
      const circumcircle = state.circumcircle!
      const euler = state.ninePoint!
      const cases = [
        { center: state.orthocenter!, scale: 0.5, points: vertices.map((point) => [(point[0] + state.orthocenter![0]) / 2, (point[1] + state.orthocenter![1]) / 2] as [number, number]) },
        { center: state.centroid, scale: -0.5, points: state.midpoints }
      ]
      cases.forEach(({ center, scale, points }) => {
        expect(center[0] + scale * (circumcircle.center[0] - center[0])).toBeCloseTo(euler.center[0], 9)
        expect(center[1] + scale * (circumcircle.center[1] - center[1])).toBeCloseTo(euler.center[1], 9)
        expect(scale * scale * circumcircle.radiusSquared).toBeCloseTo(euler.radiusSquared, 9)
        points.forEach((point) => {
          const value = sigma === 1
            ? (point[0] - euler.center[0]) ** 2 + (point[1] - euler.center[1]) ** 2
            : (point[0] - euler.center[0]) ** 2 - (point[1] - euler.center[1]) ** 2
          expect(value).toBeCloseTo(euler.radiusSquared, 9)
        })
      })
    })
  })

  it('constructs an independent Euclidean circumcircle for Minkowski vertices', () => {
    const vertices: [[number, number], [number, number], [number, number]] = [[-1.8, -0.15], [0.35, 0.25], [1.2, -0.35]]
    const circle = euclideanCircumcircle(vertices)
    expect(circle).not.toBeNull()
    const values = vertices.map(([x, y]) => (x - circle!.center[0]) ** 2 + (y - circle!.center[1]) ** 2)
    expect(Math.max(...values) - Math.min(...values)).toBeLessThan(1e-10)
  })

  it('constructs an independent Minkowski circumcycle for Euclidean vertices', () => {
    const vertices: [[number, number], [number, number], [number, number]] = [[-1.2, -0.6], [1.2, -0.4], [-0.2, 1.1]]
    const cycle = minkowskiCircumcycle(vertices)
    expect(cycle).not.toBeNull()
    const values = vertices.map(([x, y]) => (x - cycle!.center[0]) ** 2 - (y - cycle!.center[1]) ** 2)
    expect(Math.max(...values) - Math.min(...values)).toBeLessThan(1e-10)
  })

  it('does not mark a transversal cycle intersection as a contact point', () => {
    expect(flatCycleContact(1, { center: [0, 0], radiusSquared: 1, label: 'A' }, { center: [1, 0], radiusSquared: 1, label: 'B' }).point).toBeNull()
    expect(flatCycleContact(1, { center: [0, 0], radiusSquared: 1, label: 'A' }, { center: [2, 0], radiusSquared: 1, label: 'B' }).point).toEqual([1, 0])
  })

  it('marks null and degenerate flat triangles invalid', () => {
    expect(buildFlatState(1, [[0, 0], [1, 0], [2, 0]]).valid).toBe(false)
    expect(buildFlatState(-1, [[-1, 0], [1, 0], [0, 1]]).valid).toBe(false)
    expect(buildFlatState(-1, [[-1, 0], [1, 0], [0, 2]]).valid).toBe(true)
  })
})
