import { describe, expect, it } from 'vitest'
import { buildCurvedState, sampleGeodesic } from '../src/math/curvedKernel'
import { bilinear3 } from '../src/math/linalg'
import type { Vec3 } from '../src/math/types'

const spherePoint = (latitude: number, longitude: number): Vec3 => [
  Math.cos(latitude) * Math.cos(longitude),
  Math.cos(latitude) * Math.sin(longitude),
  Math.sin(latitude)
]

const hyperbolicPoint = (rho: number, theta: number): Vec3 => [
  Math.cosh(rho),
  Math.sinh(rho) * Math.cos(theta),
  Math.sinh(rho) * Math.sin(theta)
]

const maxResidual = (values: Record<string, number>) => Math.max(...Object.values(values))

describe('curved kernel', () => {
  it('constructs the spherical configuration', () => {
    const metric: Vec3 = [1, 1, 1]
    const vertices: [Vec3, Vec3, Vec3] = [
      spherePoint(0.2, -1),
      spherePoint(-0.15, 0.65),
      spherePoint(0.65, 1.75)
    ]
    const state = buildCurvedState(metric, vertices)
    expect(state.pseudomedianFeet).not.toBeNull()
    expect(state.pseudomedianDirection).not.toBeNull()
    expect(bilinear3(metric, state.pseudomedianDirection!, state.pseudomedianDirection!)).toBeGreaterThan(1e-10)
    expect(state.pseudoaltitudeDirection).not.toBeNull()
    expect(bilinear3(metric, state.pseudoaltitudeDirection!, state.pseudoaltitudeDirection!)).toBeGreaterThan(1e-10)
    expect(state.eulerNormal).not.toBeNull()
    expect(maxResidual(state.residuals)).toBeLessThan(1e-10)
    expect(state.branches).toHaveLength(4)
    expect(state.branches.every((branch) => branch.isReal)).toBe(true)
    expect(state.branches.every((branch) => branch.residuals.cycle! < 1e-10)).toBe(true)
    expect(state.branches.every((branch) => branch.residuals.euler! < 1e-10)).toBe(true)
    expect(state.branches.every((branch) => branch.contact?.kind === 'finite')).toBe(true)
    expect(state.branches.every((branch) => Math.max(...Object.values(branch.contact!.residuals)) < 1e-9)).toBe(true)
  })

  it('constructs the hyperbolic configuration', () => {
    const metric: Vec3 = [1, -1, -1]
    const vertices: [Vec3, Vec3, Vec3] = [
      hyperbolicPoint(0.8, -1.2),
      hyperbolicPoint(0.65, 0.4),
      hyperbolicPoint(0.9, 1.7)
    ]
    const state = buildCurvedState(metric, vertices)
    expect(state.pseudomedianFeet).not.toBeNull()
    expect(state.pseudomedianDirection).not.toBeNull()
    expect(bilinear3(metric, state.pseudomedianDirection!, state.pseudomedianDirection!)).toBeGreaterThan(1e-10)
    expect(state.pseudoaltitudeDirection).not.toBeNull()
    expect(bilinear3(metric, state.pseudoaltitudeDirection!, state.pseudoaltitudeDirection!)).toBeGreaterThan(1e-10)
    expect(state.eulerNormal).not.toBeNull()
    expect(maxResidual(state.residuals)).toBeLessThan(1e-10)
    expect(state.branches.every((branch) => branch.isReal)).toBe(true)
    expect(state.branches.every((branch) => branch.residuals.cycle! < 1e-10)).toBe(true)
    expect(state.branches.every((branch) => branch.residuals.euler! < 1e-10)).toBe(true)
    expect(state.branches.every((branch) => branch.contact?.kind === 'finite')).toBe(true)
    expect(state.branches.every((branch) => Math.max(...Object.values(branch.contact!.residuals)) < 1e-9)).toBe(true)
  })

  it('samples geodesics on the unit quadric', () => {
    const metric: Vec3 = [1, -1, -1]
    const start = hyperbolicPoint(0.4, -0.5)
    const end = hyperbolicPoint(0.8, 0.7)
    const points = sampleGeodesic(metric, start, end)
    expect(points).toHaveLength(48)
    expect(points.every((point) => Math.abs(bilinear3(metric, point, point) - 1) < 1e-10)).toBe(true)
  })
})
