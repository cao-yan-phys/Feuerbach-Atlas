import { describe, expect, it } from 'vitest'
import {
  adsProjectiveLift,
  adsProjectiveProject,
  antiDeSitterFromConformal,
  hyperbolicLift,
  hyperbolicProject,
  isUnitHyperboloid,
  isUnitLorentzQuadric,
  isUnitSphere,
  lorentzProjectiveLift,
  lorentzProjectiveProject,
  sphereLift,
  sphereProject
} from '../src/geometry/charts'
import { boundsForMode } from '../src/geometry/modes'

describe('geometry charts', () => {
  it('round-trips spherical stereographic coordinates', () => {
    const point = [0.45, -0.62] as [number, number]
    const lifted = sphereLift(point)
    expect(sphereProject(lifted)![0]).toBeCloseTo(point[0], 12)
    expect(sphereProject(lifted)![1]).toBeCloseTo(point[1], 12)
    expect(isUnitSphere(lifted)).toBeLessThan(1e-12)
  })

  it('round-trips Poincare disk coordinates', () => {
    const point = [0.31, -0.44] as [number, number]
    const lifted = hyperbolicLift(point)
    expect(lifted).not.toBeNull()
    expect(hyperbolicProject(lifted!)![0]).toBeCloseTo(point[0], 12)
    expect(hyperbolicProject(lifted!)![1]).toBeCloseTo(point[1], 12)
    expect(isUnitHyperboloid(lifted!)).toBeLessThan(1e-12)
  })

  it('round-trips the Lorentzian projective chart', () => {
    const point = [0.65, -0.9] as [number, number]
    const lifted = lorentzProjectiveLift(point)
    expect(lifted).not.toBeNull()
    expect(lorentzProjectiveProject(lifted!)![0]).toBeCloseTo(point[0], 12)
    expect(lorentzProjectiveProject(lifted!)![1]).toBeCloseTo(point[1], 12)
    expect(isUnitLorentzQuadric(lifted!)).toBeLessThan(1e-12)
  })

  it('round-trips the AdS chart with vertical time', () => {
    const point = [0.65, -0.9] as [number, number]
    const lifted = adsProjectiveLift(point)
    expect(lifted).not.toBeNull()
    expect(adsProjectiveProject(lifted!)![0]).toBeCloseTo(point[0], 12)
    expect(adsProjectiveProject(lifted!)![1]).toBeCloseTo(point[1], 12)
    expect(isUnitLorentzQuadric(lifted!)).toBeLessThan(1e-12)
  })

  it('places AdS conformal time on the vertical display axis', () => {
    const chi = 0.42
    const tau = -0.31
    const point = adsProjectiveProject(antiDeSitterFromConformal(chi, tau))
    expect(point).not.toBeNull()
    expect(point![0]).toBeCloseTo(Math.sin(chi) / Math.cos(tau), 12)
    expect(point![1]).toBeCloseTo(Math.tan(tau), 12)
  })

  it('keeps every chart on its configured centered coordinate range', () => {
    const first = boundsForMode('euclidean', [[-1.2, -0.6], [1.2, -0.4], [-0.2, 1.1]])
    const second = boundsForMode('minkowski', [[-12, 8], [15, -9], [4, 11]])
    const third = boundsForMode('hyperbolic', [[0, 0], [0.1, 0.2], [-0.2, 0.1]])
    const fourth = boundsForMode('desitter', [[0, 0], [0.1, 0.2], [-0.2, 0.1]])
    const fifth = boundsForMode('ads', [[0, 0], [0.1, 0.2], [-0.2, 0.1]])
    const sixth = boundsForMode('galilei', [[0, 0], [0.1, 0.2], [-0.2, 0.1]])
    expect(first).toEqual({ minX: -2.2, maxX: 2.2, minY: -2.2, maxY: 2.2 })
    expect(second).toEqual(first)
    expect(third).toEqual({ minX: -1.1, maxX: 1.1, minY: -1.1, maxY: 1.1 })
    expect(fourth).toEqual(first)
    expect(fifth).toEqual(first)
    expect(sixth).toEqual(first)
  })
})
