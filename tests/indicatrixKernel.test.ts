import { describe, expect, it } from 'vitest'
import { buildIndicatrixConstruction, lorentzFinslerIndicatrixPoint, lorentzFinslerProfile, normedIndicatrixPoint, normedSupport, normalizeLorentzFinslerShape, normalizeNormedShape } from '../src/math/indicatrixKernel'
import type { Vec2 } from '../src/math/types'

const subtract = (first: Vec2, second: Vec2): Vec2 => [first[0] - second[0], first[1] - second[1]]
const add = (first: Vec2, second: Vec2): Vec2 => [first[0] + second[0], first[1] + second[1]]
const scale = (factor: number, point: Vec2): Vec2 => [factor * point[0], factor * point[1]]

const expectPoint = (actual: Vec2, expected: Vec2, precision = 10) => {
  expect(actual[0]).toBeCloseTo(expected[0], precision)
  expect(actual[1]).toBeCloseTo(expected[1], precision)
}

describe('indicatrix kernel', () => {
  it('keeps support-function indicatrices smooth, strictly convex, and centrally symmetric', () => {
    Array.from({ length: 12 }, (_, shapeIndex) => normalizeNormedShape([
      0.36 * Math.sin(1.7 * shapeIndex),
      0.36 * Math.cos(2.3 * shapeIndex),
      0.36 * Math.sin(3.1 * shapeIndex)
    ])).forEach((shape) => {
      for (let index = 0; index <= 960; index += 1) {
        const phi = 2 * Math.PI * index / 960
        const support = normedSupport(shape, phi)
        expect(support.h).toBeGreaterThan(0)
        expect(support.curvature).toBeGreaterThan(0)
        expectPoint(add(normedIndicatrixPoint(shape, phi), normedIndicatrixPoint(shape, phi + Math.PI)), [0, 0], 9)
      }
    })
  })

  it('keeps Lorentz–Finsler profiles positive and Lorentz-admissible', () => {
    Array.from({ length: 12 }, (_, shapeIndex) => normalizeLorentzFinslerShape([
      0.52 * Math.sin(1.1 * shapeIndex),
      0.52 * Math.cos(2.1 * shapeIndex),
      0.52 * Math.sin(3.2 * shapeIndex)
    ])).forEach((shape) => {
      for (let index = 0; index <= 960; index += 1) {
        const theta = -4 + 8 * index / 960
        const profile = lorentzFinslerProfile(shape, theta)
        expect(profile.value).toBeGreaterThan(0)
        expect(profile.second - profile.value).toBeLessThan(0)
      }
    })
  })

  it('matches the Lorentz fundamental-tensor determinant numerically', () => {
    const shape = normalizeLorentzFinslerShape([0.25, -0.15, 0.1])
    const fSquared = (x: number, t: number) => {
      const radial = Math.sqrt(t * t - x * x)
      const theta = Math.atanh(x / t)
      const profile = lorentzFinslerProfile(shape, theta)
      return radial * radial * profile.value * profile.value
    }
    const step = 1e-4
    const samples: Array<[number, number]> = [[-0.45, 1.35], [0.2, 1.1], [0.65, 1.8]]
    samples.forEach(([x, t]) => {
      const center = fSquared(x, t)
      const hxx = (fSquared(x + step, t) - 2 * center + fSquared(x - step, t)) / (step * step)
      const htt = (fSquared(x, t + step) - 2 * center + fSquared(x, t - step)) / (step * step)
      const hxt = (fSquared(x + step, t + step) - fSquared(x + step, t - step) - fSquared(x - step, t + step) + fSquared(x - step, t - step)) / (4 * step * step)
      const determinant = (hxx * htt - hxt * hxt) / 4
      const theta = Math.atanh(x / t)
      const profile = lorentzFinslerProfile(shape, theta)
      expect(determinant).toBeCloseTo(profile.value ** 3 * (profile.second - profile.value), 4)
    })
  })

  it('preserves every normed-plane affine and six-point identity', () => {
    Array.from({ length: 6 }, (_, trial) => {
      const shape = normalizeNormedShape([0.18 * Math.sin(trial + 1), -0.14 * Math.cos(trial + 1), 0.06 * Math.sin(2 * trial + 1)])
      const positions: [number, number, number] = [0.2 + 0.08 * trial, 2.3 + 0.05 * trial, 4.45 - 0.06 * trial]
      const state = buildIndicatrixConstruction('normed', positions, shape, normalizeLorentzFinslerShape([0, 0, 0]))
      const [a, b, c] = state.vertices
      expectPoint(state.orthocenter, add(add(a, b), c))
      expectPoint(scale(3, state.centroid), state.orthocenter)
      expectPoint(scale(2, state.feuerbachCenter), state.orthocenter)
      state.vertices.forEach((vertex, index) => {
        const translatedCenter = [add(b, c), add(c, a), add(a, b)][index]!
        expectPoint(subtract(state.orthocenter, translatedCenter), vertex)
        expectPoint(state.vertexOrthocenterMidpoints[index]!, add(state.feuerbachCenter, scale(0.5, vertex)))
        expectPoint(state.sideMidpoints[index]!, add(state.feuerbachCenter, scale(-0.5, vertex)))
        expectPoint(state.sideMidpoints[index]!, add(state.feuerbachCenter, scale(0.5, normedIndicatrixPoint(shape, positions[index]! + Math.PI))))
      })
      expect(state.valid).toBe(true)
    })
  })

  it('preserves the Lorentz–Finsler forward and reflected three-point split', () => {
    Array.from({ length: 6 }, (_, trial) => {
      const shape = normalizeLorentzFinslerShape([0.22 * Math.sin(trial + 1), -0.18 * Math.cos(trial + 1), 0.14 * Math.sin(2 * trial + 1)])
      const positions: [number, number, number] = [-0.8 + 0.04 * trial, 0.1 - 0.03 * trial, 0.9 - 0.02 * trial]
      const state = buildIndicatrixConstruction('lorentz-finsler', positions, normalizeNormedShape([0, 0, 0]), shape)
      const [a, b, c] = state.vertices
      state.vertices.forEach((_, index) => {
        const translatedCenter = [add(b, c), add(c, a), add(a, b)][index]!
        const indicatrix = lorentzFinslerIndicatrixPoint(shape, positions[index]!)
        expectPoint(subtract(state.orthocenter, translatedCenter), indicatrix)
        expectPoint(state.vertexOrthocenterMidpoints[index]!, add(state.feuerbachCenter, scale(0.5, indicatrix)))
        expectPoint(state.sideMidpoints[index]!, add(state.feuerbachCenter, scale(-0.5, indicatrix)))
      })
      expect(state.valid).toBe(true)
    })
  })

  it('samples unbounded Lorentz branches beyond the finite construction', () => {
    const state = buildIndicatrixConstruction('lorentz-finsler', [-0.8, 0.1, 0.9], normalizeNormedShape([0, 0, 0]), normalizeLorentzFinslerShape([0.22, -0.16, 0.12]))
    const finiteReach = Math.max(1, ...[
      ...state.vertices,
      state.origin,
      state.centroid,
      state.orthocenter,
      state.feuerbachCenter,
      ...state.translatedCenters,
      ...state.sideMidpoints,
      ...state.vertexOrthocenterMidpoints
    ].map((point) => Math.max(Math.abs(point[0]), Math.abs(point[1]))))
    const endpointReach = Math.max(...[state.circumIndicatrix[0]!, state.circumIndicatrix.at(-1)!].map((point) => Math.max(Math.abs(point[0]), Math.abs(point[1]))))
    expect(endpointReach).toBeGreaterThan(20 * finiteReach)
  })

  it('retains the indicatrix while rejecting derived data for a degenerate triangle', () => {
    const state = buildIndicatrixConstruction('normed', [0.2, 0.2, 0.2], normalizeNormedShape([0.1, 0, 0]), normalizeLorentzFinslerShape([0, 0, 0]))
    expect(state.circumIndicatrix).toHaveLength(512)
    expect(state.valid).toBe(false)
  })
})
