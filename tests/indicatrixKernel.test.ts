import { describe, expect, it } from 'vitest'
import { buildIndicatrixConstruction, lorentzFinslerIndicatrixPoint, lorentzFinslerProfile, normedIndicatrixPoint, normedSupport, normalizeLorentzFinslerShape, normalizeNormedShape } from '../src/math/indicatrixKernel'
import type { Vec2 } from '../src/math/types'

const subtract = (first: Vec2, second: Vec2): Vec2 => [first[0] - second[0], first[1] - second[1]]
const add = (first: Vec2, second: Vec2): Vec2 => [first[0] + second[0], first[1] + second[1]]
const scale = (factor: number, point: Vec2): Vec2 => [factor * point[0], factor * point[1]]
const mean = (first: Vec2, second: Vec2): Vec2 => scale(0.5, add(first, second))

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
      const vertices = [0.2 + 0.08 * trial, 2.3 + 0.05 * trial, 4.45 - 0.06 * trial].map((parameter) => normedIndicatrixPoint(shape, parameter)) as [Vec2, Vec2, Vec2]
      const state = buildIndicatrixConstruction('normed', vertices, shape, normalizeLorentzFinslerShape([0, 0, 0]))
      const [a, b, c] = state.vertices
      expectPoint(state.orthocenter, subtract(add(add(a, b), c), scale(2, state.origin)))
      expectPoint(scale(3, state.centroid), add(add(a, b), c))
      expectPoint(state.feuerbachCenter, mean(state.origin, state.orthocenter))
      state.vertices.forEach((vertex, index) => {
        const translatedCenter = [add(state.origin, subtract(state.orthocenter, a)), add(state.origin, subtract(state.orthocenter, b)), add(state.origin, subtract(state.orthocenter, c))][index]!
        expectPoint(subtract(state.orthocenter, translatedCenter), subtract(vertex, state.origin))
        expectPoint(state.vertexOrthocenterMidpoints[index]!, mean(vertex, state.orthocenter))
        expectPoint(state.sideMidpoints[index]!, mean([a, b, c][(index + 1) % 3]!, [a, b, c][(index + 2) % 3]!))
      })
      expect(state.valid).toBe(true)
    })
  })

  it('preserves the Lorentz–Finsler forward and reflected three-point split', () => {
    Array.from({ length: 6 }, (_, trial) => {
      const shape = normalizeLorentzFinslerShape([0.22 * Math.sin(trial + 1), -0.18 * Math.cos(trial + 1), 0.14 * Math.sin(2 * trial + 1)])
      const vertices = [-0.8 + 0.04 * trial, 0.1 - 0.03 * trial, 0.9 - 0.02 * trial].map((parameter) => lorentzFinslerIndicatrixPoint(shape, parameter)) as [Vec2, Vec2, Vec2]
      const state = buildIndicatrixConstruction('lorentz-finsler', vertices, normalizeNormedShape([0, 0, 0]), shape)
      const [a, b, c] = state.vertices
      state.vertices.forEach((_, index) => {
        const translatedCenter = [add(state.origin, subtract(state.orthocenter, a)), add(state.origin, subtract(state.orthocenter, b)), add(state.origin, subtract(state.orthocenter, c))][index]!
        expectPoint(subtract(state.orthocenter, translatedCenter), subtract(vertices[index]!, state.origin))
        expectPoint(state.vertexOrthocenterMidpoints[index]!, mean(vertices[index]!, state.orthocenter))
        expectPoint(state.sideMidpoints[index]!, mean([a, b, c][(index + 1) % 3]!, [a, b, c][(index + 2) % 3]!))
      })
      expect(state.valid).toBe(true)
    })
  })

  it('fits a translated scaled circum-indicatrix through free normed-plane vertices', () => {
    const shape = normalizeNormedShape([0.14, -0.09, 0.04])
    const vertices: [Vec2, Vec2, Vec2] = [[-0.88, -0.46], [0.72, -0.15], [0.18, 0.94]]
    const state = buildIndicatrixConstruction('normed', vertices, shape, normalizeLorentzFinslerShape([0, 0, 0]))
    expect(state.valid).toBe(true)
    vertices.forEach((vertex) => {
      const distance = Math.min(...state.circumIndicatrix.map((point) => Math.hypot(point[0] - vertex[0], point[1] - vertex[1])))
      expect(distance).toBeLessThan(0.02)
    })
    state.translatedCenters.forEach((center, index) => {
      expectPoint(subtract(state.orthocenter, center), subtract(vertices[index]!, state.origin), 7)
    })
  })

  it('samples unbounded Lorentz branches beyond the finite construction', () => {
    const shape = normalizeLorentzFinslerShape([0.22, -0.16, 0.12])
    const vertices = [-0.8, 0.1, 0.9].map((parameter) => lorentzFinslerIndicatrixPoint(shape, parameter)) as [Vec2, Vec2, Vec2]
    const state = buildIndicatrixConstruction('lorentz-finsler', vertices, normalizeNormedShape([0, 0, 0]), shape)
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

  it('reports a degenerate triangle without constructing a circumcircle', () => {
    const vertex = normedIndicatrixPoint(normalizeNormedShape([0.1, 0, 0]), 0.2)
    const state = buildIndicatrixConstruction('normed', [vertex, vertex, vertex], normalizeNormedShape([0.1, 0, 0]), normalizeLorentzFinslerShape([0, 0, 0]))
    expect(state.circumIndicatrix).toHaveLength(0)
    expect(state.circumcircleStatus).toBe('degenerate-triangle')
    expect(state.valid).toBe(false)
  })

  it('continues a Lorentz–Minkowski circumcircle while the shape changes', () => {
    const firstShape = normalizeLorentzFinslerShape([0.22, -0.16, 0.12])
    const vertices = [-0.8, 0.1, 0.9].map((parameter) => lorentzFinslerIndicatrixPoint(firstShape, parameter)) as [Vec2, Vec2, Vec2]
    const first = buildIndicatrixConstruction('lorentz-finsler', vertices, normalizeNormedShape([0, 0, 0]), firstShape)
    const second = buildIndicatrixConstruction('lorentz-finsler', vertices, normalizeNormedShape([0, 0, 0]), normalizeLorentzFinslerShape([1.2, -0.36, 0.27]), undefined, first.circumcircleSeed)
    expect(first.circumcircleStatus).toBe('ok')
    expect(second.circumcircleStatus).toBe('ok')
    expect(second.circumcircleSeed).not.toBeNull()
  })

  it('reports the absence of a Lorentz–Minkowski circumcircle separately', () => {
    const state = buildIndicatrixConstruction('lorentz-finsler', [[0, 0], [1, 0], [0, 1]], normalizeNormedShape([0, 0, 0]), normalizeLorentzFinslerShape([0, 0, 0]))
    expect(state.circumcircleStatus).toBe('no-circumcircle')
    expect(state.valid).toBe(false)
  })
})
