import type { Vec2 } from './types'

export type IndicatrixMode = 'normed' | 'lorentz-finsler'

export interface NormedShape {
  coefficients: [number, number, number]
}

export interface LorentzFinslerShape {
  coefficients: [number, number, number]
}

export interface IndicatrixConstruction {
  mode: IndicatrixMode
  vertices: [Vec2, Vec2, Vec2]
  circumIndicatrix: Vec2[]
  translatedCenters: [Vec2, Vec2, Vec2]
  translatedIndicatrices: [Vec2[], Vec2[], Vec2[]]
  feuBranches: Vec2[][]
  origin: Vec2
  centroid: Vec2
  orthocenter: Vec2
  feuerbachCenter: Vec2
  sideMidpoints: [Vec2, Vec2, Vec2]
  vertexOrthocenterMidpoints: [Vec2, Vec2, Vec2]
  valid: boolean
}

const normedWeights: [number, number, number] = [3, 3, 15]
const lorentzCenters: [number, number, number] = [-1.1, 0.15, 1.25]
const lorentzWidth = 1.9
export const lorentzDisplayRapidity = 1.65
export const lorentzRenderRapidity = 6

const add = (first: Vec2, second: Vec2): Vec2 => [first[0] + second[0], first[1] + second[1]]
const subtract = (first: Vec2, second: Vec2): Vec2 => [first[0] - second[0], first[1] - second[1]]
const scale = (factor: number, point: Vec2): Vec2 => [factor * point[0], factor * point[1]]
const mean = (first: Vec2, second: Vec2): Vec2 => scale(0.5, add(first, second))
const sum = (points: Vec2[]): Vec2 => points.reduce(add, [0, 0])
const cross = (first: Vec2, second: Vec2) => first[0] * second[1] - first[1] * second[0]
const squaredDistance = (first: Vec2, second: Vec2) => {
  const difference = subtract(first, second)
  return difference[0] * difference[0] + difference[1] * difference[1]
}

const tuple = (values: number[]): [number, number, number] => [values[0] ?? 0, values[1] ?? 0, values[2] ?? 0]

export const normalizeNormedShape = (coefficients: readonly number[]): NormedShape => {
  const finite = coefficients.map((value) => Number.isFinite(value) ? value : 0)
  const first = finite.reduce((total, value) => total + Math.abs(value), 0)
  const second = finite.reduce((total, value, index) => total + normedWeights[index]! * Math.abs(value), 0)
  const factor = Math.min(1, first > 0 ? 0.85 / first : 1, second > 0 ? 0.85 / second : 1)
  return { coefficients: tuple(finite.map((value) => factor * value)) }
}

const gaussian = (theta: number, center: number) => Math.exp(-(((theta - center) / lorentzWidth) ** 2))

const lorentzBound = (amplitude: number) => 2 * amplitude / (lorentzWidth * lorentzWidth) + 2 * amplitude * amplitude / (Math.E * lorentzWidth * lorentzWidth)

export const normalizeLorentzFinslerShape = (coefficients: readonly number[]): LorentzFinslerShape => {
  const finite = coefficients.map((value) => Number.isFinite(value) ? value : 0)
  const amplitude = finite.reduce((total, value) => total + Math.abs(value), 0)
  if (amplitude === 0 || lorentzBound(amplitude) <= 0.85) {
    return { coefficients: tuple(finite) }
  }
  let lower = 0
  let upper = 1
  for (let index = 0; index < 64; index += 1) {
    const middle = (lower + upper) / 2
    if (lorentzBound(middle * amplitude) <= 0.85) {
      lower = middle
    } else {
      upper = middle
    }
  }
  return { coefficients: tuple(finite.map((value) => lower * value)) }
}

export const normedSupport = (shape: NormedShape, phi: number) => {
  const [a, b, c] = shape.coefficients
  const h = 1 + a * Math.cos(2 * phi) + b * Math.sin(2 * phi) + c * Math.cos(4 * phi)
  const derivative = -2 * a * Math.sin(2 * phi) + 2 * b * Math.cos(2 * phi) - 4 * c * Math.sin(4 * phi)
  const second = -4 * a * Math.cos(2 * phi) - 4 * b * Math.sin(2 * phi) - 16 * c * Math.cos(4 * phi)
  return { h, derivative, curvature: h + second }
}

export const normedIndicatrixPoint = (shape: NormedShape, phi: number): Vec2 => {
  const { h, derivative } = normedSupport(shape, phi)
  return [h * Math.cos(phi) - derivative * Math.sin(phi), h * Math.sin(phi) + derivative * Math.cos(phi)]
}

export const lorentzFinslerProfile = (shape: LorentzFinslerShape, theta: number) => {
  const q = shape.coefficients.reduce((total, coefficient, index) => total + coefficient * gaussian(theta, lorentzCenters[index]!), 0)
  const first = shape.coefficients.reduce((total, coefficient, index) => {
    const scaled = (theta - lorentzCenters[index]!) / lorentzWidth
    return total - 2 * coefficient * scaled * gaussian(theta, lorentzCenters[index]!) / lorentzWidth
  }, 0)
  const second = shape.coefficients.reduce((total, coefficient, index) => {
    const scaled = (theta - lorentzCenters[index]!) / lorentzWidth
    return total + coefficient * (4 * scaled * scaled - 2) * gaussian(theta, lorentzCenters[index]!) / (lorentzWidth * lorentzWidth)
  }, 0)
  const value = Math.exp(q)
  return { value, first, second: value * (second + first * first) }
}

export const lorentzFinslerIndicatrixPoint = (shape: LorentzFinslerShape, theta: number): Vec2 => {
  const profile = lorentzFinslerProfile(shape, theta)
  return [Math.sinh(theta) / profile.value, Math.cosh(theta) / profile.value]
}

const closedSamples = (pointAt: (parameter: number) => Vec2, count = 512) => Array.from({ length: count }, (_, index) => pointAt(2 * Math.PI * index / count))

const openSamples = (pointAt: (parameter: number) => Vec2, rapidity = lorentzDisplayRapidity, count = 501) => Array.from({ length: count }, (_, index) => pointAt(-rapidity + 2 * rapidity * index / (count - 1)))

const translateSamples = (samples: Vec2[], center: Vec2, factor = 1) => samples.map((point) => add(center, scale(factor, point)))

const triangleValid = (vertices: [Vec2, Vec2, Vec2]) => {
  const area = Math.abs(cross(subtract(vertices[1], vertices[0]), subtract(vertices[2], vertices[0]))) / 2
  const separation = Math.min(squaredDistance(vertices[0], vertices[1]), squaredDistance(vertices[1], vertices[2]), squaredDistance(vertices[2], vertices[0]))
  return area > 0.015 && separation > 0.003
}

export const buildIndicatrixConstruction = (mode: IndicatrixMode, positions: [number, number, number], normedShape: NormedShape, lorentzShape: LorentzFinslerShape, renderRapidity = lorentzRenderRapidity): IndicatrixConstruction => {
  const base = mode === 'normed'
    ? closedSamples((parameter) => normedIndicatrixPoint(normedShape, parameter))
    : openSamples((parameter) => lorentzFinslerIndicatrixPoint(lorentzShape, parameter), renderRapidity, Math.max(501, 1 + Math.round(300 * renderRapidity)))
  const pointAt = mode === 'normed'
    ? (parameter: number) => normedIndicatrixPoint(normedShape, parameter)
    : (parameter: number) => lorentzFinslerIndicatrixPoint(lorentzShape, parameter)
  const vertices = positions.map(pointAt) as [Vec2, Vec2, Vec2]
  const [a, b, c] = vertices
  const origin: Vec2 = [0, 0]
  const orthocenter = sum(vertices)
  const centroid = scale(1 / 3, orthocenter)
  const feuerbachCenter = scale(0.5, orthocenter)
  const translatedCenters: [Vec2, Vec2, Vec2] = [add(b, c), add(c, a), add(a, b)]
  const translatedIndicatrices = translatedCenters.map((center) => translateSamples(base, center)) as [Vec2[], Vec2[], Vec2[]]
  const sideMidpoints: [Vec2, Vec2, Vec2] = [mean(b, c), mean(c, a), mean(a, b)]
  const vertexOrthocenterMidpoints: [Vec2, Vec2, Vec2] = [mean(a, orthocenter), mean(b, orthocenter), mean(c, orthocenter)]
  const feuBranches = mode === 'normed'
    ? [translateSamples(base, feuerbachCenter, 0.5)]
    : [translateSamples(base, feuerbachCenter, 0.5), translateSamples(base, feuerbachCenter, -0.5)]
  return {
    mode,
    vertices,
    circumIndicatrix: base,
    translatedCenters,
    translatedIndicatrices,
    feuBranches,
    origin,
    centroid,
    orthocenter,
    feuerbachCenter,
    sideMidpoints,
    vertexOrthocenterMidpoints,
    valid: triangleValid(vertices)
  }
}

const refineNearest = (pointAt: (parameter: number) => Vec2, target: Vec2, start: number, end: number) => {
  let lower = start
  let upper = end
  for (let index = 0; index < 40; index += 1) {
    const left = (2 * lower + upper) / 3
    const right = (lower + 2 * upper) / 3
    if (squaredDistance(pointAt(left), target) <= squaredDistance(pointAt(right), target)) {
      upper = right
    } else {
      lower = left
    }
  }
  return (lower + upper) / 2
}

export const nearestNormedParameter = (shape: NormedShape, target: Vec2) => {
  const count = 1440
  let index = 0
  let distance = Infinity
  for (let candidate = 0; candidate < count; candidate += 1) {
    const parameter = 2 * Math.PI * candidate / count
    const next = squaredDistance(normedIndicatrixPoint(shape, parameter), target)
    if (next < distance) {
      distance = next
      index = candidate
    }
  }
  const step = 2 * Math.PI / count
  const value = refineNearest((parameter) => normedIndicatrixPoint(shape, parameter), target, 2 * Math.PI * index / count - step, 2 * Math.PI * index / count + step)
  return ((value % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
}

export const nearestLorentzFinslerParameter = (shape: LorentzFinslerShape, target: Vec2) => {
  const count = 1200
  let index = 0
  let distance = Infinity
  for (let candidate = 0; candidate < count; candidate += 1) {
    const parameter = -lorentzDisplayRapidity + 2 * lorentzDisplayRapidity * candidate / (count - 1)
    const next = squaredDistance(lorentzFinslerIndicatrixPoint(shape, parameter), target)
    if (next < distance) {
      distance = next
      index = candidate
    }
  }
  const step = 2 * lorentzDisplayRapidity / (count - 1)
  return Math.max(-lorentzDisplayRapidity, Math.min(lorentzDisplayRapidity, refineNearest((parameter) => lorentzFinslerIndicatrixPoint(shape, parameter), target, -lorentzDisplayRapidity + index * step - step, -lorentzDisplayRapidity + index * step + step)))
}
