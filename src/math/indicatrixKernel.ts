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

const euclideanCircle = (vertices: [Vec2, Vec2, Vec2]) => {
  const [a, b, c] = vertices
  const denominator = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
  if (Math.abs(denominator) <= 1e-10) {
    const center = scale(1 / 3, sum(vertices))
    const radius = Math.max(0.05, ...vertices.map((vertex) => Math.hypot(vertex[0] - center[0], vertex[1] - center[1])))
    return { center, radius }
  }
  const squared = (point: Vec2) => point[0] * point[0] + point[1] * point[1]
  const center: Vec2 = [
    (squared(a) * (b[1] - c[1]) + squared(b) * (c[1] - a[1]) + squared(c) * (a[1] - b[1])) / denominator,
    (squared(a) * (c[0] - b[0]) + squared(b) * (a[0] - c[0]) + squared(c) * (b[0] - a[0])) / denominator
  ]
  return { center, radius: Math.max(0.05, Math.hypot(a[0] - center[0], a[1] - center[1])) }
}

const solve = (matrix: number[][], vector: number[]) => {
  const augmented = matrix.map((row, index) => [...row, vector[index]!])
  for (let column = 0; column < vector.length; column += 1) {
    let pivot = column
    for (let row = column + 1; row < vector.length; row += 1) {
      if (Math.abs(augmented[row]![column]!) > Math.abs(augmented[pivot]![column]!)) {
        pivot = row
      }
    }
    if (Math.abs(augmented[pivot]![column]!) <= 1e-12) {
      return null
    }
    ;[augmented[column], augmented[pivot]] = [augmented[pivot]!, augmented[column]!]
    const divisor = augmented[column]![column]!
    for (let entry = column; entry <= vector.length; entry += 1) {
      augmented[column]![entry]! /= divisor
    }
    for (let row = 0; row < vector.length; row += 1) {
      if (row === column) {
        continue
      }
      const factor = augmented[row]![column]!
      for (let entry = column; entry <= vector.length; entry += 1) {
        augmented[row]![entry]! -= factor * augmented[column]![entry]!
      }
    }
  }
  return augmented.map((row) => row[vector.length]!)
}

const wrapAngle = (value: number) => ((value % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)

const fitCircumIndicatrix = (mode: IndicatrixMode, vertices: [Vec2, Vec2, Vec2], normedShape: NormedShape, lorentzShape: LorentzFinslerShape) => {
  const initial = euclideanCircle(vertices)
  const pointAt = mode === 'normed'
    ? (parameter: number) => normedIndicatrixPoint(normedShape, parameter)
    : (parameter: number) => lorentzFinslerIndicatrixPoint(lorentzShape, parameter)
  const parameterAt = mode === 'normed'
    ? (point: Vec2) => nearestNormedParameter(normedShape, point)
    : (point: Vec2) => nearestLorentzFinslerParameter(lorentzShape, point)
  const normalize = (parameter: number) => mode === 'normed' ? wrapAngle(parameter) : Math.max(-4, Math.min(4, parameter))
  let values = [initial.center[0], initial.center[1], Math.log(initial.radius), ...vertices.map((vertex) => parameterAt(scale(1 / initial.radius, subtract(vertex, initial.center))))]
  const evaluate = (candidate: number[]) => vertices.flatMap((vertex, index) => {
    const unit = pointAt(candidate[index + 3]!)
    const radius = Math.exp(candidate[2]!)
    return [candidate[0]! + radius * unit[0] - vertex[0], candidate[1]! + radius * unit[1] - vertex[1]]
  })
  let residuals = evaluate(values)
  let norm = Math.hypot(...residuals)
  for (let iteration = 0; iteration < 48 && norm > 1e-9; iteration += 1) {
    const jacobian = Array.from({ length: 6 }, () => Array(6).fill(0))
    const radius = Math.exp(values[2]!)
    vertices.forEach((_, index) => {
      const unit = pointAt(values[index + 3]!)
      const step = 1e-5
      const before = pointAt(values[index + 3]! - step)
      const after = pointAt(values[index + 3]! + step)
      const derivative: Vec2 = [(after[0] - before[0]) / (2 * step), (after[1] - before[1]) / (2 * step)]
      const row = 2 * index
      jacobian[row]![0] = 1
      jacobian[row + 1]![1] = 1
      jacobian[row]![2] = radius * unit[0]
      jacobian[row + 1]![2] = radius * unit[1]
      jacobian[row]![index + 3] = radius * derivative[0]
      jacobian[row + 1]![index + 3] = radius * derivative[1]
    })
    const normal = Array.from({ length: 6 }, (_, row) => Array.from({ length: 6 }, (_, column) => jacobian.reduce((total, current) => total + current[row]! * current[column]!, row === column ? 1e-8 : 0)))
    const right = Array.from({ length: 6 }, (_, column) => -jacobian.reduce((total, row, index) => total + row[column]! * residuals[index]!, 0))
    const delta = solve(normal, right)
    if (!delta) {
      break
    }
    let accepted = false
    for (const factor of [1, 0.5, 0.25, 0.125, 0.0625]) {
      const candidate = values.map((value, index) => value + factor * delta[index]!)
      candidate[2] = Math.max(Math.log(0.02), Math.min(Math.log(20), candidate[2]!))
      for (let index = 3; index < 6; index += 1) {
        candidate[index] = normalize(candidate[index]!)
      }
      const candidateResiduals = evaluate(candidate)
      const candidateNorm = Math.hypot(...candidateResiduals)
      if (candidateNorm < norm) {
        values = candidate
        residuals = candidateResiduals
        norm = candidateNorm
        accepted = true
        break
      }
    }
    if (!accepted) {
      break
    }
  }
  return {
    center: [values[0]!, values[1]!] as Vec2,
    radius: Math.exp(values[2]!),
    parameters: [values[3]!, values[4]!, values[5]!] as [number, number, number],
    valid: Number.isFinite(norm) && norm <= 1e-5
  }
}

export const buildIndicatrixConstruction = (mode: IndicatrixMode, vertices: [Vec2, Vec2, Vec2], normedShape: NormedShape, lorentzShape: LorentzFinslerShape, renderRapidity = lorentzRenderRapidity): IndicatrixConstruction => {
  const base = mode === 'normed'
    ? closedSamples((parameter) => normedIndicatrixPoint(normedShape, parameter))
    : openSamples((parameter) => lorentzFinslerIndicatrixPoint(lorentzShape, parameter), renderRapidity, Math.max(501, 1 + Math.round(300 * renderRapidity)))
  const fit = fitCircumIndicatrix(mode, vertices, normedShape, lorentzShape)
  const [a, b, c] = vertices
  const origin = fit.center
  const orthocenter = subtract(sum(vertices), scale(2, origin))
  const centroid = scale(1 / 3, sum(vertices))
  const feuerbachCenter = mean(origin, orthocenter)
  const translatedCenters: [Vec2, Vec2, Vec2] = [add(origin, subtract(orthocenter, a)), add(origin, subtract(orthocenter, b)), add(origin, subtract(orthocenter, c))]
  const translatedIndicatrices = translatedCenters.map((center) => translateSamples(base, center, fit.radius)) as [Vec2[], Vec2[], Vec2[]]
  const sideMidpoints: [Vec2, Vec2, Vec2] = [mean(b, c), mean(c, a), mean(a, b)]
  const vertexOrthocenterMidpoints: [Vec2, Vec2, Vec2] = [mean(a, orthocenter), mean(b, orthocenter), mean(c, orthocenter)]
  const feuBranches = mode === 'normed'
    ? [translateSamples(base, feuerbachCenter, 0.5 * fit.radius)]
    : [translateSamples(base, feuerbachCenter, 0.5 * fit.radius), translateSamples(base, feuerbachCenter, -0.5 * fit.radius)]
  return {
    mode,
    vertices,
    circumIndicatrix: translateSamples(base, origin, fit.radius),
    translatedCenters,
    translatedIndicatrices,
    feuBranches,
    origin,
    centroid,
    orthocenter,
    feuerbachCenter,
    sideMidpoints,
    vertexOrthocenterMidpoints,
    valid: triangleValid(vertices) && fit.valid
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
