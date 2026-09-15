import { cross3, determinant3 } from './linalg'
import type { Vec2, Vec3 } from './types'

export type ParabolicMode = 'nhnegative' | 'galilei' | 'nhpositive'

export type ParabolicChart = 'natural' | 'beltrami'

export type ParabolicView = 'galilei' | 'carroll'

export interface ParabolicParameters {
  u: number
  v: number
  w: number
  p: number
  q: number
  anchor: Vec2
}

export interface ParabolicPoint {
  T: number
  y: number
  vector: Vec3
}

export interface ParabolicSide {
  a: number
  b: number
}

export interface ParabolicCycle {
  c: number
  A: number
  b: number
}

export interface ParabolicState {
  kappa: number
  parameters: ParabolicParameters
  valid: boolean
  message: string
  theta: number
  sides: [ParabolicSide, ParabolicSide, ParabolicSide]
  feet: [ParabolicPoint, ParabolicPoint, ParabolicPoint] | null
  secondIntersections: [ParabolicPoint | null, ParabolicPoint | null, ParabolicPoint | null]
  pseudomedianCenter: ParabolicPoint | null
  euler: ParabolicCycle | null
  tangent: ParabolicCycle | null
  contact: ParabolicPoint | null
  contactTime: number | null
  fibers: [number, number, number]
  residuals: Record<string, number>
}

const tolerance = 1e-10

const scaled = (value: number, factor: number) => Math.abs(value * factor)

export const parabolicModeForKappa = (kappa: number): ParabolicMode => kappa < -tolerance ? 'nhnegative' : kappa > tolerance ? 'nhpositive' : 'galilei'

export const parabolicDefaultKappa = (mode: ParabolicMode) => mode === 'nhnegative' ? -0.36 : mode === 'nhpositive' ? 0.36 : 0

export const isParabolicMode = (mode: string): mode is ParabolicMode => mode === 'nhnegative' || mode === 'galilei' || mode === 'nhpositive'

export const Ck = (kappa: number, time: number) => {
  const magnitude = scaled(kappa, time * time)
  if (magnitude < 1e-7) {
    return 1 + kappa * time * time / 2 + kappa * kappa * time ** 4 / 24
  }
  if (kappa > 0) {
    return Math.cosh(Math.sqrt(kappa) * time)
  }
  if (kappa < 0) {
    return Math.cos(Math.sqrt(-kappa) * time)
  }
  return 1
}

export const Sk = (kappa: number, time: number) => {
  const magnitude = scaled(kappa, time * time)
  if (magnitude < 1e-7) {
    return time + kappa * time ** 3 / 6 + kappa * kappa * time ** 5 / 120
  }
  if (kappa > 0) {
    return Math.sinh(Math.sqrt(kappa) * time) / Math.sqrt(kappa)
  }
  if (kappa < 0) {
    return Math.sin(Math.sqrt(-kappa) * time) / Math.sqrt(-kappa)
  }
  return time
}

export const Pk = (kappa: number, time: number) => {
  const magnitude = scaled(kappa, time * time)
  if (magnitude < 1e-7) {
    return time * time / 2 + kappa * time ** 4 / 24 + kappa * kappa * time ** 6 / 720
  }
  if (kappa > 0) {
    return 2 * Math.sinh(Math.sqrt(kappa) * time / 2) ** 2 / kappa
  }
  if (kappa < 0) {
    return -2 * Math.sin(Math.sqrt(-kappa) * time / 2) ** 2 / kappa
  }
  return time * time / 2
}

export const Tk = (kappa: number, time: number) => Sk(kappa, time) / Ck(kappa, time)

export const inverseTk = (kappa: number, value: number): number | null => {
  if (kappa > tolerance) {
    const scaledValue = Math.sqrt(kappa) * value
    return Math.abs(scaledValue) < 1 ? Math.atanh(scaledValue) / Math.sqrt(kappa) : null
  }
  if (kappa < -tolerance) {
    return Math.atan(Math.sqrt(-kappa) * value) / Math.sqrt(-kappa)
  }
  return value
}

export const parabolicChartPoint = (kappa: number, point: Vec2, chart: ParabolicChart): Vec2 | null => {
  if (chart === 'natural') {
    return [point[0], point[1]]
  }
  const cosine = Ck(kappa, point[0])
  if (Math.abs(cosine) <= tolerance) {
    return null
  }
  return [Sk(kappa, point[0]) / cosine, point[1] / cosine]
}

export const parabolicPointFromChart = (kappa: number, point: Vec2, chart: ParabolicChart, referenceTime?: number): Vec2 | null => {
  if (chart === 'natural') {
    return [point[0], point[1]]
  }
  const inverse = inverseTk(kappa, point[0])
  if (inverse === null) {
    return null
  }
  const period = kappa < -tolerance ? Math.PI / Math.sqrt(-kappa) : 0
  const time = period > 0 && Number.isFinite(referenceTime)
    ? inverse + period * Math.round((referenceTime! - inverse) / period)
    : inverse
  return [time, point[1] * Ck(kappa, time)]
}

export const parabolicDisplayPoint = (kappa: number, point: Vec2, chart: ParabolicChart, view: ParabolicView): Vec2 | null => {
  const chartPoint = parabolicChartPoint(kappa, point, chart)
  return chartPoint && chart === 'beltrami' && view === 'galilei' ? [chartPoint[1], chartPoint[0]] : chartPoint
}

export const parabolicPointFromDisplay = (kappa: number, point: Vec2, chart: ParabolicChart, view: ParabolicView, referenceTime?: number): Vec2 | null => {
  const chartPoint: Vec2 = chart === 'beltrami' && view === 'galilei' ? [point[1], point[0]] : point
  return parabolicPointFromChart(kappa, chartPoint, chart, referenceTime)
}

export const parabolicVector = (kappa: number, point: Vec2): Vec3 => [Ck(kappa, point[0]), point[1], Sk(kappa, point[0])]

export const canonicalParabolicParameters = (vertices: [Vec2, Vec2, Vec2], kappa: number): ParabolicParameters => {
  const [a, b, c] = vertices
  const u = b[0] - a[0]
  const v = c[0] - b[0]
  const w = c[0] - a[0]
  return {
    u,
    v,
    w,
    p: b[1] - a[1] * Ck(kappa, u),
    q: c[1] - a[1] * Ck(kappa, w),
    anchor: a
  }
}

export const canonicalToRawParabolic = (point: Vec2, parameters: ParabolicParameters, kappa: number): Vec2 => [
  parameters.anchor[0] + point[0],
  point[1] + parameters.anchor[1] * Ck(kappa, point[0])
]

export const parabolicSideValue = (side: ParabolicSide, kappa: number, time: number) => side.a * Ck(kappa, time) + side.b * Sk(kappa, time)

export const parabolicCycleValue = (cycle: ParabolicCycle, kappa: number, time: number) => cycle.c + cycle.A * Pk(kappa, time) + cycle.b * Sk(kappa, time)

export const parabolicCycleDerivative = (cycle: ParabolicCycle, kappa: number, time: number) => cycle.A * Sk(kappa, time) + cycle.b * Ck(kappa, time)

export const parabolicLineThrough = (kappa: number, first: Vec2, second: Vec2): ParabolicSide | null => {
  const determinant = Ck(kappa, first[0]) * Sk(kappa, second[0]) - Ck(kappa, second[0]) * Sk(kappa, first[0])
  if (Math.abs(determinant) <= tolerance) {
    return null
  }
  return {
    a: (first[1] * Sk(kappa, second[0]) - second[1] * Sk(kappa, first[0])) / determinant,
    b: (Ck(kappa, first[0]) * second[1] - Ck(kappa, second[0]) * first[1]) / determinant
  }
}

const pointFromVector = (kappa: number, vector: Vec3, referenceTime?: number): ParabolicPoint | null => {
  const quadric = vector[0] * vector[0] - kappa * vector[2] * vector[2]
  if (!(quadric > tolerance)) {
    return null
  }
  const normalized = vector.map((value) => value / Math.sqrt(quadric)) as Vec3
  let T: number
  let y: number
  if (kappa > tolerance) {
    const sign = normalized[0] < 0 ? -1 : 1
    T = Math.asinh(Math.sqrt(kappa) * sign * normalized[2]) / Math.sqrt(kappa)
    y = sign * normalized[1]
  } else if (kappa < -tolerance) {
    const root = Math.sqrt(-kappa)
    const period = 2 * Math.PI / root
    const candidates = [1, -1].map((sign) => {
      const base = Math.atan2(sign * root * normalized[2], sign * normalized[0]) / root
      const time = Number.isFinite(referenceTime) ? base + period * Math.round((referenceTime! - base) / period) : base
      return { time, y: sign * normalized[1] }
    })
    const chosen = Number.isFinite(referenceTime)
      ? candidates.reduce((best, candidate) => Math.abs(candidate.time - referenceTime!) < Math.abs(best.time - referenceTime!) ? candidate : best)
      : candidates[normalized[0] < 0 ? 1 : 0]!
    T = chosen.time
    y = chosen.y
  } else {
    T = normalized[2] / normalized[0]
    y = normalized[1] / normalized[0]
  }
  return Number.isFinite(T) && Number.isFinite(y) ? { T, y, vector: parabolicVector(kappa, [T, y]) } : null
}

const pointAt = (kappa: number, time: number, y: number): ParabolicPoint => ({ T: time, y, vector: parabolicVector(kappa, [time, y]) })

const dot = (left: Vec3, right: Vec3) => left[0] * right[0] + left[1] * right[1] + left[2] * right[2]

const projectiveIntersection = (kappa: number, first: Vec3, second: Vec3, third?: Vec3) => {
  const intersection = cross3(first, second)
  return {
    point: pointFromVector(kappa, intersection),
    residual: third ? Math.abs(dot(third, intersection)) / Math.max(1, Math.hypot(...third) * Math.hypot(...intersection)) : 0
  }
}

const finiteDifference = (value: number) => Number.isFinite(value) ? Math.abs(value) : Number.POSITIVE_INFINITY

const secondRoot = (a2: number, a1: number, a0: number, known: number) => {
  if (Math.abs(a2) > tolerance) {
    return -a1 / a2 - known
  }
  if (Math.abs(a1) > tolerance) {
    return -a0 / a1
  }
  return null
}

export const parabolicArea = (kappa: number, first: Vec2, second: Vec2, third: Vec2) => {
  const vectors = [parabolicVector(kappa, first), parabolicVector(kappa, second), parabolicVector(kappa, third)]
  const determinant = determinant3([
    [vectors[0]![0], vectors[1]![0], vectors[2]![0]],
    [vectors[0]![1], vectors[1]![1], vectors[2]![1]],
    [vectors[0]![2], vectors[1]![2], vectors[2]![2]]
  ])
  const denominator = 1 + Ck(kappa, first[0] - second[0]) + Ck(kappa, second[0] - third[0]) + Ck(kappa, third[0] - first[0])
  return 2 * determinant / denominator
}

export const parabolicStripArea = (kappa: number, parameters: Pick<ParabolicParameters, 'u' | 'v' | 'w' | 'p' | 'q'>, slices = 12000) => {
  const { u, v, w, p, q } = parameters
  const su = Sk(kappa, u)
  const sv = Sk(kappa, v)
  const sw = Sk(kappa, w)
  if (Math.abs(su * sv * sw) <= tolerance) {
    return Number.NaN
  }
  let area = 0
  for (let index = 0; index < slices; index += 1) {
    const T = (index + 0.5) * w / slices
    const ab = p * Sk(kappa, T) / su
    const ac = q * Sk(kappa, T) / sw
    const bc = (p * Sk(kappa, w - T) + q * Sk(kappa, T - u)) / sv
    area += ((T <= u ? ab : bc) - ac) * w / slices
  }
  return area
}

export const buildParabolicState = (kappa: number, vertices: [Vec2, Vec2, Vec2]): ParabolicState => {
  const parameters = canonicalParabolicParameters(vertices, kappa)
  const { u, v, w, p, q } = parameters
  const invalid = (message: string): ParabolicState => ({
    kappa,
    parameters,
    valid: false,
    message,
    theta: Number.NaN,
    sides: [{ a: Number.NaN, b: Number.NaN }, { a: Number.NaN, b: Number.NaN }, { a: Number.NaN, b: Number.NaN }],
    feet: null,
    secondIntersections: [null, null, null],
    pseudomedianCenter: null,
    euler: null,
    tangent: null,
    contact: null,
    contactTime: null,
    fibers: [0, u, w],
    residuals: {}
  })

  if (!(u > tolerance && v > tolerance)) {
    return invalid('ordered time requires T_A<T_B<T_C')
  }
  if (kappa < -tolerance && !(w < Math.PI / Math.sqrt(-kappa) - tolerance)) {
    return invalid('principal NH− domain requires w<π/√−κ')
  }

  const su = Sk(kappa, u)
  const sv = Sk(kappa, v)
  const sw = Sk(kappa, w)
  if (Math.abs(su * sv * sw) <= tolerance) {
    return invalid('sideline denominator vanishes')
  }

  const sides: [ParabolicSide, ParabolicSide, ParabolicSide] = [
    (() => {
      const beta = (q - p * Ck(kappa, v)) / sv
      return { a: p * Ck(kappa, u) - beta * Sk(kappa, u), b: beta * Ck(kappa, u) - kappa * p * Sk(kappa, u) }
    })(),
    { a: 0, b: q / sw },
    { a: 0, b: p / su }
  ]
  const vectors: [Vec3, Vec3, Vec3] = [parabolicVector(kappa, [0, 0]), parabolicVector(kappa, [u, p]), parabolicVector(kappa, [w, q])]
  const x = Ck(kappa, v / 2)
  const y = Ck(kappa, w / 2)
  const z = Ck(kappa, u / 2)
  const combination = (leftScale: number, left: Vec3, rightScale: number, right: Vec3, denominator: number) => left.map((value, index) => (leftScale * value + rightScale * right[index]!) / denominator) as Vec3
  const feetVectors: [Vec3, Vec3, Vec3] = [
    combination(y * (z + x * y), vectors[1], z * (y + x * z), vectors[2], x * (y * y + z * z + 2 * x * y * z)),
    combination(z * (x + y * z), vectors[2], x * (z + x * y), vectors[0], y * (z * z + x * x + 2 * x * y * z)),
    combination(x * (y + x * z), vectors[0], y * (x + y * z), vectors[1], z * (x * x + y * y + 2 * x * y * z))
  ]
  const footReferences = [(u + w) / 2, w / 2, u / 2]
  const feet = feetVectors.map((vector, index) => pointFromVector(kappa, vector, footReferences[index])) as [ParabolicPoint | null, ParabolicPoint | null, ParabolicPoint | null]
  if (!feet[0] || !feet[1] || !feet[2]) {
    return invalid('pseudomedian foot is ideal')
  }
  const validFeet = feet as [ParabolicPoint, ParabolicPoint, ParabolicPoint]
  const pseudomedianLines = vectors.map((vertex, index) => cross3(vertex, validFeet[index]!.vector)) as [Vec3, Vec3, Vec3]
  const pseudomedian = projectiveIntersection(kappa, pseudomedianLines[0], pseudomedianLines[1], pseudomedianLines[2])

  const r = Tk(kappa, u / 2)
  const s = Tk(kappa, v / 2)
  const theta = p * (r + s) * (1 + kappa * r * s) - q * r * (1 - kappa * s * s)
  if (Math.abs(theta) <= tolerance) {
    return invalid('Θ=0')
  }
  const phi = p * (r + s) * (1 + kappa * r * s) + q * r * (1 - kappa * s * s)
  const denominator = (1 - kappa * r * r) * (1 + kappa * r * s)
  if (Math.abs(r * s * (r + s) * denominator) <= tolerance) {
    return invalid('Euler denominator vanishes')
  }
  const FB = q * r * (1 - kappa * s * s) / ((r + s) * (1 + kappa * r * s))
  const cE = theta / (s * (1 - kappa * r * r))
  const AE = theta * (1 + kappa * r * s) / (r * s * (r + s) * (1 - kappa * r * r))
  const bE = (FB - cE - AE * Pk(kappa, u)) / su
  const euler: ParabolicCycle = { c: cE, A: AE, b: bE }
  const AB = -(1 - kappa * r * r) * (1 + kappa * s * s) * theta / (8 * r * s * (r + s) * (1 + kappa * r * s))
  const bB = (1 - kappa * r * r) * phi / (4 * r * (r + s) * (1 + kappa * r * s))
  const cB = -(1 - kappa * r * r) * s * theta / (4 * r * (r + s) * (1 + kappa * r * s))
  const tangent: ParabolicCycle = kappa === 0
    ? {
        c: -v * (p * w - q * u) / (4 * u * w),
        A: -(p * w - q * u) / (2 * u * v * w),
        b: (p * w + q * u) / (2 * u * w)
      }
    : { c: cB, A: AB, b: bB }
  const eulerStable: ParabolicCycle = kappa === 0
    ? {
        c: (p * w - q * u) / v,
        A: 4 * (p * w - q * u) / (u * v * w),
        b: (q * u * (3 * u + 2 * v) - p * w * (3 * u + v)) / (u * v * w)
      }
    : euler
  const gamma = 9 - kappa * (r * r + s * s) + 8 * kappa * r * s + kappa * kappa * r * r * s * s
  const LF = theta * gamma / (8 * r * s * (r + s) * (1 + kappa * r * s))
  const RF = (kappa * r * r * s + 2 * r + s) / (3 + 2 * kappa * r * s - kappa * r * r)
  const contactHalf = inverseTk(kappa, RF)
  const contactTime = contactHalf === null ? null : 2 * contactHalf
  const contact = contactTime === null ? null : pointAt(kappa, contactTime, parabolicCycleValue(eulerStable, kappa, contactTime))

  const secondIntersections = sides.map((side, index) => {
    const foot = validFeet[index]!
    const known = Tk(kappa, foot.T / 2)
    const A2 = 2 * eulerStable.A - kappa * (side.a + eulerStable.c)
    const A1 = 2 * (eulerStable.b - side.b)
    const A0 = eulerStable.c - side.a
    const other = secondRoot(A2, A1, A0, known)
    const timeHalf = other === null ? null : inverseTk(kappa, other)
    const time = timeHalf === null ? null : 2 * timeHalf
    return time === null || !Number.isFinite(time) ? null : pointAt(kappa, time, parabolicSideValue(side, kappa, time))
  }) as [ParabolicPoint | null, ParabolicPoint | null, ParabolicPoint | null]
  const FC = p * (r + s) * (1 + kappa * r * s) / (r * (1 - kappa * s * s))
  const area = parabolicArea(kappa, [0, 0], [u, p], [w, q])
  const footArea = validFeet.map((foot, index) => {
    const opposite = [[u, p], [w, q], [0, 0]] as [Vec2, Vec2, Vec2]
    const first = [[0, 0], [u, p], [w, q]] as [Vec2, Vec2, Vec2]
    return Math.abs(parabolicArea(kappa, first[index]!, opposite[index]!, [foot.T, foot.y]) - area / 2)
  })
  const residuals: Record<string, number> = {
    area: finiteDifference(area - parabolicStripArea(kappa, parameters, 2048)),
    foot0: Math.abs(parabolicCycleValue(eulerStable, kappa, validFeet[0].T) - validFeet[0].y),
    foot1: Math.abs(parabolicCycleValue(eulerStable, kappa, validFeet[1].T) - validFeet[1].y),
    foot2: Math.abs(parabolicCycleValue(eulerStable, kappa, validFeet[2].T) - validFeet[2].y),
    euler0: Math.abs(parabolicCycleValue(eulerStable, kappa, 0) - cE),
    eulerU: Math.abs(parabolicCycleValue(eulerStable, kappa, u) - FB),
    eulerW: Math.abs(parabolicCycleValue(eulerStable, kappa, w) - FC),
    pseudomedian: pseudomedian.residual,
    bisection: Math.max(...footArea),
    contact: contactTime === null ? Number.POSITIVE_INFINITY : Math.abs(parabolicCycleValue(eulerStable, kappa, contactTime) - parabolicCycleValue(tangent, kappa, contactTime)),
    derivative: contactTime === null ? Number.POSITIVE_INFINITY : Math.abs(parabolicCycleDerivative(eulerStable, kappa, contactTime) - parabolicCycleDerivative(tangent, kappa, contactTime))
  }
  if (contactTime !== null) {
    ;[-0.6, -0.1, 0.4, 0.9, 1.4, 1.9].forEach((time, index) => {
      residuals[`factor${index}`] = Math.abs(parabolicCycleValue(eulerStable, kappa, time) - parabolicCycleValue(tangent, kappa, time) - LF * Pk(kappa, time - contactTime))
    })
  }
  return {
    kappa,
    parameters,
    valid: Object.values(residuals).every((value) => Number.isFinite(value)) && secondIntersections.every(Boolean),
    message: Object.values(residuals).every((value) => Number.isFinite(value)) && secondIntersections.every(Boolean) ? 'valid' : 'construction reaches an ideal point',
    theta,
    sides,
    feet: validFeet,
    secondIntersections,
    pseudomedianCenter: pseudomedian.point,
    euler: eulerStable,
    tangent,
    contact,
    contactTime,
    fibers: [0, u, w],
    residuals
  }
}
