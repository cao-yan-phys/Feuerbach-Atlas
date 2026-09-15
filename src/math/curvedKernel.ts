import {
  cAbs,
  cAdd,
  cMultiply,
  cScale,
  cSqrtReal,
  cSquare,
  cSubtract,
  complex,
  realify,
  type Complex
} from './complex'
import {
  add3,
  bilinear3,
  cross3,
  matVec3,
  scale3,
  solve3,
  EPSILON
} from './linalg'
import type { Mat3, RealCycle3, Vec3 } from './types'

export interface CurvedInvariants {
  p: number
  q: number
  r: number
  delta: number
  d: number
  gram: Mat3
}

export interface CurvedContact {
  kind: 'finite' | 'ideal' | 'unavailable'
  point: Vec3 | null
  determinant: number
  residuals: Record<string, number>
}

export interface CurvedBranch {
  index: number
  label: string
  signs: [number, number, number]
  cycle: RealCycle3 | null
  isReal: boolean
  centerNorm: number | null
  residuals: Record<string, number>
  contact: CurvedContact | null
}

export interface CurvedState {
  invariants: CurvedInvariants
  pseudomedianFeet: [Vec3, Vec3, Vec3] | null
  pseudomedianDirection: Vec3 | null
  eulerNormal: Vec3 | null
  pseudoaltitudePoints: [Vec3 | null, Vec3 | null, Vec3 | null]
  pseudoaltitudeDirection: Vec3 | null
  branches: CurvedBranch[]
  residuals: Record<string, number>
}

interface HalfAngles {
  x: number
  y: number
  z: number
  barX: Complex
  barY: Complex
  barZ: Complex
}

const combine3 = (first: Vec3, firstScale: number, second: Vec3, secondScale: number, third: Vec3, thirdScale: number): Vec3 =>
  add3(add3(scale3(first, firstScale), scale3(second, secondScale)), scale3(third, thirdScale))

const complexBilinear = (metric: Vec3, left: readonly Complex[], right: readonly Complex[]) => {
  let result = complex(0, 0)

  for (let index = 0; index < 3; index += 1) {
    result = cAdd(result, cScale(cMultiply(left[index]!, right[index]!), metric[index]!))
  }

  return result
}

const complexVec = (vector: Vec3): [Complex, Complex, Complex] => [
  complex(vector[0], 0),
  complex(vector[1], 0),
  complex(vector[2], 0)
]

const scaledComplexVec = (vector: Vec3, scalar: Complex): [Complex, Complex, Complex] => [
  cScale(scalar, vector[0]),
  cScale(scalar, vector[1]),
  cScale(scalar, vector[2])
]

const addComplexVec = (left: readonly Complex[], right: readonly Complex[]): [Complex, Complex, Complex] => [
  cAdd(left[0]!, right[0]!),
  cAdd(left[1]!, right[1]!),
  cAdd(left[2]!, right[2]!)
]

const realHalfAngles = (invariants: CurvedInvariants): HalfAngles | null => {
  const xSquared = (1 + invariants.q) / 2
  const ySquared = (1 + invariants.r) / 2
  const zSquared = (1 + invariants.p) / 2

  if (xSquared <= EPSILON || ySquared <= EPSILON || zSquared <= EPSILON) {
    return null
  }

  return {
    x: Math.sqrt(xSquared),
    y: Math.sqrt(ySquared),
    z: Math.sqrt(zSquared),
    barX: cSqrtReal((1 - invariants.q) / 2),
    barY: cSqrtReal((1 - invariants.r) / 2),
    barZ: cSqrtReal((1 - invariants.p) / 2)
  }
}

export const curvedInvariants = (metric: Vec3, vertices: [Vec3, Vec3, Vec3]): CurvedInvariants => {
  const [a, b, c] = vertices
  const p = bilinear3(metric, a, b)
  const q = bilinear3(metric, b, c)
  const r = bilinear3(metric, c, a)

  return {
    p,
    q,
    r,
    delta: 1 + 2 * p * q * r - p * p - q * q - r * r,
    d: 1 + p + q + r,
    gram: [[1, p, r], [p, 1, q], [r, q, 1]]
  }
}

export const sampleGeodesic = (metric: Vec3, start: Vec3, end: Vec3, samples = 48, tolerance = EPSILON, startParameter = 0, endParameter = 1): Vec3[] => {
  const dot = bilinear3(metric, start, end)
  const count = Math.max(2, samples)
  const result: Vec3[] = []

  if (dot > -1 + tolerance && dot < 1 - tolerance) {
    const angle = Math.acos(dot)
    const denominator = Math.sin(angle)

    for (let index = 0; index < count; index += 1) {
      const t = startParameter + (endParameter - startParameter) * index / (count - 1)
      result.push(add3(scale3(start, Math.sin((1 - t) * angle) / denominator), scale3(end, Math.sin(t * angle) / denominator)))
    }

    return result
  }

  if (dot > 1 + tolerance) {
    const angle = Math.acosh(dot)
    const denominator = Math.sinh(angle)

    for (let index = 0; index < count; index += 1) {
      const t = startParameter + (endParameter - startParameter) * index / (count - 1)
      result.push(add3(scale3(start, Math.sinh((1 - t) * angle) / denominator), scale3(end, Math.sinh(t * angle) / denominator)))
    }

    return result
  }

  if (Math.abs(dot - 1) <= tolerance) {
    for (let index = 0; index < count; index += 1) {
      const t = startParameter + (endParameter - startParameter) * index / (count - 1)
      result.push(add3(scale3(start, 1 - t), scale3(end, t)))
    }
  }

  return result
}

export const curvedContact = (metric: Vec3, eulerNormal: Vec3, cycle: RealCycle3, tolerance = 1e-9): CurvedContact => {
  const a = bilinear3(metric, eulerNormal, eulerNormal)
  const b = bilinear3(metric, eulerNormal, cycle.normal)
  const c = bilinear3(metric, cycle.normal, cycle.normal)
  const determinant = a * c - b * b

  if (Math.abs(determinant) > tolerance) {
    const alpha = (c - b * cycle.offset) / determinant
    const beta = (a * cycle.offset - b) / determinant
    const point = add3(scale3(eulerNormal, alpha), scale3(cycle.normal, beta))

    return {
      kind: 'finite',
      point,
      determinant,
      residuals: {
        quadric: Math.abs(bilinear3(metric, point, point) - 1),
        euler: Math.abs(bilinear3(metric, point, eulerNormal) - 1),
        tangent: Math.abs(bilinear3(metric, point, cycle.normal) - cycle.offset)
      }
    }
  }

  const metricEuler = matVec3([[metric[0], 0, 0], [0, metric[1], 0], [0, 0, metric[2]]], eulerNormal)
  const metricCycle = matVec3([[metric[0], 0, 0], [0, metric[1], 0], [0, 0, metric[2]]], cycle.normal)
  const direction = cross3(metricEuler, metricCycle)
  const magnitude = Math.hypot(direction[0], direction[1], direction[2])

  if (magnitude <= tolerance) {
    return { kind: 'unavailable', point: null, determinant, residuals: {} }
  }

  const point = scale3(direction, 1 / magnitude)

  return {
    kind: 'ideal',
    point,
    determinant,
    residuals: {
      quadric: Math.abs(bilinear3(metric, point, point)),
      euler: Math.abs(bilinear3(metric, point, eulerNormal)),
      tangent: Math.abs(bilinear3(metric, point, cycle.normal))
    }
  }
}

const pseudomedianFeet = (vertices: [Vec3, Vec3, Vec3], half: HalfAngles): [Vec3, Vec3, Vec3] | null => {
  const [a, b, c] = vertices
  const { x, y, z } = half
  const denominatorA = x * (y * y + z * z + 2 * x * y * z)
  const denominatorB = y * (z * z + x * x + 2 * x * y * z)
  const denominatorC = z * (x * x + y * y + 2 * x * y * z)

  if (Math.abs(denominatorA) <= EPSILON || Math.abs(denominatorB) <= EPSILON || Math.abs(denominatorC) <= EPSILON) {
    return null
  }

  return [
    add3(scale3(b, y * (z + x * y) / denominatorA), scale3(c, z * (y + x * z) / denominatorA)),
    add3(scale3(a, x * (z + x * y) / denominatorB), scale3(c, z * (x + y * z) / denominatorB)),
    add3(scale3(a, x * (y + x * z) / denominatorC), scale3(b, y * (x + y * z) / denominatorC))
  ]
}

const pseudoaltitudePoints = (vertices: [Vec3, Vec3, Vec3], half: HalfAngles): [Vec3 | null, Vec3 | null, Vec3 | null] => {
  const [a, b, c] = vertices
  const { x, y, z } = half
  const denominatorA = x * (2 * x * y * z - y * y - z * z)
  const denominatorB = y * (2 * x * y * z - z * z - x * x)
  const denominatorC = z * (2 * x * y * z - x * x - y * y)

  return [
    Math.abs(denominatorA) <= EPSILON ? null : add3(scale3(b, y * (x * y - z) / denominatorA), scale3(c, z * (x * z - y) / denominatorA)),
    Math.abs(denominatorB) <= EPSILON ? null : add3(scale3(a, x * (x * y - z) / denominatorB), scale3(c, z * (y * z - x) / denominatorB)),
    Math.abs(denominatorC) <= EPSILON ? null : add3(scale3(a, x * (x * z - y) / denominatorC), scale3(b, y * (y * z - x) / denominatorC))
  ]
}

const pseudoaltitudeDirection = (vertices: [Vec3, Vec3, Vec3], points: [Vec3 | null, Vec3 | null, Vec3 | null]): Vec3 | null => {
  if (!points[0] || !points[1] || !points[2]) {
    return null
  }
  const normals = vertices.map((vertex, index) => cross3(vertex, points[index]!)) as [Vec3, Vec3, Vec3]
  const direction = cross3(normals[0], normals[1])
  const directionSize = Math.hypot(direction[0], direction[1], direction[2])
  const thirdResidual = Math.abs(normals[2][0] * direction[0] + normals[2][1] * direction[1] + normals[2][2] * direction[2])
  const scale = Math.max(1, Math.hypot(...normals[2]) * directionSize)
  return directionSize > EPSILON && thirdResidual <= scale * 1e-7 ? direction : null
}

export const buildCurvedState = (metric: Vec3, vertices: [Vec3, Vec3, Vec3]): CurvedState => {
  const invariants = curvedInvariants(metric, vertices)
  const half = realHalfAngles(invariants)
  const emptyState: CurvedState = {
    invariants,
    pseudomedianFeet: null,
    pseudomedianDirection: null,
    eulerNormal: null,
    pseudoaltitudePoints: [null, null, null],
    pseudoaltitudeDirection: null,
    branches: [],
    residuals: {}
  }

  if (!half) {
    return emptyState
  }

  const [a, b, c] = vertices
  const feet = pseudomedianFeet(vertices, half)
  const eulerCoefficients = solve3(invariants.gram, [half.y * half.z / half.x, half.x * half.z / half.y, half.x * half.y / half.z])
  const eulerNormal = eulerCoefficients
    ? combine3(a, eulerCoefficients[0], b, eulerCoefficients[1], c, eulerCoefficients[2])
    : null
  const pseudomedianDirection = combine3(
    a,
    half.x * (half.y + half.x * half.z) * (half.z + half.x * half.y),
    b,
    half.y * (half.z + half.x * half.y) * (half.x + half.y * half.z),
    c,
    half.z * (half.x + half.y * half.z) * (half.y + half.x * half.z)
  )
  const pseudoaltitude = pseudoaltitudePoints(vertices, half)
  const branchPairs: Array<[number, number, string, [number, number, number]]> = [
    [1, 1, 'I0', [1, 1, 1]],
    [-1, -1, 'Ia', [-1, 1, 1]],
    [-1, 1, 'Ib', [1, -1, 1]],
    [1, -1, 'Ic', [1, 1, -1]]
  ]
  const branches = branchPairs.map(([epsilon, eta, label, signs], index): CurvedBranch => {
    const first = cScale(cMultiply(complex(half.x, 0), half.barX), 2)
    const second = cScale(cMultiply(complex(half.y, 0), half.barY), 2 * epsilon)
    const third = cScale(cMultiply(complex(half.z, 0), half.barZ), 2 * eta)
    const normal = addComplexVec(addComplexVec(scaledComplexVec(a, first), scaledComplexVec(b, second)), scaledComplexVec(c, third))
    const termOne = cScale(half.barX, half.y * half.z)
    const termTwo = cScale(half.barY, epsilon * half.x * half.z)
    const termThree = cScale(half.barZ, eta * half.x * half.y)
    const termFour = cScale(cMultiply(cMultiply(half.barX, half.barY), half.barZ), -epsilon * eta)
    const offset = cScale(cAdd(cAdd(termOne, termTwo), cAdd(termThree, termFour)), 2)
    const realified = realify([...normal, offset])
    const cycle: RealCycle3 | null = realified.values
      ? { normal: [realified.values[0]!, realified.values[1]!, realified.values[2]!], offset: realified.values[3]!, label }
      : null
    const normalNorm = complexBilinear(metric, normal, normal)
    const cycleResidual = cAbs(cSubtract(cSubtract(normalNorm, cSquare(offset)), complex(invariants.delta, 0)))
    const eulerResidual = eulerNormal
      ? cAbs(
          cSubtract(
            cSquare(cSubtract(complexBilinear(metric, complexVec(eulerNormal), normal), offset)),
            cMultiply(
              complex(bilinear3(metric, eulerNormal, eulerNormal) - 1, 0),
              cSubtract(normalNorm, cSquare(offset))
            )
          )
        )
      : Number.NaN
    const contact = cycle && eulerNormal ? curvedContact(metric, eulerNormal, cycle) : null

    return {
      index,
      label,
      signs,
      cycle,
      isReal: cycle !== null,
      centerNorm: cycle ? bilinear3(metric, cycle.normal, cycle.normal) : null,
      residuals: { cycle: cycleResidual, euler: eulerResidual },
      contact
    }
  })
  const residuals: Record<string, number> = {}

  if (feet) {
    feet.forEach((foot, index) => {
      residuals[`foot${index}`] = Math.abs(bilinear3(metric, foot, foot) - 1)
      if (eulerNormal) {
        residuals[`eulerFoot${index}`] = Math.abs(bilinear3(metric, foot, eulerNormal) - 1)
      }
    })
  }

  return {
    invariants,
    pseudomedianFeet: feet,
    pseudomedianDirection,
    eulerNormal,
    pseudoaltitudePoints: pseudoaltitude,
    pseudoaltitudeDirection: pseudoaltitudeDirection(vertices, pseudoaltitude),
    branches,
    residuals
  }
}
