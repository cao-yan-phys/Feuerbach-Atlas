import {
  add2,
  bilinear2,
  scale2,
  solve2,
  solve3,
  subtract2,
  EPSILON
} from './linalg'
import type { FlatCycle, TriangleKind, Vec2 } from './types'

export interface FlatContact {
  point: Vec2 | null
  discriminant: number | null
}

export interface FlatHomothety {
  center: Vec2
  scale: number
}

export interface FlatState {
  kind: TriangleKind
  valid: boolean
  isNonMixed: boolean
  sideSquaredNorms: [number, number, number]
  sideLambdas: [number, number, number]
  midpoints: [Vec2, Vec2, Vec2]
  altitudeFeet: [Vec2 | null, Vec2 | null, Vec2 | null]
  circumcenter: Vec2 | null
  circumcircle: FlatCycle | null
  centroid: Vec2
  orthocenter: Vec2 | null
  ninePoint: FlatCycle | null
  tangentCycles: FlatCycle[]
  contacts: FlatContact[]
  residuals: Record<string, number>
}

interface Sideline {
  normal: Vec2
  offset: number
  causalSign: 1 | -1
}

const mean2 = (left: Vec2, right: Vec2): Vec2 => scale2(add2(left, right), 0.5)

const flatNorm = (sigma: 1 | -1, value: Vec2) => bilinear2(sigma, value, value)

export const classifyMinkowskiTriangle = (vertices: [Vec2, Vec2, Vec2], tolerance = EPSILON): {
  kind: TriangleKind
  sideSquaredNorms: [number, number, number]
} => {
  const [a, b, c] = vertices
  const sideSquaredNorms: [number, number, number] = [
    flatNorm(-1, subtract2(b, c)),
    flatNorm(-1, subtract2(c, a)),
    flatNorm(-1, subtract2(a, b))
  ]

  if (sideSquaredNorms.some((value) => Math.abs(value) <= tolerance)) {
    return { kind: 'null', sideSquaredNorms }
  }

  if (sideSquaredNorms.every((value) => value > 0)) {
    return { kind: 'spacelike', sideSquaredNorms }
  }

  if (sideSquaredNorms.every((value) => value < 0)) {
    return { kind: 'timelike', sideSquaredNorms }
  }

  return { kind: 'mixed', sideSquaredNorms }
}

const circumcenter = (sigma: 1 | -1, vertices: [Vec2, Vec2, Vec2]): Vec2 | null => {
  const [a, b, c] = vertices
  const firstDirection = subtract2(b, a)
  const secondDirection = subtract2(c, a)

  return solve2(
    [
      [firstDirection[0], sigma * firstDirection[1]],
      [secondDirection[0], sigma * secondDirection[1]]
    ],
    [
      (flatNorm(sigma, b) - flatNorm(sigma, a)) / 2,
      (flatNorm(sigma, c) - flatNorm(sigma, a)) / 2
    ]
  )
}

export const euclideanCircumcircle = (vertices: [Vec2, Vec2, Vec2]): FlatCycle | null => {
  const center = circumcenter(1, vertices)
  return center ? { center, radiusSquared: flatNorm(1, subtract2(vertices[0], center)), label: 'Euclidean circumcircle' } : null
}

export const minkowskiCircumcycle = (vertices: [Vec2, Vec2, Vec2]): FlatCycle | null => {
  const center = circumcenter(-1, vertices)
  return center ? { center, radiusSquared: flatNorm(-1, subtract2(vertices[0], center)), label: 'Minkowski circumcycle' } : null
}

const altitudeFoot = (sigma: 1 | -1, vertex: Vec2, sideStart: Vec2, sideEnd: Vec2): Vec2 | null => {
  const side = subtract2(sideEnd, sideStart)
  const rawNormal: Vec2 = [sigma * side[1], -side[0]]
  const squaredNorm = flatNorm(sigma, rawNormal)

  if (Math.abs(squaredNorm) <= EPSILON) {
    return null
  }

  const normal = scale2(rawNormal, 1 / Math.sqrt(Math.abs(squaredNorm)))
  const offset = bilinear2(sigma, sideStart, normal)
  const sign = Math.sign(squaredNorm)

  return add2(vertex, scale2(normal, (offset - bilinear2(sigma, vertex, normal)) / sign))
}

const orientedSideline = (sigma: 1 | -1, sideStart: Vec2, sideEnd: Vec2, opposite: Vec2): Sideline | null => {
  const side = subtract2(sideEnd, sideStart)
  const rawNormal: Vec2 = [sigma * side[1], -side[0]]
  const squaredNorm = flatNorm(sigma, rawNormal)

  if (Math.abs(squaredNorm) <= EPSILON) {
    return null
  }

  let normal = scale2(rawNormal, 1 / Math.sqrt(Math.abs(squaredNorm)))
  let offset = bilinear2(sigma, sideStart, normal)

  if (bilinear2(sigma, opposite, normal) - offset < 0) {
    normal = scale2(normal, -1)
    offset *= -1
  }

  return { normal, offset, causalSign: Math.sign(squaredNorm) as 1 | -1 }
}

const tangentCycles = (sigma: 1 | -1, vertices: [Vec2, Vec2, Vec2]): FlatCycle[] => {
  const [a, b, c] = vertices
  const sidelines = [
    orientedSideline(sigma, b, c, a),
    orientedSideline(sigma, c, a, b),
    orientedSideline(sigma, a, b, c)
  ]

  if (sidelines.some((line) => line === null)) {
    return []
  }

  const [lineA, lineB, lineC] = sidelines as [Sideline, Sideline, Sideline]

  if (lineA.causalSign !== lineB.causalSign || lineA.causalSign !== lineC.causalSign) {
    return []
  }

  const branches: Array<[string, [number, number, number]]> = [
    ['I0', [1, 1, 1]],
    ['Ia', [-1, 1, 1]],
    ['Ib', [1, -1, 1]],
    ['Ic', [1, 1, -1]]
  ]

  return branches.flatMap(([label, signs]) => {
    const solution = solve3(
      [
        [lineA.normal[0], sigma * lineA.normal[1], -signs[0]],
        [lineB.normal[0], sigma * lineB.normal[1], -signs[1]],
        [lineC.normal[0], sigma * lineC.normal[1], -signs[2]]
      ],
      [lineA.offset, lineB.offset, lineC.offset]
    )

    if (!solution) {
      return []
    }

    return [{ center: [solution[0], solution[1]] as Vec2, radiusSquared: lineA.causalSign * solution[2] * solution[2], label }]
  })
}

export const flatCycleContact = (sigma: 1 | -1, first: FlatCycle, second: FlatCycle, tolerance = 1e-8): FlatContact => {
  const difference = subtract2(second.center, first.center)
  const differenceNorm = flatNorm(sigma, difference)

  if (Math.abs(differenceNorm) <= tolerance) {
    return { point: null, discriminant: null }
  }

  const firstConstant = flatNorm(sigma, first.center) - first.radiusSquared
  const secondConstant = flatNorm(sigma, second.center) - second.radiusSquared
  const particular = scale2(difference, (secondConstant - firstConstant) / (2 * differenceNorm))
  const direction: Vec2 = [sigma * difference[1], -difference[0]]
  const relative = subtract2(particular, first.center)
  const quadraticA = flatNorm(sigma, direction)
  const quadraticB = 2 * bilinear2(sigma, relative, direction)
  const quadraticC = flatNorm(sigma, relative) - first.radiusSquared
  const discriminant = quadraticB * quadraticB - 4 * quadraticA * quadraticC
  const scale = Math.max(1, Math.abs(quadraticB * quadraticB), Math.abs(4 * quadraticA * quadraticC))

  if (Math.abs(quadraticA) <= tolerance || Math.abs(discriminant) > tolerance * scale) {
    return { point: null, discriminant }
  }

  const root = -quadraticB / (2 * quadraticA)

  return { point: add2(particular, scale2(direction, root)), discriminant }
}

export const flatCycleHomothety = (first: FlatCycle, second: FlatCycle, contact: FlatContact, tolerance = 1e-8): FlatHomothety | null => {
  if (!contact.point) {
    return null
  }

  const source = subtract2(first.center, contact.point)
  const target = subtract2(second.center, contact.point)
  const index = Math.abs(source[0]) >= Math.abs(source[1]) ? 0 : 1
  const denominator = source[index]
  if (Math.abs(denominator) <= tolerance) {
    return null
  }

  const scale = target[index] / denominator
  const centerResidual = Math.hypot(target[0] - scale * source[0], target[1] - scale * source[1])
  const radiusResidual = Math.abs(second.radiusSquared - scale * scale * first.radiusSquared)
  const scaleSize = Math.max(1, Math.hypot(...target), Math.abs(scale) * Math.hypot(...source))
  const radiusSize = Math.max(1, Math.abs(second.radiusSquared), Math.abs(scale * scale * first.radiusSquared))
  if (!Number.isFinite(scale) || Math.abs(scale) <= tolerance || Math.abs(scale - 1) <= tolerance || centerResidual > tolerance * scaleSize || radiusResidual > tolerance * radiusSize) {
    return null
  }

  return { center: contact.point, scale }
}

export const buildFlatState = (sigma: 1 | -1, vertices: [Vec2, Vec2, Vec2]): FlatState => {
  const [a, b, c] = vertices
  const minkowski = sigma === -1 ? classifyMinkowskiTriangle(vertices) : null
  const kind: TriangleKind = minkowski?.kind ?? 'riemannian'
  const sideSquaredNorms: [number, number, number] = minkowski?.sideSquaredNorms ?? [
    flatNorm(1, subtract2(b, c)),
    flatNorm(1, subtract2(c, a)),
    flatNorm(1, subtract2(a, b))
  ]
  const sideLengths = sideSquaredNorms.map((value) => Math.sqrt(Math.abs(value))) as [number, number, number]
  const sideLambdas: [number, number, number] = [
    Math.sign(sideLengths[1] + sideLengths[2] - sideLengths[0]),
    Math.sign(sideLengths[2] + sideLengths[0] - sideLengths[1]),
    Math.sign(sideLengths[0] + sideLengths[1] - sideLengths[2])
  ]
  const midpoints: [Vec2, Vec2, Vec2] = [mean2(b, c), mean2(c, a), mean2(a, b)]
  const center = circumcenter(sigma, vertices)
  const circumcircle = center
    ? { center, radiusSquared: flatNorm(sigma, subtract2(a, center)), label: 'Circumcircle' }
    : null
  const centroid = scale2(add2(add2(a, b), c), 1 / 3)
  const orthocenter = center ? add2(add2(a, b), add2(c, scale2(center, -2))) : null
  const nineCenter = center && orthocenter ? mean2(center, orthocenter) : null
  const ninePoint = circumcircle && nineCenter
    ? { center: nineCenter, radiusSquared: circumcircle.radiusSquared / 4, label: 'Euler' }
    : null
  const altitudeFeet: [Vec2 | null, Vec2 | null, Vec2 | null] = [
    altitudeFoot(sigma, a, b, c),
    altitudeFoot(sigma, b, c, a),
    altitudeFoot(sigma, c, a, b)
  ]
  const isNonMixed = kind === 'riemannian' || kind === 'spacelike' || kind === 'timelike'
  const valid = center !== null && kind !== 'null'
  const cycles = isNonMixed ? tangentCycles(sigma, vertices) : []
  const contacts = ninePoint ? cycles.map((cycle) => flatCycleContact(sigma, ninePoint, cycle)) : []
  const residuals: Record<string, number> = {}

  if (ninePoint) {
    midpoints.forEach((point, index) => {
      residuals[`midpoint${index}`] = Math.abs(flatNorm(sigma, subtract2(point, ninePoint.center)) - ninePoint.radiusSquared)
    })
    altitudeFeet.forEach((point, index) => {
      if (point) {
        residuals[`altitude${index}`] = Math.abs(flatNorm(sigma, subtract2(point, ninePoint.center)) - ninePoint.radiusSquared)
      }
    })
  }

  return {
    kind,
    valid,
    isNonMixed,
    sideSquaredNorms,
    sideLambdas,
    midpoints,
    altitudeFeet,
    circumcenter: center,
    circumcircle,
    centroid,
    orthocenter,
    ninePoint,
    tangentCycles: cycles,
    contacts,
    residuals
  }
}
