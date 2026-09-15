import type { Mat2, Mat3, Vec2, Vec3 } from './types'

export const EPSILON = 1e-10

export const add2 = (left: Vec2, right: Vec2): Vec2 => [
  left[0] + right[0],
  left[1] + right[1]
]

export const subtract2 = (left: Vec2, right: Vec2): Vec2 => [
  left[0] - right[0],
  left[1] - right[1]
]

export const scale2 = (value: Vec2, scalar: number): Vec2 => [
  value[0] * scalar,
  value[1] * scalar
]

export const add3 = (left: Vec3, right: Vec3): Vec3 => [
  left[0] + right[0],
  left[1] + right[1],
  left[2] + right[2]
]

export const subtract3 = (left: Vec3, right: Vec3): Vec3 => [
  left[0] - right[0],
  left[1] - right[1],
  left[2] - right[2]
]

export const scale3 = (value: Vec3, scalar: number): Vec3 => [
  value[0] * scalar,
  value[1] * scalar,
  value[2] * scalar
]

export const bilinear2 = (sigma: 1 | -1, left: Vec2, right: Vec2) =>
  left[0] * right[0] + sigma * left[1] * right[1]

export const bilinear3 = (metric: Vec3, left: Vec3, right: Vec3) =>
  metric[0] * left[0] * right[0] +
  metric[1] * left[1] * right[1] +
  metric[2] * left[2] * right[2]

export const matVec3 = (matrix: Mat3, vector: Vec3): Vec3 => [
  matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
  matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
  matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]
]

export const cross3 = (left: Vec3, right: Vec3): Vec3 => [
  left[1] * right[2] - left[2] * right[1],
  left[2] * right[0] - left[0] * right[2],
  left[0] * right[1] - left[1] * right[0]
]

export const determinant2 = (matrix: Mat2) =>
  matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

export const determinant3 = (matrix: Mat3) =>
  matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
  matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
  matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])

export const solve2 = (matrix: Mat2, right: Vec2, tolerance = EPSILON): Vec2 | null => {
  const determinant = determinant2(matrix)

  if (Math.abs(determinant) <= tolerance) {
    return null
  }

  return [
    (right[0] * matrix[1][1] - matrix[0][1] * right[1]) / determinant,
    (matrix[0][0] * right[1] - right[0] * matrix[1][0]) / determinant
  ]
}

export const solve3 = (matrix: Mat3, right: Vec3, tolerance = EPSILON): Vec3 | null => {
  const augmented = [
    [matrix[0][0], matrix[0][1], matrix[0][2], right[0]],
    [matrix[1][0], matrix[1][1], matrix[1][2], right[1]],
    [matrix[2][0], matrix[2][1], matrix[2][2], right[2]]
  ]

  for (let pivotColumn = 0; pivotColumn < 3; pivotColumn += 1) {
    let pivotRow = pivotColumn

    for (let row = pivotColumn + 1; row < 3; row += 1) {
      if (Math.abs(augmented[row]![pivotColumn]!) > Math.abs(augmented[pivotRow]![pivotColumn]!)) {
        pivotRow = row
      }
    }

    if (Math.abs(augmented[pivotRow]![pivotColumn]!) <= tolerance) {
      return null
    }

    if (pivotRow !== pivotColumn) {
      const exchanged = augmented[pivotColumn]!
      augmented[pivotColumn] = augmented[pivotRow]!
      augmented[pivotRow] = exchanged
    }

    const pivot = augmented[pivotColumn]![pivotColumn]!

    for (let column = pivotColumn; column < 4; column += 1) {
      augmented[pivotColumn]![column]! /= pivot
    }

    for (let row = 0; row < 3; row += 1) {
      if (row === pivotColumn) {
        continue
      }

      const factor = augmented[row]![pivotColumn]!

      for (let column = pivotColumn; column < 4; column += 1) {
        augmented[row]![column]! -= factor * augmented[pivotColumn]![column]!
      }
    }
  }

  return [augmented[0]![3]!, augmented[1]![3]!, augmented[2]![3]!]
}

export const normalizeToQuadric = (metric: Vec3, value: Vec3, tolerance = EPSILON): Vec3 | null => {
  const squaredNorm = bilinear3(metric, value, value)

  if (squaredNorm <= tolerance) {
    return null
  }

  return scale3(value, 1 / Math.sqrt(squaredNorm))
}

export const isNearZero = (value: number, tolerance = EPSILON) => Math.abs(value) <= tolerance
