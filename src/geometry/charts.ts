import { bilinear3 } from '../math/linalg'
import type { Vec2, Vec3 } from '../math/types'

export const sphereLift = ([u, v]: Vec2): Vec3 => {
  const denominator = 1 + u * u + v * v
  return [(1 - u * u - v * v) / denominator, 2 * u / denominator, 2 * v / denominator]
}

export const sphereProject = ([x0, x1, x2]: Vec3): Vec2 | null => {
  const denominator = 1 + x0
  return Math.abs(denominator) <= 1e-10 ? null : [x1 / denominator, x2 / denominator]
}

export const sphereFromLatitudeLongitude = (latitude: number, longitude: number): Vec3 => [
  Math.cos(latitude) * Math.cos(longitude),
  Math.cos(latitude) * Math.sin(longitude),
  Math.sin(latitude)
]

export const hyperbolicLift = ([u, v]: Vec2): Vec3 | null => {
  const radiusSquared = u * u + v * v
  const denominator = 1 - radiusSquared
  return denominator <= 1e-10 ? null : [(1 + radiusSquared) / denominator, 2 * u / denominator, 2 * v / denominator]
}

export const hyperbolicProject = ([x0, x1, x2]: Vec3): Vec2 | null => {
  const denominator = x0 + 1
  return Math.abs(denominator) <= 1e-10 ? null : [x1 / denominator, x2 / denominator]
}

export const hyperbolicFromPolar = (rho: number, theta: number): Vec3 => [
  Math.cosh(rho),
  Math.sinh(rho) * Math.cos(theta),
  Math.sinh(rho) * Math.sin(theta)
]

export const lorentzProjectiveLift = ([u, v]: Vec2): Vec3 | null => {
  const denominator = 1 + u * u - v * v
  return denominator <= 1e-10 ? null : [1 / Math.sqrt(denominator), u / Math.sqrt(denominator), v / Math.sqrt(denominator)]
}

export const lorentzProjectiveProject = ([x0, x1, x2]: Vec3): Vec2 | null =>
  Math.abs(x0) <= 1e-10 ? null : [x1 / x0, x2 / x0]

export const adsProjectiveLift = ([u, v]: Vec2): Vec3 | null => {
  const denominator = 1 + v * v - u * u
  return denominator <= 1e-10 ? null : [1 / Math.sqrt(denominator), v / Math.sqrt(denominator), u / Math.sqrt(denominator)]
}

export const adsProjectiveProject = ([x0, x1, x2]: Vec3): Vec2 | null =>
  Math.abs(x0) <= 1e-10 ? null : [x2 / x0, x1 / x0]

export const antiDeSitterFromConformal = (chi: number, tau: number): Vec3 => [
  Math.cos(tau) / Math.cos(chi),
  Math.sin(tau) / Math.cos(chi),
  Math.tan(chi)
]

export const deSitterFromConformal = (theta: number, eta: number): Vec3 => [
  Math.cos(theta) / Math.cos(eta),
  Math.sin(theta) / Math.cos(eta),
  Math.tan(eta)
]

export const isUnitSphere = (point: Vec3) => Math.abs(bilinear3([1, 1, 1], point, point) - 1)

export const isUnitHyperboloid = (point: Vec3) => Math.abs(bilinear3([1, -1, -1], point, point) - 1)

export const isUnitLorentzQuadric = (point: Vec3) => Math.abs(bilinear3([1, 1, -1], point, point) - 1)
