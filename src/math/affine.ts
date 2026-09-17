import { sphereLift, sphereProject } from '../geometry/charts'
import type { Vec2 } from './types'

export interface EuclideanTransform {
  translate: Vec2
  rotation: number
  scale: number
}

export const transformTriangle = (vertices: [Vec2, Vec2, Vec2], transform: EuclideanTransform): [Vec2, Vec2, Vec2] => {
  const center: Vec2 = [
    (vertices[0][0] + vertices[1][0] + vertices[2][0]) / 3,
    (vertices[0][1] + vertices[1][1] + vertices[2][1]) / 3
  ]
  const cosine = Math.cos(transform.rotation)
  const sine = Math.sin(transform.rotation)

  return vertices.map(([x, y]) => {
    const horizontal = x - center[0]
    const vertical = y - center[1]
    return [
      center[0] + transform.translate[0] + transform.scale * (cosine * horizontal - sine * vertical),
      center[1] + transform.translate[1] + transform.scale * (sine * horizontal + cosine * vertical)
    ] as Vec2
  }) as [Vec2, Vec2, Vec2]
}

export const boostTriangle = (vertices: [Vec2, Vec2, Vec2], origin: Vec2, rapidity: number): [Vec2, Vec2, Vec2] => {
  const cosine = Math.cosh(rapidity)
  const sine = Math.sinh(rapidity)
  return vertices.map(([x, t]) => {
    const horizontal = x - origin[0]
    const temporal = t - origin[1]
    return [
      origin[0] + cosine * horizontal + sine * temporal,
      origin[1] + sine * horizontal + cosine * temporal
    ] as Vec2
  }) as [Vec2, Vec2, Vec2]
}

export const transvectLorentzianTriangle = (vertices: [Vec2, Vec2, Vec2], mode: 'desitter' | 'ads', coordinate: number): [Vec2, Vec2, Vec2] | null => {
  if (!Number.isFinite(coordinate) || Math.abs(coordinate) >= 1) {
    return null
  }
  const scale = Math.sqrt(1 - coordinate * coordinate)
  const transformed = vertices.map(([x, y]) => {
    const denominator = mode === 'desitter' ? 1 + y * coordinate : 1 + x * coordinate
    if (!Number.isFinite(denominator) || Math.abs(denominator) <= 1e-12) {
      return null
    }
    return mode === 'desitter'
      ? [x * scale / denominator, (coordinate + y) / denominator] as Vec2
      : [(coordinate + x) / denominator, y * scale / denominator] as Vec2
  })
  return transformed.every((point): point is Vec2 => point !== null) ? transformed as [Vec2, Vec2, Vec2] : null
}

export const transvectHyperbolicTriangle = (vertices: [Vec2, Vec2, Vec2], coordinate: number): [Vec2, Vec2, Vec2] | null => {
  if (!Number.isFinite(coordinate) || Math.abs(coordinate) >= 1) {
    return null
  }
  const transformed = vertices.map(([x, y]) => {
    const radiusSquared = x * x + y * y
    const denominator = 1 + 2 * coordinate * x + coordinate * coordinate * radiusSquared
    if (!Number.isFinite(denominator) || denominator <= 1e-12) {
      return null
    }
    const point: Vec2 = [
      ((1 + coordinate * coordinate) * x + coordinate * (1 + radiusSquared)) / denominator,
      (1 - coordinate * coordinate) * y / denominator
    ]
    return point[0] * point[0] + point[1] * point[1] < 1 ? point : null
  })
  return transformed.every((point): point is Vec2 => point !== null) ? transformed as [Vec2, Vec2, Vec2] : null
}

export const transvectSphereTriangle = (vertices: [Vec2, Vec2, Vec2], angle: number): [Vec2, Vec2, Vec2] | null => {
  if (!Number.isFinite(angle)) {
    return null
  }
  const cosine = Math.cos(angle)
  const sine = Math.sin(angle)
  const transformed = vertices.map((vertex) => {
    const [x0, x1, x2] = sphereLift(vertex)
    return sphereProject([
      cosine * x0 - sine * x1,
      sine * x0 + cosine * x1,
      x2
    ])
  })
  return transformed.every((point): point is Vec2 => point !== null) ? transformed as [Vec2, Vec2, Vec2] : null
}

export const rotateTriangle = (vertices: [Vec2, Vec2, Vec2], origin: Vec2, angle: number): [Vec2, Vec2, Vec2] => {
  const cosine = Math.cos(angle)
  const sine = Math.sin(angle)
  return vertices.map(([x, y]) => {
    const horizontal = x - origin[0]
    const vertical = y - origin[1]
    return [
      origin[0] + cosine * horizontal - sine * vertical,
      origin[1] + sine * horizontal + cosine * vertical
    ] as Vec2
  }) as [Vec2, Vec2, Vec2]
}
