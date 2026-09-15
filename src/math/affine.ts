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
