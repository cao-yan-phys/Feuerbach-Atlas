export type Vec2 = [number, number]
export type Vec3 = [number, number, number]
export type Mat2 = [[number, number], [number, number]]
export type Mat3 = [[number, number, number], [number, number, number], [number, number, number]]

export interface RealCycle3 {
  normal: Vec3
  offset: number
  label: string
}

export interface FlatCycle {
  center: Vec2
  radiusSquared: number
  label: string
}

export interface Diagnostics {
  validDomain: boolean
  message: string
  residuals: Record<string, number>
}

export type TriangleKind = 'spacelike' | 'timelike' | 'mixed' | 'null' | 'riemannian'

export type GeometryMode = 'sphere' | 'euclidean' | 'hyperbolic' | 'nhnegative' | 'galilei' | 'nhpositive' | 'desitter' | 'minkowski' | 'ads'
