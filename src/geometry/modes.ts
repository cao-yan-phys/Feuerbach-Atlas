import { adsProjectiveLift, hyperbolicLift, lorentzProjectiveLift, sphereLift } from './charts'
import { curvedInvariants } from '../math/curvedKernel'
import type { GeometryMode, TriangleKind, Vec2, Vec3 } from '../math/types'
import type { Bounds } from '../render/marchingSquares'

type CurvedMode = 'sphere' | 'hyperbolic' | 'desitter' | 'ads'

interface CurvedModeDefinition {
  metric: Vec3
  lift: (value: Vec2) => Vec3 | null
  bounds: Bounds
}

const curvedModes: Record<CurvedMode, CurvedModeDefinition> = {
  sphere: { metric: [1, 1, 1], lift: sphereLift, bounds: { minX: -2.2, maxX: 2.2, minY: -2.2, maxY: 2.2 } },
  hyperbolic: { metric: [1, -1, -1], lift: hyperbolicLift, bounds: { minX: -1.1, maxX: 1.1, minY: -1.1, maxY: 1.1 } },
  desitter: { metric: [1, 1, -1], lift: lorentzProjectiveLift, bounds: { minX: -2.2, maxX: 2.2, minY: -2.2, maxY: 2.2 } },
  ads: { metric: [1, 1, -1], lift: adsProjectiveLift, bounds: { minX: -2.2, maxX: 2.2, minY: -2.2, maxY: 2.2 } }
}

export const isCurvedMode = (mode: GeometryMode): mode is CurvedMode => mode in curvedModes

export const metricForMode = (mode: GeometryMode): Vec3 | null => isCurvedMode(mode) ? curvedModes[mode].metric : null

export const liftForMode = (mode: GeometryMode, value: Vec2): Vec3 | null => isCurvedMode(mode) ? curvedModes[mode].lift(value) : null

export const boundsForMode = (mode: GeometryMode, _vertices: [Vec2, Vec2, Vec2]): Bounds => {
  if (isCurvedMode(mode)) {
    return curvedModes[mode].bounds
  }

  return {
    minX: -2.2,
    maxX: 2.2,
    minY: -2.2,
    maxY: 2.2
  }
}

export interface CurvedDomain {
  valid: boolean
  kind: TriangleKind
}

export const curvedDomain = (mode: GeometryMode, vertices: [Vec3, Vec3, Vec3], tolerance = 1e-9): CurvedDomain => {
  const metric = metricForMode(mode)

  if (!metric) {
    return { valid: false, kind: 'mixed' }
  }

  const invariants = curvedInvariants(metric, vertices)

  if (mode === 'sphere') {
    return {
      valid: invariants.p > -1 + tolerance && invariants.q > -1 + tolerance && invariants.r > -1 + tolerance && invariants.d > tolerance,
      kind: 'riemannian'
    }
  }

  if (mode === 'hyperbolic') {
    return {
      valid: vertices.every((vertex) => vertex[0] > 0) && invariants.p > 1 + tolerance && invariants.q > 1 + tolerance && invariants.r > 1 + tolerance,
      kind: 'riemannian'
    }
  }

  const nullSide = Math.abs(invariants.p - 1) <= tolerance || Math.abs(invariants.q - 1) <= tolerance || Math.abs(invariants.r - 1) <= tolerance
  const values = [invariants.p - 1, invariants.q - 1, invariants.r - 1]
  const normalizedKind = nullSide
    ? 'null'
    : values.every((value) => value > 0)
      ? 'timelike'
      : values.every((value) => value < 0)
        ? 'spacelike'
        : 'mixed'

  return {
    valid: invariants.p > -1 + tolerance && invariants.q > -1 + tolerance && invariants.r > -1 + tolerance && invariants.delta < -tolerance && invariants.d > tolerance && !nullSide,
    kind: mode === 'ads' && normalizedKind === 'spacelike' ? 'timelike' : mode === 'ads' && normalizedKind === 'timelike' ? 'spacelike' : normalizedKind
  }
}
