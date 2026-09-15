import {
  adsProjectiveProject,
  antiDeSitterFromConformal,
  deSitterFromConformal,
  hyperbolicFromPolar,
  hyperbolicProject,
  lorentzProjectiveProject,
  sphereFromLatitudeLongitude,
  sphereProject
} from '../geometry/charts'
import type { GeometryMode, Vec2, Vec3 } from '../math/types'

export interface Preset {
  id: string
  label: string
  mode: GeometryMode
  vertices: [Vec2, Vec2, Vec2]
  branchIndex?: number
  visible?: boolean
}

const project = (point: Vec2 | null): Vec2 => {
  if (!point) {
    throw new Error()
  }
  return point
}

const projectedSphere = (latitude: number, longitude: number) => project(sphereProject(sphereFromLatitudeLongitude(latitude, longitude)))
const projectedHyperbolic = (rho: number, theta: number) => project(hyperbolicProject(hyperbolicFromPolar(rho, theta)))
const projectedAdS = (chi: number, tau: number) => project(adsProjectiveProject(antiDeSitterFromConformal(chi, tau)))
const projectedDeSitter = (theta: number, eta: number) => project(lorentzProjectiveProject(deSitterFromConformal(theta, eta)))

const dsFamily = (eta: number): [Vec2, Vec2, Vec2] => [
  projectedDeSitter(-0.8, 0),
  projectedDeSitter(0, eta),
  projectedDeSitter(0.8, 0)
]

export const presets: Preset[] = [
  {
    id: 'nhnegative-default',
    label: 'NH− (oscillating)',
    mode: 'nhnegative',
    vertices: [[-0.85, 0], [0.15, 0.7], [0.85, 0.2]]
  },
  {
    id: 'galilei-default',
    label: 'Galilean',
    mode: 'galilei',
    vertices: [[-0.85, 0], [0.15, 0.7], [0.85, 0.2]]
  },
  {
    id: 'nhpositive-default',
    label: 'NH+ (expanding)',
    mode: 'nhpositive',
    vertices: [[-0.85, 0], [0.15, 0.7], [0.85, 0.2]]
  },
  {
    id: 'euclidean-default',
    label: 'Default',
    mode: 'euclidean',
    vertices: [[-1.2, -0.6], [1.2, -0.4], [-0.2, 1.1]]
  },
  {
    id: 'sphere-default',
    label: 'Default',
    mode: 'sphere',
    vertices: [projectedSphere(0.2, -1), projectedSphere(-0.15, 0.65), projectedSphere(0.65, 1.75)]
  },
  {
    id: 'hyperbolic-default',
    label: 'Default',
    mode: 'hyperbolic',
    vertices: [projectedHyperbolic(0.8, -1.2), projectedHyperbolic(0.65, 0.4), projectedHyperbolic(0.9, 1.7)]
  },
  {
    id: 'minkowski-spacelike',
    label: 'Spacelike',
    mode: 'minkowski',
    vertices: [[-1.8, -0.15], [0.35, 0.25], [1.2, -0.35]]
  },
  {
    id: 'minkowski-timelike',
    label: 'Timelike',
    mode: 'minkowski',
    vertices: [[-0.15, -1.8], [0.25, 0.35], [-0.35, 1.2]]
  },
  {
    id: 'minkowski-mixed',
    label: 'Mixed',
    mode: 'minkowski',
    vertices: [[-1, 0], [1, 0], [0, 2]]
  },
  {
    id: 'minkowski-null',
    label: 'Null',
    mode: 'minkowski',
    vertices: [[-1, 0], [1, 0], [0, 1]]
  },
  {
    id: 'ads-publication',
    label: 'Publication',
    mode: 'ads',
    visible: false,
    vertices: [
      projectedAdS(-1.0023979282423225, -0.4041287035647518),
      projectedAdS(0.56968058514658, 0.3350822796467316),
      projectedAdS(1.0000553820445224, 0.4022643447879163)
    ]
  },
  {
    id: 'ads-finite',
    label: 'Finite contact',
    mode: 'ads',
    branchIndex: 0,
    vertices: [
      projectedAdS(0.8801720978945418, 0.10741976965041411),
      projectedAdS(-0.018678287775630187, -0.6345358384081672),
      projectedAdS(-0.8922330740543909, 0.04722483854760462)
    ]
  },
  {
    id: 'ads-ideal',
    label: 'Ideal contact',
    mode: 'ads',
    branchIndex: 2,
    vertices: [
      projectedAdS(0.37730434506393795, -0.004555824776901751),
      projectedAdS(-0.6564747814725098, 0.6140713213481841),
      projectedAdS(0.6076980130178555, 0.12033539836509555)
    ]
  },
  {
    id: 'ds-center-positive',
    label: 'Center on dS²',
    mode: 'desitter',
    vertices: dsFamily(-0.78)
  },
  {
    id: 'ds-center-null',
    label: 'Center on null boundary',
    mode: 'desitter',
    branchIndex: 2,
    vertices: dsFamily(-0.7471070029911566)
  },
  {
    id: 'ds-center-negative',
    label: 'Center off dS²',
    mode: 'desitter',
    branchIndex: 2,
    vertices: dsFamily(-0.65)
  }
]

export const presetsForMode = (mode: GeometryMode) => presets.filter((preset) => preset.mode === mode)

export const visiblePresetsForMode = (mode: GeometryMode) => presetsForMode(mode).filter((preset) => preset.visible !== false)

export const defaultPresetForMode = (mode: GeometryMode) => visiblePresetsForMode(mode)[0]!

export const presetVertices3 = (vertices: [Vec2, Vec2, Vec2], lift: (value: Vec2) => Vec3 | null): [Vec3, Vec3, Vec3] | null => {
  const lifted = vertices.map(lift)
  return lifted[0] && lifted[1] && lifted[2] ? [lifted[0], lifted[1], lifted[2]] : null
}
