import katex from 'katex'
import './styles.css'
import { defaultPresetForMode, presets, presetsForMode, type Preset, visiblePresetsForMode } from './data/presets'
import { adsProjectiveProject, hyperbolicProject, lorentzProjectiveProject, sphereProject } from './geometry/charts'
import { curvedDomain, isCurvedMode, liftForMode, metricForMode } from './geometry/modes'
import { boostTriangle, rotateTriangle } from './math/affine'
import { buildCurvedState } from './math/curvedKernel'
import { buildFlatState, euclideanCircumcircle, minkowskiCircumcycle } from './math/flatKernel'
import { buildIndicatrixConstruction, lorentzFinslerIndicatrixPoint, normedIndicatrixPoint, normalizeLorentzFinslerShape, normalizeNormedShape, type IndicatrixMode, type LorentzFinslerShape, type NormedShape } from './math/indicatrixKernel'
import { bilinear3, scale3 } from './math/linalg'
import { buildParabolicState, canonicalParabolicParameters, canonicalToRawParabolic, isParabolicMode, parabolicDefaultKappa, parabolicDisplayPoint, parabolicModeForKappa, parabolicPointFromDisplay, type ParabolicChart, type ParabolicView } from './math/parabolicKernel'
import type { GeometryMode, Vec2, Vec3 } from './math/types'
import { indicatrixViewportBounds, renderIndicatrixViewport, type IndicatrixOverlays, type IndicatrixSimilarity, type IndicatrixViewportTransform } from './render/indicatrixViewport'
import { renderViewport, type NinePointHomothety, type Overlays, type ViewportTransform } from './render/svgViewport'

const atlasModes: Array<[GeometryMode, string, string]> = [
  ['sphere', 'S²', 'S^2'],
  ['euclidean', 'ℝ²', '\\mathbb{R}^2'],
  ['hyperbolic', 'H²', 'H^2'],
  ['galilei', 'NH−, 𝔾², NH+', '\\mathrm{NH}_{-},\\mathbb{G}^2,\\mathrm{NH}_{+}'],
  ['ads', 'AdS²', '\\mathrm{AdS}^2'],
  ['minkowski', 'R¹,¹', '\\mathbb{R}^{1,1}'],
  ['desitter', 'dS²', '\\mathrm{dS}^2']
]

interface AtlasState {
  mode: GeometryMode
  preset: Preset
  vertices: [Vec2, Vec2, Vec2]
  branch: number
  parabolicKappa: number
  parabolicChart: ParabolicChart
  carrollDual: boolean
  homothetyProgress: number
  homothetyKind: NinePointHomothety
  overlays: Overlays
}

interface IndicatrixState {
  mode: IndicatrixMode
  vertices: [Vec2, Vec2, Vec2]
  normedShape: NormedShape
  lorentzShape: LorentzFinslerShape
  similarity: IndicatrixSimilarity
  overlays: IndicatrixOverlays
}

const indicatrixDefaultVertices = (mode: IndicatrixMode, normedShape: NormedShape, lorentzShape: LorentzFinslerShape): [Vec2, Vec2, Vec2] => {
  const parameters = mode === 'normed' ? [0.18, 2.28, 4.35] : [-0.8, 0.1, 0.9]
  const pointAt = mode === 'normed'
    ? (parameter: number) => normedIndicatrixPoint(normedShape, parameter)
    : (parameter: number) => lorentzFinslerIndicatrixPoint(lorentzShape, parameter)
  return parameters.map(pointAt) as [Vec2, Vec2, Vec2]
}

const cloneVertices = (vertices: [Vec2, Vec2, Vec2]): [Vec2, Vec2, Vec2] => vertices.map(([x, y]) => [x, y]) as [Vec2, Vec2, Vec2]

const atlasInitialPreset = defaultPresetForMode('euclidean')
const atlasState: AtlasState = {
  mode: 'euclidean',
  preset: atlasInitialPreset,
  vertices: cloneVertices(atlasInitialPreset.vertices),
  branch: atlasInitialPreset.branchIndex ?? 0,
  parabolicKappa: 0,
  parabolicChart: 'beltrami',
  carrollDual: false,
  homothetyProgress: 0,
  homothetyKind: 'euler',
  overlays: {
    bisectors: true,
    euler: true,
    altitudes: true,
    tangent: true,
    centers: false,
    circumcircle: false,
    euclideanCircumcircle: false,
    minkowskiCircumcircle: false,
    nullBoundary: true,
    horizon: false,
    homothety: false,
    grid: false,
    singularBranches: true
  }
}

const indicatrixInitialState = (): IndicatrixState => {
  const normedShape = normalizeNormedShape([0.1, -0.05, 0.02])
  const lorentzShape = normalizeLorentzFinslerShape([0.22, -0.16, 0.12])
  return {
    mode: 'normed',
    vertices: indicatrixDefaultVertices('normed', normedShape, lorentzShape),
    normedShape,
    lorentzShape,
    similarity: { translation: [0, 0], scale: 1 },
    overlays: {
      medians: true,
      circum: true,
      translated: false,
      feuerbach: true,
      midpoints: false,
      centers: true,
      grid: false
    }
  }
}

let activeView: 'atlas' | 'indicatrix' = 'atlas'
let indicatrixState = indicatrixInitialState()
let indicatrixCentroidTransformActive = false

const indicatrixOverlayKeys = ['medians', 'circum', 'translated', 'feuerbach', 'midpoints', 'centers', 'grid'] as const

const overlayKeys = ['bisectors', 'euler', 'altitudes', 'tangent', 'centers', 'circumcircle', 'euclideanCircumcircle', 'minkowskiCircumcircle', 'nullBoundary', 'horizon', 'homothety', 'grid', 'singularBranches'] as const

const atlasApp = document.querySelector<HTMLDivElement>('#app')
const wiredViewports = new WeakSet<SVGSVGElement>()
let atlasTransform: ViewportTransform | null = null
let indicatrixTransform: IndicatrixViewportTransform | null = null
let atlasGridBounds: ViewportTransform['bounds'] | null = null
let atlasDragBounds: ViewportTransform['bounds'] | null = null
let minkowskiBoostActive = false
let euclideanRotationActive = false
let centroidTransformActive = false
let circumcenterMotionActive = false
let exportAlignmentWired = false
let exportAlignmentObserver: ResizeObserver | null = null

if (!atlasApp) {
  throw new Error()
}

const selectedAtlasMode = () => atlasModes.find(([mode]) => mode === (isParabolicMode(atlasState.mode) ? 'galilei' : atlasState.mode))!

const numericalText = (value: number) => {
  if (!Number.isFinite(value)) {
    return '—'
  }
  return value.toFixed(4)
}

const coordinateText = (value: number) => {
  if (!Number.isFinite(value) || value === 0) {
    return '0'
  }
  return String(value)
}

const parabolicView = (): ParabolicView => atlasState.carrollDual ? 'carroll' : 'galilei'

const displayPoint = (point: Vec2): Vec2 | null => {
  if (!isParabolicMode(atlasState.mode)) {
    return point
  }
  return parabolicDisplayPoint(atlasState.parabolicKappa, point, atlasState.parabolicChart, parabolicView())
}

const rawPoint = (point: Vec2, referenceTime?: number): Vec2 | null => {
  if (!isParabolicMode(atlasState.mode)) {
    return point
  }
  return parabolicPointFromDisplay(atlasState.parabolicKappa, point, atlasState.parabolicChart, parabolicView(), referenceTime)
}

const displayCoordinates = () => atlasState.vertices.map((point) => displayPoint(point))

const usesBeltramiChart = () => isParabolicMode(atlasState.mode) && atlasState.parabolicChart === 'beltrami'

const parabolicBeltramiVertices = (map: (point: Vec2) => Vec2): [Vec2, Vec2, Vec2] | null => {
  const vertices = atlasState.vertices.map((vertex) => {
    const displayed = displayPoint(vertex)
    return displayed ? rawPoint(map(displayed), vertex[0]) : null
  })
  return vertices.every((vertex): vertex is Vec2 => vertex !== null) ? vertices as [Vec2, Vec2, Vec2] : null
}

const parabolicDomainMessage = (state: ReturnType<typeof buildParabolicState>) => {
  if (!atlasState.carrollDual || state.valid) {
    return state.message
  }
  const { u, v, w } = state.parameters
  if (!(u > 1e-10 && v > 1e-10)) {
    return 'ordered space requires X_A<X_B<X_C'
  }
  if (atlasState.parabolicKappa < -1e-10 && !(w < Math.PI / Math.sqrt(-atlasState.parabolicKappa) - 1e-10)) {
    return 'principal Carroll–dS² domain requires X_C-X_A<π/√−λ'
  }
  return state.message
}

const currentReadout = (): Array<[string, string]> => {
  if (atlasState.mode === 'euclidean') {
    return []
  }

  if (isParabolicMode(atlasState.mode)) {
    const state = buildParabolicState(atlasState.parabolicKappa, atlasState.vertices)
    const geometry = atlasState.carrollDual
      ? atlasState.mode === 'nhnegative' ? 'Carroll–dS²' : atlasState.mode === 'nhpositive' ? 'Carroll–AdS²' : 'Carroll'
      : atlasState.mode === 'nhnegative' ? 'oscillating Newton–Hooke' : atlasState.mode === 'nhpositive' ? 'expanding Newton–Hooke' : 'Galilean'
    const rows: Array<[string, string]> = [
      ['geometry', geometry],
      [atlasState.carrollDual ? 'λ' : 'κ', numericalText(atlasState.parabolicKappa)],
      ['Θ', numericalText(state.theta)],
      [atlasState.carrollDual ? 'X_F' : 'T_F', numericalText(state.contactTime ?? Number.NaN)]
    ]
    if (!state.valid) {
      rows.splice(2, 0, ['domain', parabolicDomainMessage(state)])
    }
    return rows
  }

  if (!isCurvedMode(atlasState.mode)) {
    const flat = buildFlatState(atlasState.mode === 'minkowski' ? -1 : 1, atlasState.vertices)
    return atlasState.mode === 'minkowski'
      ? [['type', flat.kind], ['cycle', flat.kind === 'mixed' ? '' : flat.tangentCycles.length > 0 ? 'real' : 'unavailable']]
      : [['domain', flat.valid ? 'valid' : 'invalid']]
  }

  const metric = metricForMode(atlasState.mode)!
  const lifted = atlasState.vertices.map((vertex) => liftForMode(atlasState.mode, vertex))
  if (!lifted[0] || !lifted[1] || !lifted[2]) {
    return [['domain', '—']]
  }
  const vertices = [lifted[0], lifted[1], lifted[2]] as [Vec3, Vec3, Vec3]
  const curved = buildCurvedState(metric, vertices)
  const domain = curvedDomain(atlasState.mode, vertices)
  const branch = curved.branches[atlasState.branch]
  const values: Array<[string, string]> = [
    ['p', numericalText(curved.invariants.p)],
    ['q', numericalText(curved.invariants.q)],
    ['r', numericalText(curved.invariants.r)],
    ['Δ', numericalText(curved.invariants.delta)],
    ['D', numericalText(curved.invariants.d)]
  ]
  if (atlasState.mode === 'sphere' || atlasState.mode === 'hyperbolic') {
    return [...values, ['domain', domain.valid ? 'valid' : 'invalid']]
  }
  return [...values, ['type', domain.kind], ['domain', domain.valid ? 'valid' : 'invalid'], ['cycle', domain.kind === 'mixed' ? '' : branch?.isReal ? 'real' : 'complex']]
}

const renderMath = (root: ParentNode = document) => {
  root.querySelectorAll<HTMLElement>('[data-katex]').forEach((target) => {
    const formula = target.dataset.katex
    if (formula) {
      katex.render(formula, target, { output: 'mathml', throwOnError: false })
      target.querySelectorAll('mi[mathvariant="double-struck"]').forEach((symbol) => {
        const glyph = symbol.textContent === 'R' ? 'ℝ' : symbol.textContent === 'Z' ? 'ℤ' : symbol.textContent === 'G' ? '𝔾' : null
        if (glyph) {
          symbol.textContent = glyph
          symbol.removeAttribute('mathvariant')
        }
      })
    }
  })
}

const mathematicalLabel = (label: string) => {
  const formula = label === 'Δ' ? '\\Delta' : label === 'κ' ? '\\kappa' : label === 'λ' ? '\\lambda' : label === 'Θ' ? '\\Theta' : label === 'T_F' ? 'T_F' : label === 'X_F' ? 'X_F' : ['x', 'y', 'p', 'q', 'r', 'D'].includes(label) ? label : null
  return formula ? `<span data-katex="${formula}">${label}</span>` : label
}

const readoutRows = () => currentReadout().map(([key, value]) => `<div><dt>${mathematicalLabel(key)}</dt><dd>${value}</dd></div>`).join('')

const readoutMarkup = () => {
  const rows = readoutRows()
  return rows ? `<dl class="readout">${rows}</dl>` : ''
}

const atlasModeButtons = () => atlasModes.map(([mode, label, formula]) => {
  const active = activeView === 'atlas' && (mode === 'galilei' ? isParabolicMode(atlasState.mode) : mode === atlasState.mode)
  return `
  <button class="mode-button${active ? ' is-active' : ''}" type="button" data-atlas-mode="${mode}" aria-pressed="${active}">
    <span data-katex="${formula}">${label}</span>
  </button>
`
}).join('')

const indicatrixModeButtons = () => `
  <div class="indicatrix-mode-grid" role="group" aria-label="Indicatrix geometry">
    <button class="mode-button${activeView === 'indicatrix' && indicatrixState.mode === 'normed' ? ' is-active' : ''}" type="button" data-indicatrix-mode="normed" aria-pressed="${activeView === 'indicatrix' && indicatrixState.mode === 'normed'}">Normed plane</button>
    <button class="mode-button${activeView === 'indicatrix' && indicatrixState.mode === 'lorentz-finsler' ? ' is-active' : ''}" type="button" data-indicatrix-mode="lorentz-finsler" aria-pressed="${activeView === 'indicatrix' && indicatrixState.mode === 'lorentz-finsler'}">Lorentz–Finsler plane</button>
  </div>
`

const indicatrixShapeLabels = () => indicatrixState.mode === 'normed'
  ? ['a', 'b', 'c']
  : ['q_1', 'q_2', 'q_3']

const indicatrixControl = (key: keyof IndicatrixOverlays, label: string) => `
  <label class="toggle-control toggle-indicatrix-${key}">
    <input type="checkbox" data-indicatrix-overlay="${key}"${indicatrixState.overlays[key] ? ' checked' : ''} />
    <span>${label}</span>
  </label>
`

const indicatrixControls = () => {
  const shape = indicatrixState.mode === 'normed' ? indicatrixState.normedShape : indicatrixState.lorentzShape
  const feuLabel = indicatrixState.mode === 'normed' ? 'Feuerbach circle' : 'Feuerbach indicatrix'
  return `
    <button class="panel-button" type="button" data-indicatrix-reset>Reset</button>
    <section class="indicatrix-range-section" aria-label="Shape">
      <span>Shape</span>
      ${shape.coefficients.map((value, index) => `
        <label class="indicatrix-range">
          <span data-katex="${indicatrixShapeLabels()[index]!}">${indicatrixShapeLabels()[index]!}</span>
          <output>${value.toFixed(2)}</output>
          <input type="range" min="-0.35" max="0.35" step="0.01" value="${value}" data-indicatrix-shape="${index}" aria-label="Shape coefficient ${index + 1}" />
        </label>
      `).join('')}
    </section>
    <div class="toggle-grid">
      ${indicatrixControl('medians', 'Medians')}
      ${indicatrixControl('circum', 'Circum-indicatrix')}
      ${indicatrixControl('translated', '<span data-katex="C">C</span>-orthocenter')}
      ${indicatrixControl('feuerbach', feuLabel)}
      ${indicatrixControl('midpoints', 'Midpoints')}
      ${indicatrixControl('centers', 'Centers')}
      ${indicatrixControl('grid', 'Grid')}
    </div>
  `
}

const indicatrixDataMarkup = () => {
  return `
    <div class="advanced-content">
      <div class="coordinate-controls">
        <div class="coordinate-header">
          <span></span>
          <span data-katex="x">x</span>
          <span data-katex="y">y</span>
        </div>
        ${indicatrixState.vertices.map((point, index) => `
          <div class="coordinate-row">
            <span class="data-vertex-label data-vertex-${index}">${String.fromCharCode(65 + index)}</span>
            <input type="text" inputmode="decimal" value="${coordinateText(point[0])}" data-indicatrix-coordinate="${index}:0" aria-label="${String.fromCharCode(65 + index)} x" />
            <input type="text" inputmode="decimal" value="${coordinateText(point[1])}" data-indicatrix-coordinate="${index}:1" aria-label="${String.fromCharCode(65 + index)} y" />
          </div>
        `).join('')}
      </div>
    </div>
  `
}

const indicatrixDetailsMarkup = () => {
  const normed = indicatrixState.mode === 'normed'
  const unit = normed
    ? 'h(\\phi)=1+a\\cos(2\\phi)+b\\sin(2\\phi)+c\\cos(4\\phi),\\quad (x,y)=h(\\phi)(\\cos\\phi,\\sin\\phi)+\\frac{d h}{d\\phi}(-\\sin\\phi,\\cos\\phi),\\quad 0\\leq\\phi\\lt2\\pi,\\quad h+\\frac{d^2h}{d\\phi^2}>0'
    : '(\\mu_1,\\mu_2,\\mu_3)=(-1.10,0.15,1.25),\\quad q(\\theta)=\\sum_{j=1}^{3}q_j e^{-((\\theta-\\mu_j)/1.9)^2},\\quad f(\\theta)=e^{q(\\theta)},\\quad (x,y)=\\frac{(\\sinh\\theta,\\cosh\\theta)}{f(\\theta)},\\quad -\\infty\\lt\\theta\\lt\\infty,\\quad f>0,\\quad \\frac{d^2f}{d\\theta^2}-f\\lt0'
  return `
    <div class="indicatrix-details">
      <section class="indicatrix-detail">
        <span class="indicatrix-detail-label">${normed ? 'Unit circle' : 'Forward unit indicatrix'}</span>
        <span class="indicatrix-detail-formula" data-katex="${unit}\\text{.}">${unit}.</span>
      </section>
    </div>
  `
}

const indicatrixNoteMarkup = () => `
  <div class="note-content">The Feuerbach geometry of normed planes was studied by <a href="https://faculty.washington.edu/moishe/branko/BG25%20Geometry%20of%20Minkowski%20planes.pdf" target="_blank" rel="noreferrer">Asplund and Grünbaum (1960)</a>, <a href="https://www.e-periodica.ch/digbib/view?pid=ens-001%3A2007%3A53%3A%3A273" target="_blank" rel="noreferrer">Martini and Spirova (2007)</a>, and <a href="https://arxiv.org/abs/1602.06144" target="_blank" rel="noreferrer">Leopold and Martini (2016)</a>.</div>
`

const atlasPresetOptions = () => (isParabolicMode(atlasState.mode)
  ? ['nhnegative', 'galilei', 'nhpositive'].flatMap((mode) => visiblePresetsForMode(mode as GeometryMode))
  : visiblePresetsForMode(atlasState.mode)
).map((preset) => `<option value="${preset.id}"${preset.id === atlasState.preset.id ? ' selected' : ''}>${preset.label}</option>`).join('')

const branchControls = () => ([
  [0, 'I'],
  [1, 'I_a'],
  [2, 'I_b'],
  [3, 'I_c']
] as Array<[number, string]>).map(([branch, formula]) => `
  <button class="branch-button branch-${branch}${atlasState.branch === branch ? ' is-active' : ''}" type="button" data-branch="${branch}" aria-pressed="${atlasState.branch === branch}">
    <span data-katex="${formula}">${formula}</span>
  </button>
`).join('')

const interactionDescriptions: Partial<Record<keyof Overlays, string>> = {
  centers: 'Right-click the pink point to enable translation and scaling. Drag to translate; scroll to scale.',
  euclideanCircumcircle: 'Right-click the black star, then scroll to rotate. Right-click the origin ring, then scroll to boost.',
  minkowskiCircumcircle: 'Right-click the black star, then scroll to boost. Right-click the origin ring, then scroll to rotate.'
}

const interactionDescription = (key: keyof Overlays) => key === 'centers' && isParabolicMode(atlasState.mode) ? undefined : interactionDescriptions[key]

const ninePointHomothetyAvailable = () => {
  if (atlasState.mode !== 'euclidean' && atlasState.mode !== 'minkowski') {
    return false
  }
  const state = buildFlatState(atlasState.mode === 'minkowski' ? -1 : 1, atlasState.vertices)
  return Boolean(state.valid && state.circumcircle && state.ninePoint && state.orthocenter)
}

const overlayControl = (key: keyof Overlays, label: string) => {
  const description = interactionDescription(key)
  return `
  <label class="toggle-control toggle-${key}${key === 'tangent' ? ` tangent-branch-${atlasState.branch}` : ''}${description ? ' has-tooltip' : ''}">
    <input type="checkbox" data-overlay="${key}"${atlasState.overlays[key] ? ' checked' : ''} />
    <span>${label}</span>
    ${description ? `<span class="toggle-tooltip" role="tooltip">${description}</span>` : ''}
  </label>
  `
}

const overlayControls = () => {
  const parabolic = isParabolicMode(atlasState.mode)
  const flat = atlasState.mode === 'euclidean' || atlasState.mode === 'minkowski'
  const labels: Array<[keyof Overlays, string]> = parabolic
    ? [['bisectors', 'Area bisectors'], ['euler', 'Euler cycle'], ['tangent', 'Tangent cycle'], ['centers', 'Centers']]
    : flat
    ? [['bisectors', 'Medians'], ['euler', 'Nine-point circle'], ['altitudes', 'Altitudes'], ['tangent', 'Tangent cycles'], ['circumcircle', 'Circumcircle'], ['centers', 'Centers']]
    : [['bisectors', 'Area bisectors'], ['euler', 'Euler cycle'], ['altitudes', 'Pseudoaltitudes'], ['tangent', 'Tangent cycles'], ['centers', 'Centers']]
  const nullControls: Array<[keyof Overlays, string]> = atlasState.mode === 'minkowski' || atlasState.mode === 'desitter' || atlasState.mode === 'ads' ? [['nullBoundary', 'null']] : []
  const horizonControls: Array<[keyof Overlays, string]> = atlasState.mode === 'desitter' ? [['horizon', 'Horizon (for <span data-katex="x=0">x=0</span>)']] : []
  const euclideanCircumcircleControl: Array<[keyof Overlays, string]> = atlasState.mode === 'minkowski' ? [['euclideanCircumcircle', 'Euclidean circumcircle']] : []
  const minkowskiCircumcircleControl: Array<[keyof Overlays, string]> = atlasState.mode === 'euclidean' ? [['minkowskiCircumcircle', 'Minkowski circumcircle']] : []
  const homothetyAvailable = ninePointHomothetyAvailable()
  const homothetyControl = homothetyAvailable ? `
    <div class="homothety-control">
      ${overlayControl('homothety', 'Homothety')}
      ${atlasState.overlays.homothety ? `
        <div class="homothety-motion">
          <div class="homothety-options" role="group" aria-label="Nine-point homothety">
            <button class="homothety-kind-button${atlasState.homothetyKind === 'euler' ? ' is-active' : ''}" type="button" data-homothety-kind="euler" aria-label="Euler homothety" aria-pressed="${atlasState.homothetyKind === 'euler'}"><span data-katex="\\mathrm{I}">I</span></button>
            <button class="homothety-kind-button${atlasState.homothetyKind === 'medial' ? ' is-active' : ''}" type="button" data-homothety-kind="medial" aria-label="Medial homothety" aria-pressed="${atlasState.homothetyKind === 'medial'}"><span data-katex="\\mathrm{II}">II</span></button>
          </div>
          <input class="homothety-progress" type="range" min="0" max="1" step="0.01" value="${atlasState.homothetyProgress}" data-homothety-progress aria-label="Homothety progress" />
        </div>
      ` : ''}
    </div>
  ` : ''
  const parabolicControls: Array<[keyof Overlays, string]> = parabolic ? [['singularBranches', 'Pseudoaltitudes']] : []
  const controls = [...labels, ...euclideanCircumcircleControl, ...minkowskiCircumcircleControl].map(([key, label]) => overlayControl(key, label)).join('')
  const trailingControls = [...nullControls, ...horizonControls, ...parabolicControls, ['grid', 'Grid'] as [keyof Overlays, string]].map(([key, label]) => overlayControl(key, label)).join('')
  return `${controls}${homothetyControl}${trailingControls}`
}

const coordinateControls = () => {
  const parabolic = isParabolicMode(atlasState.mode)
  const coordinates = displayCoordinates()
  const first = parabolic
    ? atlasState.carrollDual ? atlasState.parabolicChart === 'beltrami' ? 'x' : 'X' : atlasState.parabolicChart === 'beltrami' ? 'z' : 'T'
    : 'x'
  const second = parabolic
    ? atlasState.carrollDual ? atlasState.parabolicChart === 'beltrami' ? '\\tau' : 't' : atlasState.parabolicChart === 'beltrami' ? 't' : 'y'
    : 'y'
  const secondText = second === '\\tau' ? 'τ' : second
  return `
  <div class="coordinate-header">
    <span></span>
    <span data-katex="${first}">${first}</span>
    <span data-katex="${second}">${secondText}</span>
  </div>
  ${coordinates.map((point, index) => `
  <div class="coordinate-row">
    <span class="data-vertex-label data-vertex-${index}">${String.fromCharCode(65 + index)}</span>
    <input type="text" inputmode="decimal" value="${point ? coordinateText(point[0]) : ''}" data-coordinate="${index}:0" aria-label="${String.fromCharCode(65 + index)} ${first}" />
    <input type="text" inputmode="decimal" value="${point ? coordinateText(point[1]) : ''}" data-coordinate="${index}:1" aria-label="${String.fromCharCode(65 + index)} ${secondText}" />
  </div>
`).join('')}
`
}

const chartDetails = () => {
  const details: Record<GeometryMode, { firstLabel?: string, metric: string, map?: string }> = {
    euclidean: {
      metric: 'ds^2=dx^2+dy^2,\\qquad (x,y)\\in\\mathbb{R}^2'
    },
    sphere: {
      metric: 'ds^2=d\\varphi^2+\\cos^2\\!\\varphi\\,d\\lambda^2,\\qquad -\\frac{\\pi}{2}\\leq\\varphi\\leq\\frac{\\pi}{2},\\quad \\lambda\\in\\mathbb{R}/2\\pi\\mathbb{Z},\\quad (\\varphi,\\lambda)\\ne(0,\\pi)',
      map: 'x=\\frac{\\cos\\varphi\\sin\\lambda}{1+\\cos\\varphi\\cos\\lambda},\\qquad y=\\frac{\\sin\\varphi}{1+\\cos\\varphi\\cos\\lambda},\\qquad (x,y)\\in\\mathbb{R}^2'
    },
    hyperbolic: {
      metric: 'ds^2=d\\rho^2+\\sinh^2\\!\\rho\\,d\\theta^2,\\qquad \\rho\\geq0,\\quad \\theta\\in\\mathbb{R}/2\\pi\\mathbb{Z}',
      map: 'x=\\tanh\\!\\left(\\frac{\\rho}{2}\\right)\\cos\\theta,\\qquad y=\\tanh\\!\\left(\\frac{\\rho}{2}\\right)\\sin\\theta,\\qquad x^2+y^2<1'
    },
    nhnegative: {
      firstLabel: 'Basis',
      metric: atlasState.carrollDual
        ? 'C_\\lambda^{\\prime\\prime}=\\lambda C_\\lambda,\\quad S_\\lambda^{\\prime\\prime}=\\lambda S_\\lambda,\\qquad \\lambda=-\\Lambda_{\\rm C}<0'
        : 'C_\\kappa^{\\prime\\prime}=\\kappa C_\\kappa,\\quad S_\\kappa^{\\prime\\prime}=\\kappa S_\\kappa,\\qquad \\kappa<0',
      map: atlasState.carrollDual
        ? atlasState.parabolicChart === 'beltrami'
          ? 'x=\\frac{S_\\lambda(X)}{C_\\lambda(X)},\\qquad \\tau=\\frac{t}{C_\\lambda(X)},\\qquad C_\\lambda(X)\\ne0'
          : '(X,t),\\qquad 0<X_C-X_A<\\frac{\\pi}{\\sqrt{-\\lambda}}'
        : atlasState.parabolicChart === 'beltrami'
          ? 't=\\frac{S_\\kappa(T)}{C_\\kappa(T)},\\qquad z=\\frac{y}{C_\\kappa(T)},\\qquad C_\\kappa(T)\\ne0'
          : '(T,y),\\qquad 0<T_C-T_A<\\frac{\\pi}{\\sqrt{-\\kappa}}'
    },
    galilei: {
      firstLabel: 'Basis',
      metric: atlasState.carrollDual
        ? 'C_0(X)=1,\\qquad S_0(X)=X,\\qquad P_0(X)=\\frac{X^2}{2}'
        : 'C_0(T)=1,\\qquad S_0(T)=T,\\qquad P_0(T)=\\frac{T^2}{2}',
      map: atlasState.carrollDual
        ? atlasState.parabolicChart === 'beltrami'
          ? '(x,\\tau)=(X,t),\\qquad (x,\\tau)\\in\\mathbb{R}^2'
          : '(X,t),\\qquad X_A<X_B<X_C'
        : atlasState.parabolicChart === 'beltrami'
          ? '(t,z)=(T,y),\\qquad (t,z)\\in\\mathbb{R}^2'
          : '(T,y),\\qquad T_A<T_B<T_C'
    },
    nhpositive: {
      firstLabel: 'Basis',
      metric: atlasState.carrollDual
        ? 'C_\\lambda^{\\prime\\prime}=\\lambda C_\\lambda,\\quad S_\\lambda^{\\prime\\prime}=\\lambda S_\\lambda,\\qquad \\lambda=-\\Lambda_{\\rm C}>0'
        : 'C_\\kappa^{\\prime\\prime}=\\kappa C_\\kappa,\\quad S_\\kappa^{\\prime\\prime}=\\kappa S_\\kappa,\\qquad \\kappa>0',
      map: atlasState.carrollDual
        ? atlasState.parabolicChart === 'beltrami'
          ? 'x=\\frac{S_\\lambda(X)}{C_\\lambda(X)},\\qquad \\tau=\\frac{t}{C_\\lambda(X)},\\qquad |x|<\\frac{1}{\\sqrt{\\lambda}}'
          : '(X,t),\\qquad X_A<X_B<X_C'
        : atlasState.parabolicChart === 'beltrami'
          ? 't=\\frac{S_\\kappa(T)}{C_\\kappa(T)},\\qquad z=\\frac{y}{C_\\kappa(T)},\\qquad |t|<\\frac{1}{\\sqrt{\\kappa}}'
          : '(T,y),\\qquad T_A<T_B<T_C'
    },
    minkowski: {
      metric: 'ds^2=dX^2-dT^2,\\qquad (X,T)\\in\\mathbb{R}^2',
      map: 'x=X,\\qquad y=T,\\qquad (x,y)\\in\\mathbb{R}^2'
    },
    desitter: {
      metric: 'ds^2=\\sec^2\\!\\eta\\left(d\\theta^2-d\\eta^2\\right),\\qquad -\\frac{\\pi}{2}<\\theta,\\eta<\\frac{\\pi}{2}',
      map: 'x=\\tan\\theta,\\qquad y=\\frac{\\sin\\eta}{\\cos\\theta},\\qquad 1+x^2-y^2>0'
    },
    ads: {
      metric: 'ds^2=\\sec^2\\!\\chi\\left(-d\\tau^2+d\\chi^2\\right),\\qquad -\\frac{\\pi}{2}<\\tau,\\chi<\\frac{\\pi}{2}',
      map: 'x=\\frac{\\sin\\chi}{\\cos\\tau},\\qquad y=\\tan\\tau,\\qquad 1+y^2-x^2>0'
    }
  }
  const detail = details[atlasState.mode]
  return `
    <div class="details-content">
      <div class="detail-row"><span>${detail.firstLabel ?? 'Metric'}</span><span class="detail-formula"><span data-katex="${detail.metric}\\text{.}">${detail.metric}.</span></span></div>
      ${detail.map ? `<div class="detail-row"><span>Chart</span><span class="detail-formula"><span data-katex="${detail.map}\\text{.}">${detail.map}.</span></span></div>` : ''}
    </div>
  `
}

const noteMarkup = () => {
  const euclideanFormula = '\\mathbb{R}^2'
  const minkowskiFormula = '\\mathbb{R}^{1,1}'
  return `
    <div class="note-content">The <span data-katex="${euclideanFormula}">ℝ²</span> construction illustrates <a href="https://en.wikipedia.org/wiki/Feuerbach%27s_theorem" target="_blank" rel="noreferrer">Feuerbach’s theorem</a>, first stated in 1822. For a short proof, see <a href="https://arxiv.org/abs/1610.03962" target="_blank" rel="noreferrer">Hofbauer (2016)</a>. The <span data-katex="S^2">S²</span> and <span data-katex="H^2">H²</span> constructions are due to <a href="https://arxiv.org/abs/1105.2153" target="_blank" rel="noreferrer">Akopyan’s extension (2009)</a>. The <span data-katex="${minkowskiFormula}">${minkowskiFormula}</span> version was pointed out by <a href="https://www.ams.org/journals/notices/196406/196406FullIssue.pdf" target="_blank" rel="noreferrer">Anderson (1964)</a>, <a href="https://books.google.com/books?id=FyToBwAAQBAJ" target="_blank" rel="noreferrer">Yaglom (1979)</a>, and <a href="https://books.google.com/books?id=-kFVAAAAYAAJ" target="_blank" rel="noreferrer">Schröder (1998)</a>. The Galilean analogue was treated by <a href="https://books.google.com/books?id=FyToBwAAQBAJ" target="_blank" rel="noreferrer">Yaglom (1979)</a>, <a href="https://www.delpher.nl/nl/tijdschriften/view?identifier=KBKWG02:022200001:00005&amp;coll=dts&amp;cql%5B%5D=%28alternative+exact+%22Nieuw+Archief+voor+Wiskunde%22%29&amp;query=Coxeter&amp;sortfield=date&amp;utm_source=chatgpt.com&amp;page=2&amp;rowid=2" target="_blank" rel="noreferrer">Coxeter (1983)</a>, and <a href="https://www.heldermann-verlag.de/jgg/jgg10/j10h2beba.pdf" target="_blank" rel="noreferrer">Beban-Brkić et al. (2006)</a>.</div>
  `
}

const parabolicPresetControl = () => `
  <section class="branch-control parabolic-preset-control" aria-label="Preset">
    <span>Preset</span>
    <div class="parabolic-preset-options" role="group" aria-label="Preset">
      ${([
        ['nhnegative-default', atlasState.carrollDual ? '\\mathrm{Carroll\\text{-}dS}^{2}' : '\\mathrm{NH}_{-}'],
        ['galilei-default', atlasState.carrollDual ? '\\mathrm{Carroll}' : '\\mathbb{G}^2'],
        ['nhpositive-default', atlasState.carrollDual ? '\\mathrm{Carroll\\text{-}AdS}^{2}' : '\\mathrm{NH}_{+}']
      ] as Array<[string, string]>).map(([id, formula]) => `
        <button class="branch-button${atlasState.preset.id === id ? ' is-active' : ''}" type="button" data-parabolic-preset="${id}" aria-pressed="${atlasState.preset.id === id}"><span data-katex="${formula}">${formula}</span></button>
      `).join('')}
    </div>
  </section>
`

const presetControl = () => isParabolicMode(atlasState.mode) ? parabolicPresetControl() : (atlasState.mode === 'minkowski' || atlasState.mode === 'desitter' || atlasState.mode === 'ads') ? `
  <label class="select-control">Preset
    <select data-preset aria-label="Preset">${atlasPresetOptions()}</select>
  </label>
` : ''

const parabolicControl = () => isParabolicMode(atlasState.mode) ? `
  <label class="curvature-control">
    <span class="curvature-label">Curvature</span>
    <output data-parabolic-kappa-value><span data-katex="${atlasState.carrollDual ? '\\lambda' : '\\kappa'}=${atlasState.parabolicKappa.toFixed(2)}">${atlasState.carrollDual ? 'λ' : 'κ'}=${atlasState.parabolicKappa.toFixed(2)}</span></output>
    <input type="range" min="-0.9" max="0.9" step="0.01" value="${atlasState.parabolicKappa}" data-parabolic-kappa aria-label="Signed ${atlasState.carrollDual ? 'lambda' : 'kappa'}" />
  </label>
` : ''

const parabolicChartControl = () => isParabolicMode(atlasState.mode) ? `
  <section class="branch-control parabolic-chart-control" aria-label="Charts">
    <span>Charts</span>
    <div class="parabolic-chart-options" role="group" aria-label="Charts">
      <button class="branch-button${atlasState.parabolicChart === 'natural' ? ' is-active' : ''}" type="button" data-parabolic-chart="natural" aria-pressed="${atlasState.parabolicChart === 'natural'}">Natural</button>
      <button class="branch-button${atlasState.parabolicChart === 'beltrami' ? ' is-active' : ''}" type="button" data-parabolic-chart="beltrami" aria-pressed="${atlasState.parabolicChart === 'beltrami'}">Beltrami</button>
    </div>
  </section>
` : ''

const parabolicDualControl = () => isParabolicMode(atlasState.mode) ? `
  <button class="dual-button${atlasState.carrollDual ? ' is-active' : ''}" type="button" data-carroll-dual aria-pressed="${atlasState.carrollDual}">
    <span>Carroll dual</span>
  </button>
` : ''

const atlasShell = () => {
  const [, atlasModeLabel] = selectedAtlasMode()
  const modeLabel = activeView === 'indicatrix' ? indicatrixState.mode === 'normed' ? 'Normed plane' : 'Lorentz–Finsler plane' : atlasModeLabel
  const indicatrixNotice = activeView === 'indicatrix' ? '<div class="indicatrix-notice"></div>' : ''
  const workspacePanels = activeView === 'indicatrix' ? `
      <details class="advanced-panel indicatrix-data-panel">
        <summary>Data</summary>
        ${indicatrixDataMarkup()}
      </details>
      <details class="advanced-panel details-panel">
        <summary>Details</summary>
        ${indicatrixDetailsMarkup()}
      </details>
      ${indicatrixState.mode === 'normed' ? `<details class="advanced-panel note-panel">
        <summary>Note</summary>
        ${indicatrixNoteMarkup()}
      </details>` : ''}
  ` : `
      <details class="advanced-panel">
        <summary>Data</summary>
        <div class="advanced-content">
          <div class="coordinate-controls">${coordinateControls()}</div>
          ${readoutMarkup()}
        </div>
      </details>
      <details class="advanced-panel details-panel">
        <summary>Details</summary>
        ${chartDetails()}
      </details>
      <details class="advanced-panel note-panel">
        <summary>Note</summary>
        ${noteMarkup()}
      </details>
  `
  const controlContent = activeView === 'indicatrix' ? indicatrixControls() : `
        <button class="panel-button" type="button" data-reset>Reset</button>
        ${isParabolicMode(atlasState.mode) ? '' : `<section class="branch-control" aria-label="Tangent cycle">
          <span>Tangent cycle</span>
          <div class="branch-options" role="group" aria-label="Tangent branch">${branchControls()}</div>
        </section>`}
        ${presetControl()}
        ${parabolicControl()}
        ${parabolicChartControl()}
        ${parabolicDualControl()}
        <div class="toggle-grid">
          ${overlayControls()}
        </div>
  `
  atlasApp.innerHTML = `
    <main class="atlas-shell">
      <aside class="atlas-sidebar">
        <div class="mode-grid" role="group" aria-label="Geometry">
          ${atlasModeButtons()}
        </div>
        <header class="atlas-title">
          <h1><span data-katex="\\mathrm{Feuerbach}">Feuerbach</span><span class="atlas-word">Atlas</span></h1>
        </header>
        ${indicatrixModeButtons()}
        <div class="export-controls">
          <button class="panel-button" type="button" data-copy-link>Copy link</button>
          <button class="panel-button" type="button" data-export-svg>Export</button>
        </div>
      </aside>
      <div class="atlas-workspace">
      <section class="viewport-panel" aria-label="${modeLabel} viewport">
        <svg class="geometry-viewport" viewBox="0 0 1000 700" aria-label="${modeLabel} viewport"></svg>
      </section>
      ${indicatrixNotice}
      ${workspacePanels}
      </div>
      <section class="control-panel" aria-label="Controls">
        ${controlContent}
      </section>
    </main>
    <footer class="atlas-copyright">© 2026 Yan Cao</footer>
  `
  renderMath()
  wireControls()
  renderScene()
  wireExportAlignment()
}

const clampVertex = (value: Vec2, mode = atlasState.mode): Vec2 | null => {
  if (mode === 'hyperbolic') {
    const radius = Math.hypot(value[0], value[1])
    return radius > 0.97 ? [value[0] * 0.97 / radius, value[1] * 0.97 / radius] : value
  }
  if (mode === 'desitter') {
    return 1 + value[0] * value[0] - value[1] * value[1] > 0.01 ? value : null
  }
  if (mode === 'ads') {
    return 1 + value[1] * value[1] - value[0] * value[0] > 0.01 ? value : null
  }
  if (mode === 'sphere') {
    return [Math.max(-4, Math.min(4, value[0])), Math.max(-4, Math.min(4, value[1]))]
  }
  return value
}

const sameVertex = (first: Vec2, second: Vec2) => Math.abs(first[0] - second[0]) <= 1e-10 && Math.abs(first[1] - second[1]) <= 1e-10

const visibleAreaCenter = (): Vec2 | null => {
  if (!atlasState.overlays.centers) {
    return null
  }
  if (isParabolicMode(atlasState.mode)) {
    if (!usesBeltramiChart()) {
      return null
    }
    const state = buildParabolicState(atlasState.parabolicKappa, atlasState.vertices)
    if (!state.valid || !state.pseudomedianCenter) {
      return null
    }
    return displayPoint(canonicalToRawParabolic([state.pseudomedianCenter.T, state.pseudomedianCenter.y], state.parameters, atlasState.parabolicKappa))
  }
  if (!isCurvedMode(atlasState.mode)) {
    const state = buildFlatState(atlasState.mode === 'minkowski' ? -1 : 1, atlasState.vertices)
    return state.valid ? state.centroid : null
  }
  const metric = metricForMode(atlasState.mode)!
  const lifted = atlasState.vertices.map((vertex) => liftForMode(atlasState.mode, vertex))
  if (!lifted[0] || !lifted[1] || !lifted[2]) {
    return null
  }
  const vertices = [lifted[0], lifted[1], lifted[2]] as [Vec3, Vec3, Vec3]
  if (!curvedDomain(atlasState.mode, vertices).valid) {
    return null
  }
  const direction = buildCurvedState(metric, vertices).pseudomedianDirection
  if (!direction) {
    return null
  }
  const norm = bilinear3(metric, direction, direction)
  if (norm <= 1e-10) {
    return null
  }
  const point = scale3(direction, 1 / Math.sqrt(norm))
  if (atlasState.mode === 'sphere') {
    return sphereProject(point)
  }
  if (atlasState.mode === 'hyperbolic') {
    return hyperbolicProject(point)
  }
  return atlasState.mode === 'ads' ? adsProjectiveProject(point) : lorentzProjectiveProject(point)
}

const visibleCircumcenter = (): Vec2 | null => {
  if (atlasState.mode === 'minkowski' && atlasState.overlays.euclideanCircumcircle) {
    return euclideanCircumcircle(atlasState.vertices)?.center ?? null
  }
  if (atlasState.mode === 'euclidean' && atlasState.overlays.minkowskiCircumcircle) {
    return minkowskiCircumcycle(atlasState.vertices)?.center ?? null
  }
  return null
}

const acceptedTriangle = (vertices: [Vec2, Vec2, Vec2] | null) => vertices?.every((vertex) => {
  const accepted = clampVertex(vertex)
  return accepted && sameVertex(accepted, vertex)
}) ? vertices : null

const setParabolicKappa = (kappa: number) => {
  const next = Math.max(-0.9, Math.min(0.9, kappa))
  if (!isParabolicMode(atlasState.mode)) {
    atlasState.parabolicKappa = next
    return
  }
  const parameters = canonicalParabolicParameters(atlasState.vertices, atlasState.parabolicKappa)
  atlasState.vertices = [
    parameters.anchor,
    canonicalToRawParabolic([parameters.u, parameters.p], parameters, next),
    canonicalToRawParabolic([parameters.w, parameters.q], parameters, next)
  ]
  atlasState.parabolicKappa = next
  const mode = parabolicModeForKappa(next)
  atlasState.mode = mode
  atlasState.preset = defaultPresetForMode(mode)
}

const coordinateIndices = (value: string): [number, 0 | 1] => [Number(value.charAt(0)), Number(value.charAt(2)) as 0 | 1]

const currentLink = () => {
  const url = new URL(window.location.href)
  if (activeView === 'indicatrix') {
    const state = new URLSearchParams({
      i: indicatrixState.mode,
      v: indicatrixState.vertices.flat().join(','),
      n: indicatrixState.normedShape.coefficients.join(','),
      l: indicatrixState.lorentzShape.coefficients.join(','),
      a: indicatrixState.similarity.translation.join(','),
      s: String(indicatrixState.similarity.scale),
      o: indicatrixOverlayKeys.map((key) => indicatrixState.overlays[key] ? '1' : '0').join('')
    })
    url.hash = state.toString()
    return url.href
  }
  const state = new URLSearchParams({
    m: atlasState.mode,
    p: atlasState.preset.id,
    v: atlasState.vertices.flat().join(','),
    b: String(atlasState.branch),
    k: String(atlasState.parabolicKappa),
    c: isParabolicMode(atlasState.mode) ? atlasState.parabolicChart : '',
    d: isParabolicMode(atlasState.mode) && atlasState.carrollDual ? '1' : '',
    h: String(atlasState.homothetyProgress),
    u: atlasState.homothetyKind,
    o: overlayKeys.map((key) => atlasState.overlays[key] ? '1' : '0').join('')
  })
  url.hash = state.toString()
  return url.href
}

const copyCurrentLink = async (button: HTMLButtonElement) => {
  const link = currentLink()
  const copied = () => {
    button.textContent = 'Copied'
    window.setTimeout(() => {
      button.textContent = 'Copy link'
    }, 1200)
  }

  try {
    await navigator.clipboard.writeText(link)
    copied()
  } catch {
    const input = document.createElement('textarea')
    input.value = link
    input.setAttribute('readonly', '')
    input.style.position = 'fixed'
    input.style.opacity = '0'
    document.body.append(input)
    input.select()
    const success = document.execCommand('copy')
    input.remove()
    if (success) {
      copied()
    }
  }
}

interface SvgWritable {
  write: (data: Blob) => Promise<void>
  close: () => Promise<void>
}

interface SvgFileHandle {
  createWritable: () => Promise<SvgWritable>
}

interface SvgPickerWindow extends Window {
  showSaveFilePicker?: (options: {
    suggestedName: string
    types: Array<{ description: string; accept: Record<string, string[]> }>
  }) => Promise<SvgFileHandle>
}

const exportFilename = () => {
  const date = new Date()
  const number = (value: number) => String(value).padStart(2, '0')
  return `feuerbach-atlas-${date.getFullYear()}-${number(date.getMonth() + 1)}-${number(date.getDate())}-${number(date.getHours())}${number(date.getMinutes())}${number(date.getSeconds())}.svg`
}

const exportCurrentSvg = async () => {
  const source = atlasApp.querySelector<SVGSVGElement>('.geometry-viewport')
  if (!source) {
    return
  }

  const copy = source.cloneNode(true) as SVGSVGElement
  copy.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
  const background = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
  background.setAttribute('width', '100%')
  background.setAttribute('height', '100%')
  background.setAttribute('fill', '#fff')
  copy.insertBefore(background, copy.firstChild)
  const sourceNodes = source.querySelectorAll<SVGElement>('*')
  const copyNodes = copy.querySelectorAll<SVGElement>('*')
  const properties = ['fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin', 'font-family', 'font-size', 'font-style', 'font-weight', 'text-anchor']

  sourceNodes.forEach((sourceNode, index) => {
    const copyNode = copyNodes[index + 1]
    if (!copyNode) {
      return
    }
    const style = getComputedStyle(sourceNode)
    properties.forEach((property) => {
      const value = style.getPropertyValue(property)
      if (value) {
        copyNode.setAttribute(property, value)
      }
    })
  })

  const content = new XMLSerializer().serializeToString(copy)
  const file = new Blob([content], { type: 'image/svg+xml' })
  const filename = exportFilename()
  const pickerWindow = window as SvgPickerWindow
  if (pickerWindow.showSaveFilePicker) {
    try {
      const handle = await pickerWindow.showSaveFilePicker({
        suggestedName: filename,
        types: [{ description: 'SVG image', accept: { 'image/svg+xml': ['.svg'] } }]
      })
      const writable = await handle.createWritable()
      await writable.write(file)
      await writable.close()
      return
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
    }
  }

  const url = URL.createObjectURL(file)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 0)
}

const syncReadout = () => {
  const readout = atlasApp.querySelector<HTMLElement>('.readout')
  if (readout) {
    readout.innerHTML = readoutRows()
    renderMath(readout)
  }
  atlasApp.querySelectorAll<HTMLInputElement>('[data-coordinate]').forEach((input) => {
    const [vertexIndex, coordinateIndex] = coordinateIndices(input.dataset.coordinate!)
    const point = displayPoint(atlasState.vertices[vertexIndex]!)
    input.value = point ? coordinateText(point[coordinateIndex]) : ''
  })
}

const syncDetails = () => {
  const details = atlasApp.querySelector<HTMLDetailsElement>('.details-panel')
  if (!details) {
    return
  }
  const wasOpen = details.open
  details.innerHTML = `<summary>Details</summary>${chartDetails()}`
  details.open = wasOpen
  renderMath(details)
}

const syncIndicatrixData = () => {
  if (activeView !== 'indicatrix') {
    return
  }
  atlasApp.querySelectorAll<HTMLInputElement>('[data-indicatrix-coordinate]').forEach((input) => {
    const [vertexIndex, coordinateIndex] = coordinateIndices(input.dataset.indicatrixCoordinate!)
    input.value = coordinateText(indicatrixState.vertices[vertexIndex]![coordinateIndex])
  })
}

const syncIndicatrixNotice = () => {
  if (activeView !== 'indicatrix') {
    return
  }
  const notice = atlasApp.querySelector<HTMLElement>('.indicatrix-notice')
  if (!notice) {
    return
  }
  const construction = buildIndicatrixConstruction(indicatrixState.mode, indicatrixState.vertices, indicatrixState.normedShape, indicatrixState.lorentzShape)
  notice.textContent = construction.valid ? '' : 'Choose three separated points.'
}

const renderScene = () => {
  const svg = atlasApp.querySelector<SVGSVGElement>('.geometry-viewport')
  if (!svg) {
    return
  }
  if (activeView === 'indicatrix') {
    const rendered = renderIndicatrixViewport(svg, indicatrixState.mode, indicatrixState.vertices, indicatrixState.normedShape, indicatrixState.lorentzShape, indicatrixState.overlays, indicatrixState.similarity, indicatrixCentroidTransformActive, indicatrixViewportBounds)
    indicatrixTransform = rendered.transform
    wireIndicatrixDrag(svg)
    syncIndicatrixData()
    syncIndicatrixNotice()
    return
  }
  const motionActive = atlasState.mode === 'minkowski' ? minkowskiBoostActive : atlasState.mode === 'euclidean' ? euclideanRotationActive : false
  const transform = renderViewport(svg, atlasState.mode, atlasState.vertices, atlasState.overlays, atlasState.branch, atlasDragBounds ?? atlasGridBounds ?? undefined, motionActive, centroidTransformActive, circumcenterMotionActive, atlasState.parabolicKappa, atlasState.parabolicChart, parabolicView(), atlasState.homothetyProgress, atlasState.homothetyKind)
  atlasTransform = transform
  if (atlasState.overlays.grid && !atlasGridBounds) {
    atlasGridBounds = transform.bounds
  }
  wireDrag(svg)
  syncReadout()
}

const loadPreset = (preset: Preset) => {
  const wasParabolic = isParabolicMode(atlasState.mode)
  if (atlasState.mode !== preset.mode) {
    atlasGridBounds = null
  }
  atlasState.mode = preset.mode
  atlasState.preset = preset
  atlasState.vertices = cloneVertices(preset.vertices)
  atlasState.branch = preset.branchIndex ?? 0
  atlasState.homothetyProgress = 0
  atlasState.homothetyKind = 'euler'
  atlasState.parabolicKappa = isParabolicMode(preset.mode) ? parabolicDefaultKappa(preset.mode) : 0
  if (isParabolicMode(preset.mode)) {
    if (!wasParabolic) {
      atlasState.parabolicChart = 'beltrami'
      atlasState.carrollDual = false
    }
    atlasState.overlays.bisectors = true
    atlasState.overlays.euler = true
    atlasState.overlays.altitudes = true
    atlasState.overlays.tangent = true
    atlasState.overlays.centers = false
    atlasState.overlays.singularBranches = true
  }
  minkowskiBoostActive = false
  euclideanRotationActive = false
  centroidTransformActive = false
  circumcenterMotionActive = false
}

const wireControls = () => {
  atlasApp.querySelectorAll<HTMLButtonElement>('[data-atlas-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      const mode = button.dataset.atlasMode as GeometryMode
      activeView = 'atlas'
      if (isParabolicMode(mode) && isParabolicMode(atlasState.mode)) {
        setParabolicKappa(parabolicDefaultKappa(mode))
      } else {
        loadPreset(defaultPresetForMode(mode))
      }
      atlasShell()
    })
  })
  atlasApp.querySelectorAll<HTMLButtonElement>('[data-indicatrix-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      const mode = button.dataset.indicatrixMode
      if (mode !== 'normed' && mode !== 'lorentz-finsler') {
        return
      }
      activeView = 'indicatrix'
      if (indicatrixState.mode !== mode) {
        indicatrixState.vertices = indicatrixDefaultVertices(mode, indicatrixState.normedShape, indicatrixState.lorentzShape)
        indicatrixState.similarity = { translation: [0, 0], scale: 1 }
      }
      indicatrixState.mode = mode
      indicatrixCentroidTransformActive = false
      atlasGridBounds = null
      atlasDragBounds = null
      atlasShell()
    })
  })
  atlasApp.querySelector<HTMLSelectElement>('[data-preset]')?.addEventListener('change', (event) => {
    const candidates = isParabolicMode(atlasState.mode)
      ? ['nhnegative', 'galilei', 'nhpositive'].flatMap((mode) => presetsForMode(mode as GeometryMode))
      : presetsForMode(atlasState.mode)
    const preset = candidates.find((candidate) => candidate.id === (event.currentTarget as HTMLSelectElement).value)
    if (preset) {
      loadPreset(preset)
      atlasShell()
    }
  })
  atlasApp.querySelectorAll<HTMLButtonElement>('[data-branch]').forEach((button) => {
    button.addEventListener('click', () => {
      atlasState.branch = Number(button.dataset.branch)
      atlasApp.querySelectorAll<HTMLButtonElement>('[data-branch]').forEach((candidate) => {
        const active = candidate.dataset.branch === button.dataset.branch
        candidate.classList.toggle('is-active', active)
        candidate.setAttribute('aria-pressed', String(active))
      })
      const tangentControl = atlasApp.querySelector<HTMLElement>('.toggle-tangent')
      if (tangentControl) {
        tangentControl.classList.remove('tangent-branch-0', 'tangent-branch-1', 'tangent-branch-2', 'tangent-branch-3')
        tangentControl.classList.add(`tangent-branch-${atlasState.branch}`)
      }
      renderScene()
    })
  })
  atlasApp.querySelector<HTMLButtonElement>('[data-reset]')?.addEventListener('click', () => {
    loadPreset(atlasState.preset)
    atlasShell()
  })
  atlasApp.querySelector<HTMLButtonElement>('[data-indicatrix-reset]')?.addEventListener('click', () => {
    const mode = indicatrixState.mode
    indicatrixState = indicatrixInitialState()
    indicatrixState.mode = mode
    indicatrixState.vertices = indicatrixDefaultVertices(mode, indicatrixState.normedShape, indicatrixState.lorentzShape)
    indicatrixCentroidTransformActive = false
    atlasShell()
  })
  atlasApp.querySelectorAll<HTMLButtonElement>('[data-parabolic-preset]').forEach((button) => {
    button.addEventListener('click', () => {
      const preset = presets.find((candidate) => candidate.id === button.dataset.parabolicPreset)
      if (preset) {
        loadPreset(preset)
        atlasShell()
      }
    })
  })
  atlasApp.querySelectorAll<HTMLButtonElement>('[data-parabolic-chart]').forEach((button) => {
    button.addEventListener('click', () => {
      const chart = button.dataset.parabolicChart
      if (chart !== 'natural' && chart !== 'beltrami') {
        return
      }
      atlasState.parabolicChart = chart
      atlasGridBounds = null
      centroidTransformActive = false
      atlasShell()
    })
  })
  atlasApp.querySelector<HTMLButtonElement>('[data-carroll-dual]')?.addEventListener('click', () => {
    atlasState.carrollDual = !atlasState.carrollDual
    atlasGridBounds = null
    centroidTransformActive = false
    atlasShell()
  })
  const copyLinkButton = atlasApp.querySelector<HTMLButtonElement>('[data-copy-link]')
  copyLinkButton?.addEventListener('click', () => {
    void copyCurrentLink(copyLinkButton)
  })
  atlasApp.querySelector<HTMLButtonElement>('[data-export-svg]')?.addEventListener('click', exportCurrentSvg)
  atlasApp.querySelectorAll<HTMLInputElement>('[data-overlay]').forEach((input) => {
    input.addEventListener('change', () => {
      const key = input.dataset.overlay as keyof Overlays
      atlasState.overlays[key] = input.checked
      if (key === 'euclideanCircumcircle' && !input.checked) {
        minkowskiBoostActive = false
        circumcenterMotionActive = false
      }
      if (key === 'minkowskiCircumcircle' && !input.checked) {
        euclideanRotationActive = false
        circumcenterMotionActive = false
      }
      if (key === 'centers' && !input.checked) {
        centroidTransformActive = false
      }
      if (key === 'homothety') {
        if (input.checked) {
          atlasState.homothetyProgress = 0
        }
        atlasShell()
        return
      }
      if (key === 'grid') {
        atlasGridBounds = null
      }
      renderScene()
    })
  })
  atlasApp.querySelector<HTMLInputElement>('[data-homothety-progress]')?.addEventListener('input', (event) => {
    const value = Number((event.currentTarget as HTMLInputElement).value)
    if (Number.isFinite(value)) {
      atlasState.homothetyProgress = Math.max(0, Math.min(1, value))
      renderScene()
    }
  })
  atlasApp.querySelectorAll<HTMLButtonElement>('[data-homothety-kind]').forEach((button) => {
    button.addEventListener('click', () => {
      const kind = button.dataset.homothetyKind
      if (kind !== 'euler' && kind !== 'medial') {
        return
      }
      atlasState.homothetyKind = kind
      atlasState.homothetyProgress = 0
      atlasShell()
    })
  })
  atlasApp.querySelectorAll<HTMLInputElement>('[data-indicatrix-coordinate]').forEach((input) => {
    input.addEventListener('change', () => {
      const [index, coordinate] = coordinateIndices(input.dataset.indicatrixCoordinate!)
      const value = Number(input.value)
      if (!Number.isInteger(index) || index < 0 || index > 2 || !Number.isFinite(value)) {
        syncIndicatrixData()
        return
      }
      indicatrixState.vertices[index]![coordinate] = value
      renderScene()
    })
  })
  atlasApp.querySelectorAll<HTMLInputElement>('[data-indicatrix-shape]').forEach((input) => {
    input.addEventListener('input', () => {
      const index = Number(input.dataset.indicatrixShape)
      const value = Number(input.value)
      if (!Number.isInteger(index) || index < 0 || index > 2 || !Number.isFinite(value)) {
        return
      }
      const shape = indicatrixState.mode === 'normed' ? indicatrixState.normedShape : indicatrixState.lorentzShape
      const values = [...shape.coefficients]
      values[index] = value
      if (indicatrixState.mode === 'normed') {
        indicatrixState.normedShape = normalizeNormedShape(values)
      } else {
        indicatrixState.lorentzShape = normalizeLorentzFinslerShape(values)
      }
      const normalized = indicatrixState.mode === 'normed' ? indicatrixState.normedShape : indicatrixState.lorentzShape
      atlasApp.querySelectorAll<HTMLInputElement>('[data-indicatrix-shape]').forEach((candidate, candidateIndex) => {
        candidate.value = String(normalized.coefficients[candidateIndex]!)
        const output = candidate.closest('.indicatrix-range')?.querySelector<HTMLOutputElement>('output')
        if (output) {
          output.textContent = normalized.coefficients[candidateIndex]!.toFixed(2)
        }
      })
      renderScene()
    })
  })
  atlasApp.querySelectorAll<HTMLInputElement>('[data-indicatrix-overlay]').forEach((input) => {
    input.addEventListener('change', () => {
      const key = input.dataset.indicatrixOverlay as keyof IndicatrixOverlays
      indicatrixState.overlays[key] = input.checked
      if (key === 'centers' && !input.checked) {
        indicatrixCentroidTransformActive = false
      }
      renderScene()
    })
  })
  atlasApp.querySelector<HTMLInputElement>('[data-parabolic-kappa]')?.addEventListener('input', (event) => {
    const value = Number((event.currentTarget as HTMLInputElement).value)
    if (!Number.isFinite(value)) {
      return
    }
    setParabolicKappa(value)
    atlasApp.querySelectorAll<HTMLButtonElement>('[data-atlas-mode]').forEach((button) => {
      const active = button.dataset.atlasMode === 'galilei' ? isParabolicMode(atlasState.mode) : button.dataset.atlasMode === atlasState.mode
      button.classList.toggle('is-active', active)
      button.setAttribute('aria-pressed', String(active))
    })
    atlasApp.querySelectorAll<HTMLButtonElement>('[data-parabolic-preset]').forEach((button) => {
      const active = button.dataset.parabolicPreset === atlasState.preset.id
      button.classList.toggle('is-active', active)
      button.setAttribute('aria-pressed', String(active))
    })
    const preset = atlasApp.querySelector<HTMLSelectElement>('[data-preset]')
    if (preset) {
      preset.value = atlasState.preset.id
    }
    const output = atlasApp.querySelector<HTMLOutputElement>('[data-parabolic-kappa-value]')
    if (output) {
      output.innerHTML = `<span data-katex="${atlasState.carrollDual ? '\\lambda' : '\\kappa'}=${atlasState.parabolicKappa.toFixed(2)}">${atlasState.carrollDual ? 'λ' : 'κ'}=${atlasState.parabolicKappa.toFixed(2)}</span>`
      renderMath(output)
    }
    syncDetails()
    renderScene()
  })
  atlasApp.querySelectorAll<HTMLInputElement>('[data-coordinate]').forEach((input) => {
    input.addEventListener('change', () => {
      const [vertexIndex, coordinateIndex] = coordinateIndices(input.dataset.coordinate!)
      const value = Number(input.value)
      if (!Number.isFinite(value)) {
        syncReadout()
        return
      }
      const displayed = displayPoint(atlasState.vertices[vertexIndex]!)
      if (!displayed) {
        syncReadout()
        return
      }
      displayed[coordinateIndex] = value
      const candidate = rawPoint(displayed, atlasState.vertices[vertexIndex]![0])
      const accepted = candidate ? clampVertex(candidate) : null
      if (accepted) {
        atlasState.vertices[vertexIndex] = accepted
        renderScene()
      } else {
        syncReadout()
      }
    })
  })
}

const pointerPosition = (svg: SVGSVGElement, event: MouseEvent): Vec2 => {
  const rect = svg.getBoundingClientRect()
  return [(event.clientX - rect.left) * 1000 / rect.width, (event.clientY - rect.top) * 700 / rect.height]
}

const displayIndicatrixPoint = ([x, y]: Vec2): Vec2 => [
  indicatrixState.similarity.translation[0] + indicatrixState.similarity.scale * x,
  indicatrixState.similarity.translation[1] + indicatrixState.similarity.scale * y
]

const canonicalIndicatrixPoint = ([x, y]: Vec2): Vec2 => [
  (x - indicatrixState.similarity.translation[0]) / indicatrixState.similarity.scale,
  (y - indicatrixState.similarity.translation[1]) / indicatrixState.similarity.scale
]

const wireIndicatrixDrag = (svg: SVGSVGElement) => {
  if (wiredViewports.has(svg)) {
    return
  }
  wiredViewports.add(svg)
  let dragIndex: number | null = null
  let translationAnchor: Vec2 | null = null
  svg.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || !indicatrixTransform) {
      return
    }
    const pointer = pointerPosition(svg, event)
    if (indicatrixCentroidTransformActive) {
      translationAnchor = indicatrixTransform.toWorld(pointer)
      svg.setPointerCapture(event.pointerId)
      event.preventDefault()
      return
    }
    const construction = buildIndicatrixConstruction(indicatrixState.mode, indicatrixState.vertices, indicatrixState.normedShape, indicatrixState.lorentzShape)
    const closest = construction.vertices.map((vertex, index) => {
      const screen = indicatrixTransform!.toScreen(displayIndicatrixPoint(vertex))
      return { index, distance: Math.hypot(screen[0] - pointer[0], screen[1] - pointer[1]) }
    }).sort((first, second) => first.distance - second.distance)[0]
    if (!closest || closest.distance > 24) {
      return
    }
    dragIndex = closest.index
    svg.setPointerCapture(event.pointerId)
    event.preventDefault()
  })
  svg.addEventListener('pointermove', (event) => {
    if (!indicatrixTransform) {
      return
    }
    if (translationAnchor) {
      const current = indicatrixTransform.toWorld(pointerPosition(svg, event))
      indicatrixState.similarity.translation = [
        indicatrixState.similarity.translation[0] + current[0] - translationAnchor[0],
        indicatrixState.similarity.translation[1] + current[1] - translationAnchor[1]
      ]
      translationAnchor = current
      renderScene()
      return
    }
    if (dragIndex === null) {
      return
    }
    const target = canonicalIndicatrixPoint(indicatrixTransform.toWorld(pointerPosition(svg, event)))
    if (!target.every(Number.isFinite)) {
      return
    }
    indicatrixState.vertices[dragIndex] = [
      Math.max(-2.2, Math.min(2.2, target[0])),
      Math.max(-1.54, Math.min(1.54, target[1]))
    ]
    renderScene()
  })
  const stop = (event: PointerEvent) => {
    if (svg.hasPointerCapture(event.pointerId)) {
      svg.releasePointerCapture(event.pointerId)
    }
    dragIndex = null
    translationAnchor = null
  }
  svg.addEventListener('pointerup', stop)
  svg.addEventListener('pointercancel', stop)
  svg.addEventListener('contextmenu', (event) => {
    if (!indicatrixState.overlays.centers || !indicatrixTransform) {
      return
    }
    const construction = buildIndicatrixConstruction(indicatrixState.mode, indicatrixState.vertices, indicatrixState.normedShape, indicatrixState.lorentzShape)
    const centroid = indicatrixTransform.toScreen(displayIndicatrixPoint(construction.centroid))
    const pointer = pointerPosition(svg, event)
    if (Math.hypot(centroid[0] - pointer[0], centroid[1] - pointer[1]) > 16) {
      return
    }
    indicatrixCentroidTransformActive = !indicatrixCentroidTransformActive
    event.preventDefault()
    renderScene()
  })
  svg.addEventListener('wheel', (event) => {
    if (!indicatrixCentroidTransformActive || !Number.isFinite(event.deltaY) || Math.abs(event.deltaY) <= 1e-12) {
      return
    }
    const construction = buildIndicatrixConstruction(indicatrixState.mode, indicatrixState.vertices, indicatrixState.normedShape, indicatrixState.lorentzShape)
    const factor = Math.exp(-Math.sign(event.deltaY) * 0.1)
    const previousScale = indicatrixState.similarity.scale
    const nextScale = Math.max(0.05, Math.min(20, previousScale * factor))
    const centroid = displayIndicatrixPoint(construction.centroid)
    indicatrixState.similarity.scale = nextScale
    indicatrixState.similarity.translation = [
      centroid[0] - nextScale * construction.centroid[0],
      centroid[1] - nextScale * construction.centroid[1]
    ]
    event.preventDefault()
    renderScene()
  })
}

const wireDrag = (svg: SVGSVGElement) => {
  if (wiredViewports.has(svg)) {
    return
  }
  wiredViewports.add(svg)
  let dragIndex: number | null = null
  let translationAnchor: Vec2 | null = null
  svg.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) {
      return
    }
    if (!atlasTransform) {
      return
    }
    const pointer = pointerPosition(svg, event)
    if (centroidTransformActive) {
      if (!visibleAreaCenter()) {
        centroidTransformActive = false
        renderScene()
        return
      }
      translationAnchor = atlasTransform.toWorld(pointer)
      atlasDragBounds = { ...atlasTransform.bounds }
      svg.setPointerCapture(event.pointerId)
      event.preventDefault()
      return
    }
    const closest = atlasState.vertices.map((vertex, index) => {
      const displayed = displayPoint(vertex)
      if (!displayed) {
        return null
      }
      const screen = atlasTransform!.toScreen(displayed)
      return { index, distance: Math.hypot(screen[0] - pointer[0], screen[1] - pointer[1]) }
    }).filter((candidate): candidate is { index: number, distance: number } => candidate !== null).sort((left, right) => left.distance - right.distance)[0]
    if (!closest || closest.distance > 24) {
      return
    }
    dragIndex = closest.index
    atlasDragBounds = { ...atlasTransform.bounds }
    svg.setPointerCapture(event.pointerId)
    event.preventDefault()
  })
  svg.addEventListener('pointermove', (event) => {
    if (!atlasTransform) {
      return
    }
    if (translationAnchor) {
      const current = atlasTransform.toWorld(pointerPosition(svg, event))
      const vertices = usesBeltramiChart()
        ? parabolicBeltramiVertices(([x, y]) => [x + current[0] - translationAnchor![0], y + current[1] - translationAnchor![1]])
        : atlasState.vertices.map(([x, y]) => [x + current[0] - translationAnchor![0], y + current[1] - translationAnchor![1]] as Vec2) as [Vec2, Vec2, Vec2]
      const accepted = acceptedTriangle(vertices)
      if (accepted) {
        atlasState.vertices = accepted
        translationAnchor = current
        renderScene()
      }
      return
    }
    if (dragIndex === null) {
      return
    }
    const displayed = atlasTransform.toWorld(pointerPosition(svg, event))
    const raw = rawPoint(displayed, atlasState.vertices[dragIndex]![0])
    const candidate = raw ? clampVertex(raw) : null
    if (candidate) {
      atlasState.vertices[dragIndex] = candidate
      renderScene()
    }
  })
  const stopDragging = (event: PointerEvent) => {
    if (svg.hasPointerCapture(event.pointerId)) {
      svg.releasePointerCapture(event.pointerId)
    }
    dragIndex = null
    translationAnchor = null
    atlasDragBounds = null
  }
  svg.addEventListener('pointerup', stopDragging)
  svg.addEventListener('pointercancel', stopDragging)
  svg.addEventListener('contextmenu', (event) => {
    const pointer = pointerPosition(svg, event)
    const areaCenter = visibleAreaCenter()
    if (areaCenter && atlasTransform) {
      const center = atlasTransform.toScreen(areaCenter)
      if (Math.hypot(center[0] - pointer[0], center[1] - pointer[1]) <= 16) {
        centroidTransformActive = !centroidTransformActive
        event.preventDefault()
        renderScene()
        return
      }
    }
    const circumcenter = visibleCircumcenter()
    if (circumcenter && atlasTransform) {
      const center = atlasTransform.toScreen(circumcenter)
      if (Math.hypot(center[0] - pointer[0], center[1] - pointer[1]) <= 12) {
        circumcenterMotionActive = !circumcenterMotionActive
        event.preventDefault()
        renderScene()
        return
      }
    }
    const boostToggle = atlasState.mode === 'minkowski' && atlasState.overlays.euclideanCircumcircle
    const rotationToggle = atlasState.mode === 'euclidean' && atlasState.overlays.minkowskiCircumcircle
    if ((!boostToggle && !rotationToggle) || !atlasTransform) {
      return
    }
    const origin = atlasTransform.toScreen([0, 0])
    if (Math.hypot(origin[0] - pointer[0], origin[1] - pointer[1]) > 14) {
      return
    }
    if (boostToggle) {
      minkowskiBoostActive = !minkowskiBoostActive
    }
    if (rotationToggle) {
      euclideanRotationActive = !euclideanRotationActive
    }
    event.preventDefault()
    renderScene()
  })
  svg.addEventListener('wheel', (event) => {
    if (centroidTransformActive) {
      if (!Number.isFinite(event.deltaY) || Math.abs(event.deltaY) <= 1e-12) {
        return
      }
      event.preventDefault()
      const center = visibleAreaCenter()
      if (!center) {
        centroidTransformActive = false
        renderScene()
        return
      }
      const factor = Math.exp(-Math.sign(event.deltaY) * 0.1)
      const vertices = usesBeltramiChart()
        ? parabolicBeltramiVertices(([x, y]) => [center[0] + (x - center[0]) * factor, center[1] + (y - center[1]) * factor])
        : atlasState.vertices.map(([x, y]) => [center[0] + (x - center[0]) * factor, center[1] + (y - center[1]) * factor] as Vec2) as [Vec2, Vec2, Vec2]
      const accepted = acceptedTriangle(vertices)
      if (accepted) {
        atlasState.vertices = accepted
        renderScene()
      }
      return
    }
    if (circumcenterMotionActive) {
      if (!Number.isFinite(event.deltaY) || Math.abs(event.deltaY) <= 1e-12) {
        return
      }
      const center = visibleCircumcenter()
      if (!center) {
        circumcenterMotionActive = false
        renderScene()
        return
      }
      const amount = -Math.sign(event.deltaY) * 0.1
      const vertices = atlasState.mode === 'minkowski'
        ? rotateTriangle(atlasState.vertices, center, amount)
        : boostTriangle(atlasState.vertices, center, amount)
      if (!vertices.every((vertex) => vertex.every(Number.isFinite))) {
        return
      }
      event.preventDefault()
      atlasState.vertices = vertices
      renderScene()
      return
    }
    const boostActive = atlasState.mode === 'minkowski' && minkowskiBoostActive
    const rotationActive = atlasState.mode === 'euclidean' && euclideanRotationActive
    if (!boostActive && !rotationActive) {
      return
    }
    if (!Number.isFinite(event.deltaY) || Math.abs(event.deltaY) <= 1e-12) {
      return
    }
    const amount = -Math.sign(event.deltaY) * 0.1
    const vertices = boostActive
      ? boostTriangle(atlasState.vertices, [0, 0], amount)
      : rotateTriangle(atlasState.vertices, [0, 0], amount)
    if (!vertices.every((vertex) => vertex.every(Number.isFinite))) {
      return
    }
    event.preventDefault()
    atlasState.vertices = vertices
    renderScene()
  }, { passive: false })
}

const alignExportControls = () => {
  const sidebar = atlasApp.querySelector<HTMLElement>('.atlas-sidebar')
  const modeGrid = atlasApp.querySelector<HTMLElement>('.mode-grid')
  const viewport = atlasApp.querySelector<HTMLElement>('.viewport-panel')
  const exportControls = atlasApp.querySelector<HTMLElement>('.export-controls')
  if (!sidebar || !modeGrid || !viewport || !exportControls) {
    return
  }
  if (window.matchMedia('(max-width: 760px)').matches) {
    exportControls.style.removeProperty('top')
    exportControls.style.removeProperty('left')
    exportControls.style.removeProperty('right')
    exportControls.style.removeProperty('width')
    return
  }
  const sidebarRect = sidebar.getBoundingClientRect()
  const modeGridRect = modeGrid.getBoundingClientRect()
  const viewportRect = viewport.getBoundingClientRect()
  const exportHeight = exportControls.getBoundingClientRect().height
  exportControls.style.top = `${Math.round(viewportRect.bottom - sidebarRect.top - exportHeight)}px`
  exportControls.style.left = `${Math.round(modeGridRect.left - sidebarRect.left)}px`
  exportControls.style.right = 'auto'
  exportControls.style.width = `${Math.round(modeGridRect.width)}px`
}

const wireExportAlignment = () => {
  if (!exportAlignmentWired) {
    exportAlignmentWired = true
    window.addEventListener('resize', alignExportControls)
  }
  exportAlignmentObserver?.disconnect()
  exportAlignmentObserver = new ResizeObserver(alignExportControls)
  atlasApp.querySelectorAll<HTMLElement>('.mode-grid, .viewport-panel').forEach((element) => exportAlignmentObserver!.observe(element))
  void document.fonts.ready.then(alignExportControls)
  window.requestAnimationFrame(alignExportControls)
}

const hydrateLinkedState = () => {
  const state = new URLSearchParams(window.location.hash.slice(1))
  const indicatrixMode = state.get('i')
  if (indicatrixMode === 'normed' || indicatrixMode === 'lorentz-finsler') {
    const vertices = state.get('v')?.split(',').map(Number)
    const positions = state.get('p')?.split(',').map(Number)
    const normed = state.get('n')?.split(',').map(Number)
    const lorentz = state.get('l')?.split(',').map(Number)
    const translation = state.get('a')?.split(',').map(Number)
    const scale = Number(state.get('s'))
    const overlays = state.get('o')
    activeView = 'indicatrix'
    indicatrixState.mode = indicatrixMode
    if (normed?.length === 3 && normed.every(Number.isFinite)) {
      indicatrixState.normedShape = normalizeNormedShape(normed)
    }
    if (lorentz?.length === 3 && lorentz.every(Number.isFinite)) {
      indicatrixState.lorentzShape = normalizeLorentzFinslerShape(lorentz)
    }
    if (vertices?.length === 6 && vertices.every(Number.isFinite)) {
      indicatrixState.vertices = [[vertices[0]!, vertices[1]!], [vertices[2]!, vertices[3]!], [vertices[4]!, vertices[5]!]]
    } else if (positions?.length === 3 && positions.every(Number.isFinite)) {
      const pointAt = indicatrixMode === 'normed'
        ? (parameter: number) => normedIndicatrixPoint(indicatrixState.normedShape, parameter)
        : (parameter: number) => lorentzFinslerIndicatrixPoint(indicatrixState.lorentzShape, parameter)
      indicatrixState.vertices = positions.map(pointAt) as [Vec2, Vec2, Vec2]
    } else {
      indicatrixState.vertices = indicatrixDefaultVertices(indicatrixMode, indicatrixState.normedShape, indicatrixState.lorentzShape)
    }
    if (translation?.length === 2 && translation.every(Number.isFinite) && Number.isFinite(scale) && scale >= 0.05 && scale <= 20) {
      indicatrixState.similarity = { translation: [translation[0]!, translation[1]!], scale }
    }
    if (overlays && overlays.length === indicatrixOverlayKeys.length && /^[01]+$/.test(overlays)) {
      indicatrixOverlayKeys.forEach((key, index) => {
        indicatrixState.overlays[key] = overlays[index] === '1'
      })
    }
    return
  }
  const mode = state.get('m')
  const values = state.get('v')?.split(',').map(Number)
  if ((!atlasModes.some(([candidate]) => candidate === mode) && !isParabolicMode(mode ?? '')) || values?.length !== 6 || !values.every(Number.isFinite)) {
    return
  }

  const linkedMode = mode as GeometryMode
  const vertices: [Vec2, Vec2, Vec2] = [[values[0]!, values[1]!], [values[2]!, values[3]!], [values[4]!, values[5]!]]
  if (!vertices.every((vertex) => {
    const accepted = clampVertex(vertex, linkedMode)
    return accepted && sameVertex(accepted, vertex)
  })) {
    return
  }

  const linkedPreset = presetsForMode(linkedMode).find((preset) => preset.id === state.get('p')) ?? defaultPresetForMode(linkedMode)
  const linkedBranch = Number(state.get('b'))
  const linkedKappa = Number(state.get('k'))
  const linkedHomothetyProgress = Number(state.get('h'))
  const linkedHomothetyKind = state.get('u')
  const linkedChart = state.get('c')
  const linkedDual = state.get('d')
  const linkedOverlays = state.get('o')
  atlasState.mode = linkedMode
  atlasState.preset = linkedPreset
  atlasState.vertices = vertices
  atlasState.branch = Number.isInteger(linkedBranch) && linkedBranch >= 0 && linkedBranch <= 3 ? linkedBranch : linkedPreset.branchIndex ?? 0
  atlasState.parabolicKappa = isParabolicMode(linkedMode) && Number.isFinite(linkedKappa) ? Math.max(-0.9, Math.min(0.9, linkedKappa)) : isParabolicMode(linkedMode) ? parabolicDefaultKappa(linkedMode) : 0
  atlasState.parabolicChart = isParabolicMode(linkedMode) && linkedChart === 'natural' ? 'natural' : 'beltrami'
  atlasState.carrollDual = isParabolicMode(linkedMode) && linkedDual === '1'
  atlasState.homothetyProgress = Number.isFinite(linkedHomothetyProgress) ? Math.max(0, Math.min(1, linkedHomothetyProgress)) : 0
  atlasState.homothetyKind = linkedHomothetyKind === 'medial' ? 'medial' : 'euler'
  if (linkedOverlays && linkedOverlays.length <= overlayKeys.length && /^[01]+$/.test(linkedOverlays)) {
    overlayKeys.slice(0, linkedOverlays.length).forEach((key, index) => {
      atlasState.overlays[key] = linkedOverlays[index] === '1'
    })
  }
}

hydrateLinkedState()
atlasShell()
