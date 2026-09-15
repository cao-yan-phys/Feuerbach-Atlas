import { buildIndicatrixConstruction, lorentzRenderRapidity, type IndicatrixMode, type LorentzFinslerShape, type NormedShape } from '../math/indicatrixKernel'
import type { Vec2 } from '../math/types'

const namespace = 'http://www.w3.org/2000/svg'
const width = 1000
const height = 700

export interface IndicatrixOverlays {
  medians: boolean
  circum: boolean
  translated: boolean
  feuerbach: boolean
  midpoints: boolean
  centers: boolean
}

export interface IndicatrixViewportTransform {
  toScreen: (point: Vec2) => Vec2
  toWorld: (point: Vec2) => Vec2
  bounds: IndicatrixBounds
}

export interface IndicatrixBounds {
  halfWidth: number
  halfHeight: number
}

export interface IndicatrixSimilarity {
  translation: Vec2
  scale: number
}

const element = <T extends keyof SVGElementTagNameMap>(tag: T, attributes: Record<string, string | number> = {}) => {
  const node = document.createElementNS(namespace, tag)
  Object.entries(attributes).forEach(([name, value]) => node.setAttribute(name, String(value)))
  return node as SVGElementTagNameMap[T]
}

const append = <T extends SVGElement>(parent: SVGElement, child: T) => {
  parent.append(child)
  return child
}

const pathData = (points: Vec2[], transform: IndicatrixViewportTransform, closed = false) => `${points.map((point, index) => {
  const [x, y] = transform.toScreen(point)
  return `${index === 0 ? 'M' : 'L'}${x.toFixed(3)} ${y.toFixed(3)}`
}).join(' ')}${closed ? ' Z' : ''}`

const transformFor = (points: Vec2[], fixedBounds?: IndicatrixBounds): IndicatrixViewportTransform => {
  const finite = points.filter((point) => point.every(Number.isFinite))
  let halfWidth = fixedBounds?.halfWidth ?? 1.12 * Math.max(1, ...finite.map((point) => Math.abs(point[0])))
  let halfHeight = fixedBounds?.halfHeight ?? 1.12 * Math.max(1, ...finite.map((point) => Math.abs(point[1])))
  if (!fixedBounds) {
    const aspect = width / height
    const current = halfWidth / halfHeight
    if (current < aspect) {
      halfWidth = halfHeight * aspect
    } else {
      halfHeight = halfWidth / aspect
    }
  }
  const scale = Math.min(width / (2 * halfWidth), height / (2 * halfHeight))
  return {
    toScreen: ([x, y]) => [width / 2 + x * scale, height / 2 - y * scale],
    toWorld: ([x, y]) => [(x - width / 2) / scale, (height / 2 - y) / scale],
    bounds: { halfWidth, halfHeight }
  }
}

const drawPath = (root: SVGElement, points: Vec2[], transform: IndicatrixViewportTransform, className: string, closed = false) => {
  if (points.length < 2) {
    return
  }
  append(root, element('path', { d: pathData(points, transform, closed), class: className, fill: 'none' }))
}

const drawMarker = (root: SVGElement, point: Vec2, transform: IndicatrixViewportTransform, className: string, radius: number, attribute?: Record<string, string>) => {
  const [cx, cy] = transform.toScreen(point)
  append(root, element('circle', { cx, cy, r: radius, class: className, ...attribute }))
}

const drawTriangle = (root: SVGElement, vertices: [Vec2, Vec2, Vec2], transform: IndicatrixViewportTransform, mapPoint: (point: Vec2) => Vec2) => {
  vertices.forEach((_, index) => {
    const side: [Vec2, Vec2] = [vertices[(index + 1) % 3]!, vertices[(index + 2) % 3]!]
    drawPath(root, side.map(mapPoint), transform, `indicatrix-triangle vertex-${index}`)
  })
}

const drawCone = (root: SVGElement, reach: number, transform: IndicatrixViewportTransform, mapPoint: (point: Vec2) => Vec2) => {
  const rightRay: [Vec2, Vec2] = [[0, 0], [reach, reach]]
  const leftRay: [Vec2, Vec2] = [[0, 0], [-reach, reach]]
  drawPath(root, rightRay.map(mapPoint), transform, 'lorentz-cone')
  drawPath(root, leftRay.map(mapPoint), transform, 'lorentz-cone')
}

const isOutsideViewport = ([x, y]: Vec2) => x <= 0 || x >= width || y <= 0 || y >= height

export const renderIndicatrixViewport = (svg: SVGSVGElement, mode: IndicatrixMode, positions: [number, number, number], normedShape: NormedShape, lorentzShape: LorentzFinslerShape, overlays: IndicatrixOverlays, similarity: IndicatrixSimilarity = { translation: [0, 0], scale: 1 }, centroidTransformActive = false, fixedBounds?: IndicatrixBounds): { transform: IndicatrixViewportTransform; valid: boolean } => {
  svg.replaceChildren()
  const construction = buildIndicatrixConstruction(mode, positions, normedShape, lorentzShape)
  const mapPoint = ([x, y]: Vec2): Vec2 => [similarity.translation[0] + similarity.scale * x, similarity.translation[1] + similarity.scale * y]
  const structurePoints: Vec2[] = [
    ...construction.vertices,
    construction.origin,
    construction.centroid,
    construction.orthocenter,
    construction.feuerbachCenter,
    ...construction.translatedCenters,
    ...construction.sideMidpoints,
    ...construction.vertexOrthocenterMidpoints
  ]
  const points = mode === 'normed'
    ? [...structurePoints, ...construction.circumIndicatrix, ...construction.translatedIndicatrices.flat(), ...construction.feuBranches.flat()]
    : structurePoints
  const transform = transformFor(points.map(mapPoint), fixedBounds)
  let display = construction
  if (mode === 'lorentz-finsler') {
    let rapidity = lorentzRenderRapidity
    while (rapidity < 24) {
      const candidates = [
        display.circumIndicatrix[0]!,
        display.circumIndicatrix.at(-1)!,
        ...display.translatedIndicatrices.flatMap((curve) => [curve[0]!, curve.at(-1)!]),
        ...display.feuBranches.flatMap((curve) => [curve[0]!, curve.at(-1)!])
      ]
      if (candidates.every((point) => isOutsideViewport(transform.toScreen(mapPoint(point))))) {
        break
      }
      rapidity *= 2
      display = buildIndicatrixConstruction(mode, positions, normedShape, lorentzShape, rapidity)
    }
  }
  const root = append(svg, element('g'))
  const drawMappedPath = (points: Vec2[], className: string, closed = false) => drawPath(root, points.map(mapPoint), transform, className, closed)
  const drawMappedMarker = (point: Vec2, className: string, radius: number, attribute?: Record<string, string>) => drawMarker(root, mapPoint(point), transform, className, radius, attribute)

  if (mode === 'lorentz-finsler') {
    const reach = 20 * Math.max(1, ...structurePoints.map((point) => Math.max(Math.abs(point[0]), Math.abs(point[1]))))
    drawCone(root, reach, transform, mapPoint)
  }
  if (overlays.circum) {
    drawMappedPath(display.circumIndicatrix, 'indicatrix-circum', mode === 'normed')
  }
  if (display.valid && overlays.medians) {
    display.vertices.forEach((vertex, index) => drawMappedPath([vertex, display.sideMidpoints[index]!], `indicatrix-median vertex-${index}`))
  }
  if (display.valid && overlays.translated) {
    display.translatedIndicatrices.forEach((curve, index) => {
      drawMappedPath(curve, `indicatrix-translated vertex-${index}`, mode === 'normed')
    })
  }
  if (display.valid && overlays.feuerbach) {
    display.feuBranches.forEach((curve, index) => drawMappedPath(curve, index === 0 ? 'indicatrix-feuerbach-forward' : 'indicatrix-feuerbach-reflected', mode === 'normed'))
  }
  if (display.valid && overlays.midpoints) {
    display.vertices.forEach((vertex, index) => drawMappedPath([vertex, display.orthocenter], `indicatrix-orthocenter-link vertex-${index}`))
  }
  drawTriangle(root, display.vertices, transform, mapPoint)
  if (display.valid && overlays.midpoints) {
    display.sideMidpoints.forEach((point, index) => drawMappedMarker(point, `indicatrix-side-midpoint vertex-${index}`, 3.7))
    display.vertexOrthocenterMidpoints.forEach((point, index) => drawMappedMarker(point, `indicatrix-vertex-midpoint vertex-${index}`, 3.9))
  }
  if (display.valid && overlays.centers) {
    drawMappedMarker(display.origin, 'indicatrix-origin', 3.8)
    drawMappedMarker(display.centroid, `indicatrix-centroid${centroidTransformActive ? ' is-active' : ''}`, 3.8)
  }
  if (display.valid && overlays.translated) {
    drawMappedMarker(display.orthocenter, 'indicatrix-orthocenter', 4.5)
  }
  display.vertices.forEach((point, index) => drawMappedMarker(point, `vertex vertex-${index}`, 6, { 'data-indicatrix-vertex': String(index) }))
  return { transform, valid: display.valid }
}
