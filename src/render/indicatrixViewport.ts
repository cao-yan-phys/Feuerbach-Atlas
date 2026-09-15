import { buildIndicatrixConstruction, lorentzRenderRapidity, type IndicatrixMode, type LorentzFinslerShape, type NormedShape } from '../math/indicatrixKernel'
import type { Vec2 } from '../math/types'

const namespace = 'http://www.w3.org/2000/svg'
const mathNamespace = 'http://www.w3.org/1998/Math/MathML'
const width = 1000
const height = 700

export interface IndicatrixOverlays {
  medians: boolean
  circum: boolean
  translated: boolean
  feuerbach: boolean
  midpoints: boolean
  centers: boolean
  grid: boolean
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

export const indicatrixViewportBounds: IndicatrixBounds = { halfWidth: 2.2, halfHeight: 1.54 }

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

const gridStep = (span: number) => {
  const base = 10 ** Math.floor(Math.log10(span / 7))
  const ratio = span / (7 * base)
  return ratio <= 1 ? base : ratio <= 2 ? 2 * base : ratio <= 5 ? 5 * base : 10 * base
}

const gridValues = (minimum: number, maximum: number, step: number) => {
  const first = Math.ceil(minimum / step - 1e-10)
  const last = Math.floor(maximum / step + 1e-10)
  return Array.from({ length: Math.max(0, last - first + 1) }, (_, index) => (first + index) * step)
}

const gridText = (value: number, step: number) => {
  const decimals = Math.max(0, -Math.floor(Math.log10(step)))
  return Math.abs(value) <= step * 1e-8 ? '0' : Number(value.toFixed(decimals)).toString()
}

const drawGridLabel = (root: SVGElement, value: string, x: number, y: number, anchor: 'start' | 'middle' | 'end') => {
  const label = append(root, element('text', { x: x.toFixed(2), y: y.toFixed(2), class: 'grid-label', 'text-anchor': anchor }))
  label.textContent = value
}

const terminalSegmentCenter = (values: number[]) => values.length < 2 ? null : (values[values.length - 2]! + values[values.length - 1]!) / 2

const drawGridAxisDirection = (root: SVGElement, transform: IndicatrixViewportTransform, axis: 'x' | 'y', point: Vec2) => {
  const [centerX, centerY] = transform.toScreen(point)
  const arrow = axis === 'x'
    ? `M${(centerX - 12).toFixed(2)} ${centerY.toFixed(2)} L${(centerX + 12).toFixed(2)} ${centerY.toFixed(2)} M${(centerX + 12).toFixed(2)} ${centerY.toFixed(2)} L${(centerX + 6).toFixed(2)} ${(centerY - 5).toFixed(2)} M${(centerX + 12).toFixed(2)} ${centerY.toFixed(2)} L${(centerX + 6).toFixed(2)} ${(centerY + 5).toFixed(2)}`
    : `M${centerX.toFixed(2)} ${(centerY + 12).toFixed(2)} L${centerX.toFixed(2)} ${(centerY - 12).toFixed(2)} M${centerX.toFixed(2)} ${(centerY - 12).toFixed(2)} L${(centerX - 5).toFixed(2)} ${(centerY - 6).toFixed(2)} M${centerX.toFixed(2)} ${(centerY - 12).toFixed(2)} L${(centerX + 5).toFixed(2)} ${(centerY - 6).toFixed(2)}`
  append(root, element('path', { d: arrow, class: 'grid-axis-arrow' }))
  const label = append(root, element('foreignObject', {
    x: (axis === 'x' ? centerX - 12 : centerX + 10).toFixed(2),
    y: (axis === 'x' ? centerY + 7 : centerY - 12).toFixed(2),
    width: 24,
    height: 24,
    class: 'grid-coordinate-label'
  }))
  const math = document.createElementNS(mathNamespace, 'math')
  const identifier = document.createElementNS(mathNamespace, 'mi')
  identifier.setAttribute('mathvariant', 'italic')
  identifier.textContent = axis
  math.append(identifier)
  label.append(math)
}

const drawGrid = (root: SVGElement, transform: IndicatrixViewportTransform) => {
  const { halfWidth, halfHeight } = transform.bounds
  const minX = -halfWidth
  const maxX = halfWidth
  const minY = -halfHeight
  const maxY = halfHeight
  const xStep = gridStep(maxX - minX)
  const yStep = gridStep(maxY - minY)
  const xValues = gridValues(minX, maxX, xStep)
  const yValues = gridValues(minY, maxY, yStep)
  const tickSize = Math.min(maxX - minX, maxY - minY) * 0.009
  xValues.forEach((x) => {
    if (Math.abs(x) > xStep * 1e-8) {
      drawPath(root, [[x, minY], [x, maxY]], transform, 'grid-line')
    }
    drawPath(root, [[x, -tickSize], [x, tickSize]], transform, 'grid-tick')
    const [screenX, screenY] = transform.toScreen([x, 0])
    drawGridLabel(root, gridText(x, xStep), screenX, Math.min(height - 6, screenY + 15), 'middle')
  })
  yValues.forEach((y) => {
    if (Math.abs(y) > yStep * 1e-8) {
      drawPath(root, [[minX, y], [maxX, y]], transform, 'grid-line')
    }
    drawPath(root, [[-tickSize, y], [tickSize, y]], transform, 'grid-tick')
    if (Math.abs(y) > yStep * 1e-8) {
      const [screenX, screenY] = transform.toScreen([0, y])
      drawGridLabel(root, gridText(y, yStep), Math.max(6, screenX - 7), screenY + 4, 'end')
    }
  })
  drawPath(root, [[minX, 0], [maxX, 0]], transform, 'grid-axis')
  drawPath(root, [[0, minY], [0, maxY]], transform, 'grid-axis')
  const xDirection = terminalSegmentCenter(xValues)
  const yDirection = terminalSegmentCenter(yValues)
  if (xDirection !== null) {
    drawGridAxisDirection(root, transform, 'x', [xDirection, 0])
  }
  if (yDirection !== null) {
    drawGridAxisDirection(root, transform, 'y', [0, yDirection])
  }
}

export const renderIndicatrixViewport = (svg: SVGSVGElement, mode: IndicatrixMode, vertices: [Vec2, Vec2, Vec2], normedShape: NormedShape, lorentzShape: LorentzFinslerShape, overlays: IndicatrixOverlays, similarity: IndicatrixSimilarity = { translation: [0, 0], scale: 1 }, centroidTransformActive = false, fixedBounds: IndicatrixBounds = indicatrixViewportBounds): { transform: IndicatrixViewportTransform; valid: boolean } => {
  svg.replaceChildren()
  const construction = buildIndicatrixConstruction(mode, vertices, normedShape, lorentzShape)
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
      display = buildIndicatrixConstruction(mode, vertices, normedShape, lorentzShape, rapidity)
    }
  }
  const root = append(svg, element('g'))
  const drawMappedPath = (points: Vec2[], className: string, closed = false) => drawPath(root, points.map(mapPoint), transform, className, closed)
  const drawMappedMarker = (point: Vec2, className: string, radius: number, attribute?: Record<string, string>) => drawMarker(root, mapPoint(point), transform, className, radius, attribute)

  if (overlays.grid) {
    drawGrid(root, transform)
  }
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
    drawPath(root, [display.origin, display.centroid, display.orthocenter].map(mapPoint), transform, 'indicatrix-euler-line')
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
