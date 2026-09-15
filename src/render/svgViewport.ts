import { hyperbolicProject, lorentzProjectiveProject, sphereProject } from '../geometry/charts'
import { boundsForMode, curvedDomain, isCurvedMode, liftForMode, metricForMode } from '../geometry/modes'
import { buildCurvedState, sampleGeodesic } from '../math/curvedKernel'
import { buildFlatState, euclideanCircumcircle, flatCycleHomothety, minkowskiCircumcycle } from '../math/flatKernel'
import { bilinear2, bilinear3, scale3 } from '../math/linalg'
import { buildParabolicState, canonicalToRawParabolic, isParabolicMode, parabolicCycleValue, parabolicDisplayPoint, parabolicLineThrough, parabolicSideValue, type ParabolicChart, type ParabolicParameters, type ParabolicView } from '../math/parabolicKernel'
import type { FlatCycle, GeometryMode, Vec2, Vec3 } from '../math/types'
import { marchingSquares, type Bounds } from './marchingSquares'

const namespace = 'http://www.w3.org/2000/svg'
const mathNamespace = 'http://www.w3.org/1998/Math/MathML'
const width = 1000
const height = 700
const margin = 0
const contourColumns = 300
const contourRows = 210

export interface Overlays {
  bisectors: boolean
  euler: boolean
  altitudes: boolean
  tangent: boolean
  centers: boolean
  circumcircle: boolean
  euclideanCircumcircle: boolean
  minkowskiCircumcircle: boolean
  nullBoundary: boolean
  horizon: boolean
  homothety: boolean
  grid: boolean
  singularBranches: boolean
}

export interface ViewportTransform {
  bounds: Bounds
  toScreen: (point: Vec2) => Vec2
  toWorld: (point: Vec2) => Vec2
}

const createTransform = (bounds: Bounds, preserveBounds = false): ViewportTransform => {
  const contentWidth = width - 2 * margin
  const contentHeight = height - 2 * margin
  const worldWidth = bounds.maxX - bounds.minX
  const worldHeight = bounds.maxY - bounds.minY
  const targetAspect = contentWidth / contentHeight
  const worldAspect = worldWidth / worldHeight
  const adjusted = { ...bounds }

  if (!preserveBounds) {
    if (worldAspect < targetAspect) {
      const extra = (worldHeight * targetAspect - worldWidth) / 2
      adjusted.minX -= extra
      adjusted.maxX += extra
    } else if (worldAspect > targetAspect) {
      const extra = (worldWidth / targetAspect - worldHeight) / 2
      adjusted.minY -= extra
      adjusted.maxY += extra
    }
  }

  const scale = Math.min(contentWidth / (adjusted.maxX - adjusted.minX), contentHeight / (adjusted.maxY - adjusted.minY))
  const offsetX = (width - scale * (adjusted.maxX - adjusted.minX)) / 2
  const offsetY = (height - scale * (adjusted.maxY - adjusted.minY)) / 2

  return {
    bounds: adjusted,
    toScreen: ([x, y]) => [offsetX + (x - adjusted.minX) * scale, height - offsetY - (y - adjusted.minY) * scale],
    toWorld: ([x, y]) => [(x - offsetX) / scale + adjusted.minX, (height - offsetY - y) / scale + adjusted.minY]
  }
}

const element = <T extends keyof SVGElementTagNameMap>(tag: T, attributes: Record<string, string | number> = {}): SVGElementTagNameMap[T] => {
  const value = document.createElementNS(namespace, tag)
  Object.entries(attributes).forEach(([name, attribute]) => value.setAttribute(name, String(attribute)))
  return value
}

const append = <T extends SVGElement>(parent: SVGElement, child: T) => {
  parent.append(child)
  return child
}

const pathData = (points: Vec2[], transform: ViewportTransform) => points.map((point, index) => {
  const [x, y] = transform.toScreen(point)
  return `${index === 0 ? 'M' : 'L'}${x.toFixed(3)} ${y.toFixed(3)}`
}).join(' ')

const samePoint = (first: Vec2, second: Vec2) => Math.abs(first[0] - second[0]) <= 1e-9 && Math.abs(first[1] - second[1]) <= 1e-9

const clipSegment = (start: Vec2, end: Vec2, bounds: Bounds): [Vec2, Vec2] | null => {
  const [x0, y0] = start
  const [x1, y1] = end
  const dx = x1 - x0
  const dy = y1 - y0
  const constraints: Array<[number, number]> = [
    [-dx, x0 - bounds.minX],
    [dx, bounds.maxX - x0],
    [-dy, y0 - bounds.minY],
    [dy, bounds.maxY - y0]
  ]
  let from = 0
  let to = 1

  for (const [p, q] of constraints) {
    if (Math.abs(p) <= 1e-12) {
      if (q < 0) {
        return null
      }
      continue
    }
    const ratio = q / p
    if (p < 0) {
      from = Math.max(from, ratio)
    } else {
      to = Math.min(to, ratio)
    }
  }

  if (from > to) {
    return null
  }

  return [[x0 + dx * from, y0 + dy * from], [x0 + dx * to, y0 + dy * to]]
}

const clippedPolylines = (points: Vec2[], bounds: Bounds) => {
  const lines: Vec2[][] = []
  let line: Vec2[] = []

  for (let index = 1; index < points.length; index += 1) {
    const section = clipSegment(points[index - 1]!, points[index]!, bounds)
    if (!section) {
      if (line.length > 1) {
        lines.push(line)
      }
      line = []
      continue
    }
    if (line.length === 0) {
      line = [section[0], section[1]]
      continue
    }
    if (samePoint(line[line.length - 1]!, section[0])) {
      line.push(section[1])
    } else {
      if (line.length > 1) {
        lines.push(line)
      }
      line = [section[0], section[1]]
    }
  }

  if (line.length > 1) {
    lines.push(line)
  }

  return lines
}

const screenLength = (points: Vec2[], transform: ViewportTransform) => points.slice(1).reduce((length, point, index) => {
  const previous = transform.toScreen(points[index]!)
  const current = transform.toScreen(point)
  return length + Math.hypot(current[0] - previous[0], current[1] - previous[1])
}, 0)

const indexedClass = (className: string, index: number) => `${className} vertex-${index}`

const tangentConstructionClass = (branch: number) => branch === 0 ? 'incenter-construction' : `excenter-construction excenter-${branch}`

const tangentCycleClass = (branch: number) => `cycle-tangent ${tangentConstructionClass(branch)}`

const tangentCenterClass = (branch: number) => `${branch === 0 ? 'incenter-marker' : 'excenter-marker'} ${tangentConstructionClass(branch)}`

const tangentIdealMarkerClass = (branch: number, contact = false) => branch === 0
  ? contact ? 'ideal-contact-marker' : 'incenter-ideal-marker'
  : `excenter-ideal-marker excenter-${branch}`

const drawPolyline = (root: SVGElement, points: Vec2[], transform: ViewportTransform, className: string) => {
  if (points.length < 2 || points.every((point) => samePoint(point, points[0]!))) {
    return
  }
  clippedPolylines(points, transform.bounds).forEach((line) => {
    if (!line.every((point) => samePoint(point, line[0]!)) && screenLength(line, transform) >= 0.5) {
      append(root, element('path', { d: pathData(line, transform), class: className, fill: 'none' }))
    }
  })
}

const drawFlatTriangle = (root: SVGElement, vertices: [Vec2, Vec2, Vec2], transform: ViewportTransform, emphasizedSide: number | null = null) => {
  vertices.forEach((_, index) => drawPolyline(root, [vertices[(index + 1) % 3]!, vertices[(index + 2) % 3]!], transform, `${indexedClass('triangle', index)}${index === emphasizedSide ? ' lambda-negative-side' : ''}`))
}

const drawMarker = (root: SVGElement, point: Vec2, transform: ViewportTransform, className: string, radius = 4) => {
  const [cx, cy] = transform.toScreen(point)
  if (!Number.isFinite(cx) || !Number.isFinite(cy) || cx < -radius || cx > width + radius || cy < -radius || cy > height + radius) {
    return
  }
  append(root, element('circle', { cx, cy, r: radius, class: className }))
}

const drawStarMarker = (root: SVGElement, point: Vec2, transform: ViewportTransform, className: string, radius = 6) => {
  const [cx, cy] = transform.toScreen(point)
  if (!Number.isFinite(cx) || !Number.isFinite(cy) || cx < -radius || cx > width + radius || cy < -radius || cy > height + radius) {
    return
  }
  const vertices = Array.from({ length: 10 }, (_, index) => {
    const angle = -Math.PI / 2 + index * Math.PI / 5
    const distance = index % 2 === 0 ? radius : radius * 0.42
    return [cx + Math.cos(angle) * distance, cy + Math.sin(angle) * distance] as Vec2
  })
  const path = vertices.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(3)} ${y.toFixed(3)}`).join(' ') + ' Z'
  append(root, element('path', { d: path, class: className, 'data-euclidean-circumcenter': 'true' }))
}

const drawRingMarker = (root: SVGElement, point: Vec2, transform: ViewportTransform, className: string, radius = 4) => {
  const [cx, cy] = transform.toScreen(point)
  if (!Number.isFinite(cx) || !Number.isFinite(cy) || cx < -radius || cx > width + radius || cy < -radius || cy > height + radius) {
    return
  }
  append(root, element('circle', { cx, cy, r: radius, class: className, fill: 'none', 'data-boost-toggle': 'true' }))
}

const gridStep = (span: number) => {
  const magnitude = 10 ** Math.floor(Math.log10(span / 8))
  return [1, 2, 5, 10].map((factor) => factor * magnitude).find((value) => span / value <= 10) ?? magnitude
}

const gridValues = (minimum: number, maximum: number, step: number) => {
  const first = Math.ceil((minimum - step * 1e-8) / step)
  const last = Math.floor((maximum + step * 1e-8) / step)
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

const drawGridAxisDirection = (root: SVGElement, transform: ViewportTransform, axis: 'x' | 'y', point: Vec2, labelText: string = axis) => {
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
  identifier.textContent = labelText
  math.append(identifier)
  label.append(math)
}

const drawGrid = (root: SVGElement, transform: ViewportTransform, labels: [string, string] = ['x', 'y']) => {
  const { minX, maxX, minY, maxY } = transform.bounds
  const xStep = gridStep(maxX - minX)
  const yStep = gridStep(maxY - minY)
  const xValues = gridValues(minX, maxX, xStep)
  const yValues = gridValues(minY, maxY, yStep)
  const xAxis = minY <= 0 && maxY >= 0 ? 0 : minY
  const yAxis = minX <= 0 && maxX >= 0 ? 0 : minX
  const tickSize = Math.min(maxX - minX, maxY - minY) * 0.009

  xValues.forEach((x) => {
    if (Math.abs(x) > xStep * 1e-8) {
      drawPolyline(root, [[x, minY], [x, maxY]], transform, 'grid-line')
    }
    drawPolyline(root, [[x, xAxis - tickSize], [x, xAxis + tickSize]], transform, 'grid-tick')
    const [screenX, screenY] = transform.toScreen([x, xAxis])
    drawGridLabel(root, gridText(x, xStep), screenX, Math.min(height - 6, screenY + 15), 'middle')
  })

  yValues.forEach((y) => {
    if (Math.abs(y) > yStep * 1e-8) {
      drawPolyline(root, [[minX, y], [maxX, y]], transform, 'grid-line')
    }
    drawPolyline(root, [[yAxis - tickSize, y], [yAxis + tickSize, y]], transform, 'grid-tick')
    if (Math.abs(y) > yStep * 1e-8) {
      const [screenX, screenY] = transform.toScreen([yAxis, y])
      drawGridLabel(root, gridText(y, yStep), Math.max(6, screenX - 7), screenY + 4, 'end')
    }
  })

  if (minY <= 0 && maxY >= 0) {
    drawPolyline(root, [[minX, 0], [maxX, 0]], transform, 'grid-axis')
    const xDirection = terminalSegmentCenter(xValues)
    if (xDirection !== null) {
      drawGridAxisDirection(root, transform, 'x', [xDirection, 0], labels[0])
    }
  }
  if (minX <= 0 && maxX >= 0) {
    drawPolyline(root, [[0, minY], [0, maxY]], transform, 'grid-axis')
    const yDirection = terminalSegmentCenter(yValues)
    if (yDirection !== null) {
      drawGridAxisDirection(root, transform, 'y', [0, yDirection], labels[1])
    }
  }
}

export const projectiveBoundaryIntersection = (point: Vec2, neighbor: Vec2, maximumDistance = Infinity): Vec2 | null => {
  const dx = point[0] - neighbor[0]
  const dy = point[1] - neighbor[1]
  const a = dx * dx - dy * dy
  const b = 2 * (point[0] * dx - point[1] * dy)
  const c = 1 + point[0] * point[0] - point[1] * point[1]
  const discriminant = b * b - 4 * a * c
  if (discriminant < 0) {
    return null
  }
  const roots = Math.abs(a) <= 1e-12
    ? Math.abs(b) <= 1e-12 ? [] : [-c / b]
    : [(-b - Math.sqrt(discriminant)) / (2 * a), (-b + Math.sqrt(discriminant)) / (2 * a)]
  const parameter = roots.filter((value) => value >= 0).sort((left, right) => left - right)[0]
  if (parameter === undefined) {
    return null
  }
  const candidate: Vec2 = [point[0] + dx * parameter, point[1] + dy * parameter]
  return Math.hypot(candidate[0] - point[0], candidate[1] - point[1]) <= maximumDistance ? candidate : null
}

const closestPoint = (point: Vec2, candidates: Vec2[], maximumDistance: number) => candidates
  .map((candidate) => ({ candidate, distance: Math.hypot(candidate[0] - point[0], candidate[1] - point[1]) }))
  .filter(({ distance }) => distance <= maximumDistance)
  .sort((left, right) => left.distance - right.distance)[0]?.candidate ?? null

const snapProjectiveEndpoints = (line: Vec2[], maximumDistance: number, boundaryPoints: Vec2[]) => {
  if (line.length < 3 || samePoint(line[0]!, line[line.length - 1]!)) {
    return line
  }
  const snapped = line.slice()
  const first = line[0]!
  const last = line[line.length - 1]!
  snapped[0] = closestPoint(first, boundaryPoints, maximumDistance) ?? projectiveBoundaryIntersection(first, line[1]!, maximumDistance) ?? first
  snapped[snapped.length - 1] = closestPoint(last, boundaryPoints, maximumDistance) ?? projectiveBoundaryIntersection(last, line[line.length - 2]!, maximumDistance) ?? last
  return snapped
}

const drawContours = (root: SVGElement, sample: (point: Vec2) => number, transform: ViewportTransform, className: string) => {
  marchingSquares(sample, transform.bounds, contourColumns, contourRows).forEach((line) => {
    drawPolyline(root, line, transform, className)
  })
}

const drawFlatCycle = (root: SVGElement, sigma: 1 | -1, cycle: FlatCycle, transform: ViewportTransform, className: string) => {
  drawContours(root, (point) => {
    const relative: Vec2 = [point[0] - cycle.center[0], point[1] - cycle.center[1]]
    return bilinear2(sigma, relative, relative) - cycle.radiusSquared
  }, transform, className)
}

const drawFlatHomothety = (root: SVGElement, sigma: 1 | -1, first: FlatCycle, center: Vec2, scale: number, progress: number, transform: ViewportTransform, className: string) => {
  const imageScale = 1 + progress * (scale - 1)
  if (Math.abs(imageScale) > 1e-10) {
    drawFlatCycle(root, sigma, {
      center: [center[0] + imageScale * (first.center[0] - center[0]), center[1] + imageScale * (first.center[1] - center[1])],
      radiusSquared: imageScale * imageScale * first.radiusSquared,
      label: 'Homothety image'
    }, transform, `${className} cycle-homothety-image`)
  }
}

const projectiveEndpointClass = (cycleClass: string) => (point: Vec2) => {
  if (Math.abs(1 + point[0] * point[0] - point[1] * point[1]) >= 1e-6) {
    return null
  }
  if (cycleClass.includes('cycle-euler')) {
    return 'euler-ideal-marker'
  }
  const branch = cycleClass.match(/excenter-(\d)/)?.[1]
  return branch ? `excenter-ideal-marker excenter-${branch}` : 'tangent-ideal-marker'
}

const projectiveValue = ([x, y]: Vec2) => 1 + x * x - y * y

const pointAt = (start: Vec2, end: Vec2, parameter: number): Vec2 => [
  start[0] + (end[0] - start[0]) * parameter,
  start[1] + (end[1] - start[1]) * parameter
]

export const projectiveSegmentPaths = (start: Vec2, end: Vec2): Vec2[][] => {
  const dx = end[0] - start[0]
  const dy = end[1] - start[1]
  const a = dx * dx - dy * dy
  const b = 2 * (start[0] * dx - start[1] * dy)
  const c = projectiveValue(start)
  const discriminant = b * b - 4 * a * c
  const roots = Math.abs(a) <= 1e-12
    ? Math.abs(b) <= 1e-12 ? [] : [-c / b]
    : discriminant < 0 ? [] : [(-b - Math.sqrt(discriminant)) / (2 * a), (-b + Math.sqrt(discriminant)) / (2 * a)]
  const cuts = [0, ...roots.filter((root) => root > 1e-12 && root < 1 - 1e-12).sort((left, right) => left - right), 1]
  const paths: Vec2[][] = []

  for (let index = 1; index < cuts.length; index += 1) {
    const from = cuts[index - 1]!
    const to = cuts[index]!
    if (projectiveValue(pointAt(start, end, (from + to) / 2)) > 0) {
      paths.push([pointAt(start, end, from), pointAt(start, end, to)])
    }
  }

  return paths
}

export const diskSegmentPaths = (start: Vec2, end: Vec2): Vec2[][] => {
  const dx = end[0] - start[0]
  const dy = end[1] - start[1]
  const a = dx * dx + dy * dy
  const b = 2 * (start[0] * dx + start[1] * dy)
  const c = start[0] * start[0] + start[1] * start[1] - 1
  const discriminant = b * b - 4 * a * c
  const roots = Math.abs(a) <= 1e-12
    ? []
    : discriminant < 0 ? [] : [(-b - Math.sqrt(discriminant)) / (2 * a), (-b + Math.sqrt(discriminant)) / (2 * a)]
  const cuts = [0, ...roots.filter((root) => root > 1e-12 && root < 1 - 1e-12).sort((left, right) => left - right), 1]
  const paths: Vec2[][] = []

  for (let index = 1; index < cuts.length; index += 1) {
    const from = cuts[index - 1]!
    const to = cuts[index]!
    const midpoint = pointAt(start, end, (from + to) / 2)
    if (midpoint[0] * midpoint[0] + midpoint[1] * midpoint[1] < 1) {
      paths.push([pointAt(start, end, from), pointAt(start, end, to)])
    }
  }

  return paths
}

export const projectiveCycleBoundaryPoints = (normal: Vec3): Vec2[] => {
  const scale = Math.max(Math.abs(normal[0]), Math.abs(normal[1]), Math.abs(normal[2]))
  if (scale <= 1e-12) {
    return []
  }
  const [n0, n1, n2] = normal.map((value) => value / scale) as Vec3

  if (Math.abs(n2) <= 1e-12) {
    if (Math.abs(n1) <= 1e-12) {
      return []
    }
    const u = -n0 / n1
    const v = Math.sqrt(1 + u * u)
    return [[u, v], [u, -v]]
  }

  const a = n2 * n2 - n1 * n1
  const b = -2 * n0 * n1
  const c = n2 * n2 - n0 * n0
  const discriminant = b * b - 4 * a * c
  if (discriminant < -1e-12) {
    return []
  }
  const roots = Math.abs(a) <= 1e-12
    ? Math.abs(b) <= 1e-12 ? [] : [-c / b]
    : [(-b - Math.sqrt(Math.max(0, discriminant))) / (2 * a), (-b + Math.sqrt(Math.max(0, discriminant))) / (2 * a)]
  return roots.map((u) => [u, (n0 + n1 * u) / n2] as Vec2)
}

const joinPaths = (paths: Vec2[][], section: Vec2[]) => {
  const current = paths[paths.length - 1]
  if (current && samePoint(current[current.length - 1]!, section[0]!)) {
    current.push(section[1]!)
    return
  }
  paths.push(section.slice())
}

const lineBoxIntersections = (normal: Vec3, bounds: Bounds) => {
  const [n0, n1, n2] = normal
  const points: Vec2[] = []
  const add = (point: Vec2) => {
    if (point[0] >= bounds.minX - 1e-9 && point[0] <= bounds.maxX + 1e-9 && point[1] >= bounds.minY - 1e-9 && point[1] <= bounds.maxY + 1e-9 && !points.some((candidate) => samePoint(candidate, point))) {
      points.push(point)
    }
  }
  if (Math.abs(n2) > 1e-12) {
    add([bounds.minX, (n0 + n1 * bounds.minX) / n2])
    add([bounds.maxX, (n0 + n1 * bounds.maxX) / n2])
  }
  if (Math.abs(n1) > 1e-12) {
    add([(n2 * bounds.minY - n0) / n1, bounds.minY])
    add([(n2 * bounds.maxY - n0) / n1, bounds.maxY])
  }
  if (points.length < 2) {
    return []
  }
  let pair: [Vec2, Vec2] = [points[0]!, points[1]!]
  points.forEach((first, firstIndex) => points.slice(firstIndex + 1).forEach((second) => {
    if (Math.hypot(second[0] - first[0], second[1] - first[1]) > Math.hypot(pair[1][0] - pair[0][0], pair[1][1] - pair[0][1])) {
      pair = [first, second]
    }
  }))
  return projectiveSegmentPaths(pair[0], pair[1])
}

const drawProjectiveCycle = (root: SVGElement, normal: Vec3, offset: number, transform: ViewportTransform, className: string) => {
  const normalScale = Math.max(Math.abs(normal[0]), Math.abs(normal[1]), Math.abs(normal[2]))
  if (normalScale <= 1e-12) {
    return
  }
  const normalizedNormal = normal.map((value) => value / normalScale) as Vec3
  const normalizedOffset = offset / normalScale
  const boundaryPoints = projectiveCycleBoundaryPoints(normalizedNormal)
  const step = Math.hypot((transform.bounds.maxX - transform.bounds.minX) / contourColumns, (transform.bounds.maxY - transform.bounds.minY) / contourRows)
  const drawPath = (path: Vec2[]) => {
    const contour = snapProjectiveEndpoints(path, step * 4, boundaryPoints)
    drawPolyline(root, contour, transform, className)
  }

  if (Math.abs(normalizedOffset) <= 1e-10) {
    lineBoxIntersections(normalizedNormal, transform.bounds).forEach(drawPath)
    return
  }

  const paths: Vec2[][] = []
  marchingSquares((point) => {
    const linear = normalizedNormal[0] + normalizedNormal[1] * point[0] - normalizedNormal[2] * point[1]
    return linear * linear - normalizedOffset * normalizedOffset * projectiveValue(point)
  }, transform.bounds, contourColumns, contourRows).forEach((line) => {
    let current: Vec2[][] = []
    for (let index = 1; index < line.length; index += 1) {
      projectiveSegmentPaths(line[index - 1]!, line[index]!).forEach((section) => {
        const midpoint = pointAt(section[0]!, section[1]!, 0.5)
        const linear = normalizedNormal[0] + normalizedNormal[1] * midpoint[0] - normalizedNormal[2] * midpoint[1]
        if (linear * normalizedOffset > 1e-12) {
          joinPaths(current, section)
        }
      })
    }
    paths.push(...current)
  })
  paths.forEach(drawPath)
  const markerClass = projectiveEndpointClass(className)
  boundaryPoints.forEach((point) => {
    const className = markerClass(point)
    if (className) {
      drawMarker(root, point, transform, className, 4)
    }
  })
}

const drawHyperbolicCycle = (root: SVGElement, normal: Vec3, offset: number, transform: ViewportTransform, className: string) => {
  const normalScale = Math.max(Math.abs(normal[0]), Math.abs(normal[1]), Math.abs(normal[2]))
  if (normalScale <= 1e-12) {
    return
  }
  const normalizedNormal = normal.map((value) => value / normalScale) as Vec3
  const normalizedOffset = offset / normalScale
  const paths: Vec2[][] = []
  marchingSquares((point) => {
    const radiusSquared = point[0] * point[0] + point[1] * point[1]
    return (normalizedNormal[0] + normalizedOffset) * radiusSquared - 2 * normalizedNormal[1] * point[0] - 2 * normalizedNormal[2] * point[1] + normalizedNormal[0] - normalizedOffset
  }, transform.bounds, contourColumns, contourRows).forEach((line) => {
    const current: Vec2[][] = []
    for (let index = 1; index < line.length; index += 1) {
      diskSegmentPaths(line[index - 1]!, line[index]!).forEach((section) => joinPaths(current, section))
    }
    paths.push(...current)
  })
  paths.forEach((path) => {
    drawPolyline(root, path, transform, className)
    if (!samePoint(path[0]!, path[path.length - 1]!)) {
      ;[path[0]!, path[path.length - 1]!].forEach((point) => {
        if (Math.abs(point[0] * point[0] + point[1] * point[1] - 1) <= 1e-6) {
          drawMarker(root, point, transform, className.includes('cycle-euler') ? 'euler-ideal-marker' : className.includes('excenter-construction') ? 'excenter-ideal-marker' : 'tangent-ideal-marker', 4)
        }
      })
    }
  })
}

const drawCurvedCycle = (root: SVGElement, mode: GeometryMode, metric: Vec3, normal: Vec3, offset: number, transform: ViewportTransform, className: string) => {
  if (mode === 'desitter' || mode === 'ads') {
    drawProjectiveCycle(root, normal, offset, transform, className)
    return
  }
  if (mode === 'hyperbolic') {
    drawHyperbolicCycle(root, normal, offset, transform, className)
    return
  }
  drawContours(root, (point) => {
    const liftedPoint = liftForMode(mode, point)
    return liftedPoint ? bilinear3(metric, liftedPoint, normal) - offset : Number.NaN
  }, transform, className)
}

const drawBoundary = (root: SVGElement, mode: GeometryMode, transform: ViewportTransform, className = 'boundary') => {
  if (mode === 'hyperbolic') {
    const points: Vec2[] = []
    for (let index = 0; index <= 240; index += 1) {
      const angle = index / 240 * Math.PI * 2
      points.push([Math.cos(angle), Math.sin(angle)])
    }
    drawPolyline(root, points, transform, className)
    return
  }

  if (mode === 'desitter' || mode === 'ads') {
    const upper: Vec2[] = []
    const lower: Vec2[] = []
    const { minX, maxX } = transform.bounds
    for (let index = 0; index <= 240; index += 1) {
      const x = minX + (maxX - minX) * index / 240
      const y = Math.sqrt(1 + x * x)
      upper.push([x, y])
      lower.push([x, -y])
    }
    drawPolyline(root, upper, transform, className)
    drawPolyline(root, lower, transform, className)
    return
  }

  if (mode === 'minkowski') {
    const { minX, maxX, minY, maxY } = transform.bounds
    const extent = Math.max(Math.abs(minX), Math.abs(maxX), Math.abs(minY), Math.abs(maxY)) + 1
    const xValues = gridValues(minX, maxX, gridStep(maxX - minX))
    const yValues = gridValues(minY, maxY, gridStep(maxY - minY))
    const offsets = (sign: 1 | -1) => {
      const values = xValues.flatMap((x) => yValues.map((y) => Number((y - sign * x).toFixed(12))))
      return [...new Set(values)]
    }

    offsets(1).forEach((offset) => {
      drawPolyline(root, [[-extent, -extent + offset], [extent, extent + offset]], transform, className)
    })
    offsets(-1).forEach((offset) => {
      drawPolyline(root, [[-extent, extent + offset], [extent, -extent + offset]], transform, className)
    })
  }
}

const drawLorentzianNullGrid = (root: SVGElement, transform: ViewportTransform) => {
  const { minX, maxX } = transform.bounds
  const thetaMin = Math.atan(minX)
  const thetaMax = Math.atan(maxX)
  const limit = Math.PI / 2 - 1e-5
  const step = Math.PI / 8

  ;([-1, 1] as const).forEach((sign) => {
    for (let index = -7; index <= 7; index += 1) {
      const offset = index * step
      const etaMin = sign === 1 ? -limit - offset : offset - limit
      const etaMax = sign === 1 ? limit - offset : offset + limit
      const from = Math.max(thetaMin, etaMin)
      const to = Math.min(thetaMax, etaMax)
      if (to - from <= 1e-8) {
        continue
      }
      const points: Vec2[] = []
      for (let sample = 0; sample <= 180; sample += 1) {
        const theta = from + (to - from) * sample / 180
        const eta = sign * theta + offset
        points.push([Math.tan(theta), Math.sin(eta) / Math.cos(theta)])
      }
      drawPolyline(root, points, transform, 'lorentzian-null-grid')
    }
  })
}

const drawDeSitterHorizons = (root: SVGElement, transform: ViewportTransform) => {
  const { minX, maxX } = transform.bounds

  ;([-1, 1] as const).forEach((y) => {
    if (minX < 0) {
      drawPolyline(root, [[minX, y], [0, y]], transform, 'horizon')
    }
    if (maxX > 0) {
      drawPolyline(root, [[0, y], [maxX, y]], transform, 'horizon')
    }
  })
}

const projectForMode = (mode: GeometryMode, point: Vec3): Vec2 | null => {
  if (mode === 'sphere') {
    return sphereProject(point)
  }
  if (mode === 'hyperbolic') {
    return hyperbolicProject(point)
  }
  return lorentzProjectiveProject(point)
}

const accurateResiduals = (residuals: Record<string, number>) => Object.values(residuals).every((value) => Number.isFinite(value) && value <= 1e-7)

const projectedUnitPoint = (mode: GeometryMode, metric: Vec3, point: Vec3): Vec2 | null =>
  Math.abs(bilinear3(metric, point, point) - 1) <= 1e-7 ? projectForMode(mode, point) : null

const projectedPaths = (mode: GeometryMode, points: Vec3[], bounds: Bounds): Vec2[][] => {
  const paths: Vec2[][] = []
  const span = Math.hypot(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY)
  let path: Vec2[] = []

  points.forEach((point) => {
    const projected = projectForMode(mode, point)
    const previous = path[path.length - 1]
    if (!projected || (previous && Math.hypot(projected[0] - previous[0], projected[1] - previous[1]) > span)) {
      if (path.length > 1) {
        paths.push(path)
      }
      path = projected ? [projected] : []
      return
    }
    path.push(projected)
  })

  if (path.length > 1) {
    paths.push(path)
  }

  return paths
}

const curvedGeodesic = (mode: GeometryMode, metric: Vec3, start: Vec3, end: Vec3, transform: ViewportTransform): Vec2[][] => {
  if (mode === 'desitter' || mode === 'ads') {
    const startPoint = projectForMode(mode, start)
    const endPoint = projectForMode(mode, end)
    return startPoint && endPoint ? projectiveSegmentPaths(startPoint, endPoint) : []
  }
  return projectedPaths(mode, sampleGeodesic(metric, start, end), transform.bounds)
}

const drawFlat = (root: SVGElement, mode: GeometryMode, vertices: [Vec2, Vec2, Vec2], transform: ViewportTransform, overlays: Overlays, selectedBranch: number, homothetyProgress: number) => {
  const sigma = mode === 'minkowski' ? -1 : 1
  const state = buildFlatState(sigma, vertices)
  const showTheorem = state.valid
  const euclideanCircle = mode === 'minkowski' && overlays.euclideanCircumcircle ? euclideanCircumcircle(vertices) : null
  const minkowskiCircle = mode === 'euclidean' && overlays.minkowskiCircumcircle ? minkowskiCircumcycle(vertices) : null
  const tangentCycle = state.tangentCycles[selectedBranch]
  const contact = state.contacts[selectedBranch]
  const homothety = showTheorem && overlays.homothety && state.ninePoint && tangentCycle && contact
    ? flatCycleHomothety(state.ninePoint, tangentCycle, contact)
    : null

  if (euclideanCircle) {
    drawFlatCycle(root, 1, euclideanCircle, transform, 'cycle-euclidean-circumcircle')
  }

  if (minkowskiCircle) {
    drawFlatCycle(root, -1, minkowskiCircle, transform, 'cycle-minkowski-circumcircle')
  }

  if (showTheorem && overlays.circumcircle && state.circumcircle) {
    drawFlatCycle(root, sigma, state.circumcircle, transform, 'cycle-circumcircle')
  }

  if (showTheorem && overlays.euler && state.ninePoint) {
    drawFlatCycle(root, sigma, state.ninePoint, transform, 'cycle-euler')
  }

  if (homothety && state.ninePoint && tangentCycle) {
    drawFlatHomothety(root, sigma, state.ninePoint, homothety.center, homothety.scale, homothetyProgress, transform, tangentCycleClass(selectedBranch))
  }

  if (showTheorem && overlays.tangent) {
    if (tangentCycle) {
      drawFlatCycle(root, sigma, tangentCycle, transform, tangentCycleClass(selectedBranch))
    }
  }

  if (showTheorem && overlays.bisectors) {
    vertices.forEach((vertex, index) => drawPolyline(root, [vertex, state.midpoints[index]!], transform, indexedClass('bisector', index)))
  }

  if (showTheorem && overlays.altitudes) {
    vertices.forEach((vertex, index) => {
      const foot = state.altitudeFeet[index]
      if (foot) {
        drawPolyline(root, [vertex, foot], transform, indexedClass('altitude', index))
        drawPolyline(root, [foot, state.orthocenter!], transform, indexedClass('altitude', index))
      }
    })
  }

  if (showTheorem && overlays.centers) {
    if (state.circumcenter) {
      drawMarker(root, state.circumcenter, transform, 'circumcircle-center-marker', 3)
    }
    drawMarker(root, state.centroid, transform, 'area-bisector-center-marker', 3.5)
    if (state.orthocenter) {
      drawMarker(root, state.orthocenter, transform, 'altitude-center-marker', 3.5)
    }
  }

  if (showTheorem && overlays.centers) {
    if (state.ninePoint) {
      drawMarker(root, state.ninePoint.center, transform, 'ninepoint-center-marker', 3)
    }
    if (tangentCycle) {
      drawMarker(root, tangentCycle.center, transform, tangentCenterClass(selectedBranch), 3)
    }
  }

  drawFlatTriangle(root, vertices, transform, mode === 'minkowski' && state.isNonMixed ? state.sideLambdas.findIndex((value) => value < 0) : null)
  if (euclideanCircle) {
    drawStarMarker(root, euclideanCircle.center, transform, 'euclidean-circumcenter-marker')
  }
  if (minkowskiCircle) {
    drawStarMarker(root, minkowskiCircle.center, transform, 'minkowski-circumcenter-marker')
  }
  if (showTheorem) {
    state.midpoints.forEach((point, index) => drawMarker(root, point, transform, indexedClass('foot-marker', index), 3))
    state.altitudeFeet.forEach((point, index) => {
      if (point) {
        drawMarker(root, point, transform, indexedClass('altitude-marker', index), 3)
      }
    })
    if (contact?.point) {
      drawMarker(root, contact.point, transform, `contact-marker ${tangentConstructionClass(selectedBranch)}`, 4.5)
    }
  }
}

const parabolicChartIntervals = (kappa: number, parameters: ParabolicParameters, transform: ViewportTransform, chart: ParabolicChart, view: ParabolicView): Array<[number, number]> => {
  if (chart === 'natural') {
    return [[transform.bounds.minX - parameters.anchor[0], transform.bounds.maxX - parameters.anchor[0]]]
  }
  if (Math.abs(kappa) <= 1e-10) {
    const minimum = view === 'carroll' ? transform.bounds.minX : transform.bounds.minY
    const maximum = view === 'carroll' ? transform.bounds.maxX : transform.bounds.maxY
    return [[minimum - parameters.anchor[0], maximum - parameters.anchor[0]]]
  }
  if (kappa > 0) {
    const limit = 1 / Math.sqrt(kappa)
    const inset = limit * 1e-6
    const minimum = chart === 'beltrami' && view === 'carroll' ? transform.bounds.minX : transform.bounds.minY
    const maximum = chart === 'beltrami' && view === 'carroll' ? transform.bounds.maxX : transform.bounds.maxY
    const min = Math.max(minimum, -limit + inset)
    const max = Math.min(maximum, limit - inset)
    if (!(min < max)) {
      return []
    }
    return [[Math.atanh(Math.sqrt(kappa) * min) / Math.sqrt(kappa) - parameters.anchor[0], Math.atanh(Math.sqrt(kappa) * max) / Math.sqrt(kappa) - parameters.anchor[0]]]
  }
  const scale = Math.sqrt(-kappa)
  const minimum = view === 'carroll' ? transform.bounds.minX : transform.bounds.minY
  const maximum = view === 'carroll' ? transform.bounds.maxX : transform.bounds.maxY
  const first = Math.floor((scale * minimum - Math.PI / 2) / Math.PI)
  const last = Math.ceil((scale * maximum - Math.PI / 2) / Math.PI)
  const inset = 1e-6 / scale
  return [[(Math.PI / 2 + first * Math.PI) / scale + inset - parameters.anchor[0], (Math.PI / 2 + last * Math.PI) / scale - inset - parameters.anchor[0]]]
}

const parabolicChartBreaks = (kappa: number, parameters: ParabolicParameters, chart: ParabolicChart, from: number, to: number) => {
  if (chart !== 'beltrami' || kappa >= -1e-10) {
    return []
  }
  const scale = Math.sqrt(-kappa)
  const minimum = Math.min(from, to) + parameters.anchor[0]
  const maximum = Math.max(from, to) + parameters.anchor[0]
  const first = Math.ceil((scale * minimum - Math.PI / 2) / Math.PI)
  const last = Math.floor((scale * maximum - Math.PI / 2) / Math.PI)
  return Array.from({ length: Math.max(0, last - first + 1) }, (_, index) => (Math.PI / 2 + (first + index) * Math.PI) / scale - parameters.anchor[0])
    .filter((time) => time > Math.min(from, to) + 1e-10 && time < Math.max(from, to) - 1e-10)
}

const drawParabolicFunction = (root: SVGElement, kappa: number, parameters: ParabolicParameters, evaluator: (time: number) => number, transform: ViewportTransform, chart: ParabolicChart, view: ParabolicView, className: string, from?: number, to?: number) => {
  const intervals = from === undefined || to === undefined
    ? parabolicChartIntervals(kappa, parameters, transform, chart, view)
    : [[from, to] as [number, number]]
  intervals.forEach(([start, end]) => {
    const cuts = [start, ...parabolicChartBreaks(kappa, parameters, chart, start, end), end]
    for (let section = 1; section < cuts.length; section += 1) {
      const rawStart = cuts[section - 1]!
      const rawEnd = cuts[section]!
      const inset = chart === 'beltrami' && kappa < -1e-10 ? Math.abs(rawEnd - rawStart) * 1e-8 : 0
      const startTime = rawStart + Math.sign(rawEnd - rawStart) * inset
      const endTime = rawEnd - Math.sign(rawEnd - rawStart) * inset
      const points: Vec2[] = []
      for (let index = 0; index <= 720; index += 1) {
        const time = startTime + (endTime - startTime) * index / 720
        const y = evaluator(time)
        const point = Number.isFinite(y) ? parabolicDisplayPoint(kappa, canonicalToRawParabolic([time, y], parameters, kappa), chart, view) : null
        if (point && point.every(Number.isFinite)) {
          points.push(point)
        }
      }
      drawPolyline(root, points, transform, className)
    }
  })
}

const drawParabolicFiber = (root: SVGElement, parameters: ParabolicParameters, time: number, transform: ViewportTransform, kappa: number, chart: ParabolicChart, view: ParabolicView) => {
  if (chart === 'beltrami') {
    const point = parabolicDisplayPoint(kappa, [parameters.anchor[0] + time, 0], chart, view)
    if (point) {
      const ends: [Vec2, Vec2] = view === 'carroll'
        ? [[point[0], transform.bounds.minY], [point[0], transform.bounds.maxY]]
        : [[transform.bounds.minX, point[1]], [transform.bounds.maxX, point[1]]]
      drawPolyline(root, ends, transform, 'parabolic-fiber')
    }
    return
  }
  const x = parameters.anchor[0] + time
  drawPolyline(root, [[x, transform.bounds.minY], [x, transform.bounds.maxY]], transform, 'parabolic-fiber')
}

const drawParabolicBeltramiBoundary = (root: SVGElement, kappa: number, transform: ViewportTransform, chart: ParabolicChart, view: ParabolicView) => {
  if (chart !== 'beltrami' || kappa <= 1e-10) {
    return
  }
  const limit = 1 / Math.sqrt(kappa)
  ;[-limit, limit].forEach((coordinate) => {
    const ends: [Vec2, Vec2] = view === 'carroll'
      ? [[coordinate, transform.bounds.minY], [coordinate, transform.bounds.maxY]]
      : [[transform.bounds.minX, coordinate], [transform.bounds.maxX, coordinate]]
    drawPolyline(root, ends, transform, 'boundary')
  })
}

const drawParabolicBeltramiEndpoints = (root: SVGElement, kappa: number, parameters: ParabolicParameters, evaluator: (time: number) => number, transform: ViewportTransform, chart: ParabolicChart, view: ParabolicView, className: string) => {
  if (chart !== 'beltrami' || kappa <= 1e-10) {
    return
  }
  const scale = Math.sqrt(kappa)
  const limit = 1 / scale
  const inset = limit * 1e-6
  ;[-1, 1].forEach((sign) => {
    const coordinateTime = sign * (limit - inset)
    const time = Math.atanh(scale * coordinateTime) / scale - parameters.anchor[0]
    const point = parabolicDisplayPoint(kappa, canonicalToRawParabolic([time, evaluator(time)], parameters, kappa), chart, view)
    if (point && point.every(Number.isFinite)) {
      drawMarker(root, view === 'carroll' ? [sign * limit, point[1]] : [point[0], sign * limit], transform, className, 4)
    }
  })
}

const drawParabolic = (root: SVGElement, vertices: [Vec2, Vec2, Vec2], transform: ViewportTransform, overlays: Overlays, kappa: number, chart: ParabolicChart, view: ParabolicView) => {
  const state = buildParabolicState(kappa, vertices)
  if (!state.valid || !state.euler || !state.tangent || !state.feet) {
    return
  }
  const { parameters } = state
  const pointForChart = (point: Vec2) => parabolicDisplayPoint(kappa, point, chart, view)
  const drawParabolicMarker = (point: Vec2, className: string, radius: number) => {
    const projected = pointForChart(point)
    if (projected) {
      drawMarker(root, projected, transform, className, radius)
    }
  }
  drawParabolicBeltramiBoundary(root, kappa, transform, chart, view)
  if (overlays.singularBranches) {
    state.fibers.forEach((time) => drawParabolicFiber(root, parameters, time, transform, kappa, chart, view))
  }
  if (overlays.euler) {
    drawParabolicFunction(root, kappa, parameters, (time) => parabolicCycleValue(state.euler!, kappa, time), transform, chart, view, 'cycle-euler')
    drawParabolicBeltramiEndpoints(root, kappa, parameters, (time) => parabolicCycleValue(state.euler!, kappa, time), transform, chart, view, 'euler-ideal-marker')
  }
  if (overlays.tangent) {
    drawParabolicFunction(root, kappa, parameters, (time) => parabolicCycleValue(state.tangent!, kappa, time), transform, chart, view, 'cycle-tangent incenter-construction')
    drawParabolicBeltramiEndpoints(root, kappa, parameters, (time) => parabolicCycleValue(state.tangent!, kappa, time), transform, chart, view, 'tangent-ideal-marker')
  }
  if (overlays.bisectors) {
    state.feet.forEach((foot, index) => {
      const start = [0, parameters.u, parameters.w][index]!
      const source = [[0, 0], [parameters.u, parameters.p], [parameters.w, parameters.q]][index]! as Vec2
      const line = parabolicLineThrough(kappa, source, [foot.T, foot.y])
      if (line) {
        const centerTime = state.pseudomedianCenter?.T
        drawParabolicFunction(root, kappa, parameters, (time) => parabolicSideValue(line, kappa, time), transform, chart, view, indexedClass('bisector', index), Math.min(start, foot.T, centerTime ?? start), Math.max(start, foot.T, centerTime ?? start))
      }
    })
  }
  if (overlays.altitudes) {
    state.secondIntersections.forEach((intersection, index) => {
      if (intersection) {
        const source = [[0, 0], [parameters.u, parameters.p], [parameters.w, parameters.q]][index]! as Vec2
        const line = parabolicLineThrough(kappa, source, [intersection.T, intersection.y])
        if (line) {
          const centerTime = state.pseudoaltitudeCenter?.T
          drawParabolicFunction(root, kappa, parameters, (time) => parabolicSideValue(line, kappa, time), transform, chart, view, indexedClass('altitude', index), Math.min(source[0], intersection.T, centerTime ?? source[0]), Math.max(source[0], intersection.T, centerTime ?? source[0]))
        }
      }
    })
  }
  state.sides.forEach((side, index) => {
    const endpoints: Array<[number, number]> = [[parameters.u, parameters.w], [parameters.w, 0], [0, parameters.u]]
    const [from, to] = endpoints[index]!
    drawParabolicFunction(root, kappa, parameters, (time) => parabolicSideValue(side, kappa, time), transform, chart, view, indexedClass('triangle', index), from, to)
  })
  if (overlays.centers) {
    if (state.pseudomedianCenter) {
      drawParabolicMarker(canonicalToRawParabolic([state.pseudomedianCenter.T, state.pseudomedianCenter.y], parameters, kappa), 'area-bisector-center-marker', 3.5)
    }
    if (state.pseudoaltitudeCenter) {
      drawParabolicMarker(canonicalToRawParabolic([state.pseudoaltitudeCenter.T, state.pseudoaltitudeCenter.y], parameters, kappa), 'altitude-center-marker', 3.5)
    }
  }
  if (overlays.bisectors) {
    state.feet.forEach((foot, index) => drawParabolicMarker(canonicalToRawParabolic([foot.T, foot.y], parameters, kappa), indexedClass('foot-marker', index), 3))
  }
  if (overlays.altitudes) {
    state.secondIntersections.forEach((point, index) => {
      if (point) {
        drawParabolicMarker(canonicalToRawParabolic([point.T, point.y], parameters, kappa), indexedClass('altitude-marker', index), 3)
      }
    })
  }
  if (state.contact) {
    drawParabolicMarker(canonicalToRawParabolic([state.contact.T, state.contact.y], parameters, kappa), 'contact-marker incenter-construction', 4.5)
  }
}

const adsProjectiveTransform = (transform: ViewportTransform): ViewportTransform => ({
  bounds: {
    minX: transform.bounds.minY,
    maxX: transform.bounds.maxY,
    minY: transform.bounds.minX,
    maxY: transform.bounds.maxX
  },
  toScreen: ([x, y]) => transform.toScreen([y, x]),
  toWorld: (point) => {
    const [x, y] = transform.toWorld(point)
    return [y, x]
  }
})

const drawCurved = (root: SVGElement, mode: GeometryMode, vertices2: [Vec2, Vec2, Vec2], transform: ViewportTransform, overlays: Overlays, selectedBranch: number) => {
  const metric = metricForMode(mode)!
  const curvedTransform = mode === 'ads' ? adsProjectiveTransform(transform) : transform
  const lifted = vertices2.map((vertex) => liftForMode(mode, vertex))

  if (!lifted[0] || !lifted[1] || !lifted[2]) {
    return
  }

  const vertices = [lifted[0], lifted[1], lifted[2]] as [Vec3, Vec3, Vec3]
  const domain = curvedDomain(mode, vertices)
  const state = buildCurvedState(metric, vertices)
  const pseudoaltitudeNorm = state.pseudoaltitudeDirection ? bilinear3(metric, state.pseudoaltitudeDirection, state.pseudoaltitudeDirection) : Number.NaN
  const pseudoaltitudeCenter = state.pseudoaltitudeDirection && (mode === 'desitter' || mode === 'ads' || pseudoaltitudeNorm > 1e-10)
    ? mode === 'desitter' || mode === 'ads'
      ? state.pseudoaltitudeDirection
      : scale3(state.pseudoaltitudeDirection, 1 / Math.sqrt(pseudoaltitudeNorm))
    : null

  if (mode === 'hyperbolic') {
    drawBoundary(root, mode, curvedTransform)
  }

  if (domain.valid && overlays.euler && state.eulerNormal) {
    drawCurvedCycle(root, mode, metric, state.eulerNormal, 1, curvedTransform, 'cycle-euler')
  }

  const branch = state.branches[selectedBranch]
  if (domain.valid && domain.kind !== 'mixed' && overlays.tangent && branch?.cycle) {
    drawCurvedCycle(root, mode, metric, branch.cycle.normal, branch.cycle.offset, curvedTransform, tangentCycleClass(selectedBranch))
  }

  if (domain.valid && overlays.bisectors && state.pseudomedianFeet) {
    state.pseudomedianFeet.forEach((foot, index) => {
      curvedGeodesic(mode, metric, vertices[index]!, foot, curvedTransform).forEach((line) => drawPolyline(root, line, curvedTransform, indexedClass('bisector', index)))
    })
  }

  if (domain.valid && overlays.altitudes) {
    state.pseudoaltitudePoints.forEach((point, index) => {
      if (point) {
        curvedGeodesic(mode, metric, vertices[index]!, point, curvedTransform).forEach((line) => drawPolyline(root, line, curvedTransform, indexedClass('altitude', index)))
        if (pseudoaltitudeCenter) {
          curvedGeodesic(mode, metric, point, pseudoaltitudeCenter, curvedTransform).forEach((line) => drawPolyline(root, line, curvedTransform, indexedClass('altitude', index)))
        }
      }
    })
  }

  if (domain.valid && overlays.centers) {
    if (pseudoaltitudeCenter) {
      const projected = projectForMode(mode, pseudoaltitudeCenter)
      if (projected) {
        drawMarker(root, projected, curvedTransform, 'altitude-center-marker', 3.5)
      }
    }
    if (state.pseudomedianDirection) {
      const squaredNorm = bilinear3(metric, state.pseudomedianDirection, state.pseudomedianDirection)
      if (squaredNorm > 1e-10) {
        const projected = projectForMode(mode, scale3(state.pseudomedianDirection, 1 / Math.sqrt(squaredNorm)))
        if (projected) {
          drawMarker(root, projected, curvedTransform, 'area-bisector-center-marker', 3.5)
        }
      }
    }
    const normals: Array<[Vec3 | null | undefined, string]> = [
      [state.eulerNormal, 'ninepoint-center-marker'],
      [branch?.cycle?.normal, tangentCenterClass(selectedBranch)]
    ]
    normals.forEach(([normal, className]) => {
      if (!normal) {
        return
      }
      const squaredNorm = bilinear3(metric, normal, normal)
      if (squaredNorm > 1e-10) {
        const projected = projectForMode(mode, scale3(normal, 1 / Math.sqrt(squaredNorm)))
        if (projected) {
          drawMarker(root, projected, curvedTransform, className, 3)
        }
        return
      }
      if (Math.abs(squaredNorm) <= 1e-10) {
        const projected = projectForMode(mode, normal)
        if (projected) {
          drawMarker(root, projected, curvedTransform, className === 'ninepoint-center-marker' ? 'ninepoint-ideal-marker' : tangentIdealMarkerClass(selectedBranch), 4)
        }
      }
    })
  }

  for (let index = 0; index < 3; index += 1) {
    curvedGeodesic(mode, metric, vertices[(index + 1) % 3]!, vertices[(index + 2) % 3]!, curvedTransform).forEach((line) => drawPolyline(root, line, curvedTransform, indexedClass('triangle', index)))
  }

  if (domain.valid) {
    state.pseudomedianFeet?.forEach((point, index) => {
      const projected = projectedUnitPoint(mode, metric, point)
      if (projected) {
        drawMarker(root, projected, curvedTransform, indexedClass('foot-marker', index), 3)
      }
    })
    state.pseudoaltitudePoints.forEach((point, index) => {
      if (point) {
        const projected = projectedUnitPoint(mode, metric, point)
        if (projected) {
          drawMarker(root, projected, curvedTransform, indexedClass('altitude-marker', index), 3)
        }
      }
    })
    if (branch?.contact?.point && accurateResiduals(branch.contact.residuals)) {
      const projected = projectForMode(mode, branch.contact.point)
      if (projected) {
        drawMarker(root, projected, curvedTransform, branch.contact.kind === 'ideal' ? tangentIdealMarkerClass(selectedBranch, true) : `contact-marker ${tangentConstructionClass(selectedBranch)}`, 4.5)
      }
    }
  }
}

export const renderViewport = (svg: SVGSVGElement, mode: GeometryMode, vertices: [Vec2, Vec2, Vec2], overlays: Overlays, selectedBranch: number, fixedBounds?: Bounds, boostActive = false, centroidTransformActive = false, circumcenterMotionActive = false, parabolicKappa = 0, parabolicChart: ParabolicChart = 'natural', parabolicView: ParabolicView = 'galilei', homothetyProgress = 0.5): ViewportTransform => {
  svg.replaceChildren()
  const transform = createTransform(fixedBounds ?? boundsForMode(mode, vertices), Boolean(fixedBounds))
  const projectiveTransform = mode === 'ads' ? adsProjectiveTransform(transform) : transform
  const root = append(svg, element('g'))

  if (overlays.nullBoundary) {
    if (mode === 'minkowski') {
      drawBoundary(root, mode, transform, 'minkowski-null-boundary')
    }
    if (mode === 'desitter' || mode === 'ads') {
      drawBoundary(root, mode, projectiveTransform)
      drawLorentzianNullGrid(root, projectiveTransform)
    }
  }

  if (overlays.grid) {
    const labels: [string, string] = !isParabolicMode(mode)
      ? ['x', 'y']
      : parabolicView === 'carroll'
        ? parabolicChart === 'beltrami' ? ['x', 'τ'] : ['X', 't']
        : parabolicChart === 'beltrami' ? ['z', 't'] : ['T', 'y']
    drawGrid(root, transform, labels)
  }

  if (mode === 'desitter' && overlays.horizon) {
    drawDeSitterHorizons(root, transform)
  }

  if (isParabolicMode(mode)) {
    drawParabolic(root, vertices, transform, overlays, parabolicKappa, parabolicChart, parabolicView)
  } else if (isCurvedMode(mode)) {
    drawCurved(root, mode, vertices, transform, overlays, selectedBranch)
  } else {
    drawFlat(root, mode, vertices, transform, overlays, selectedBranch, homothetyProgress)
  }

  if (centroidTransformActive) {
    root.querySelectorAll('.area-bisector-center-marker').forEach((marker) => marker.classList.add('is-active'))
  }

  if (circumcenterMotionActive) {
    root.querySelectorAll('.euclidean-circumcenter-marker, .minkowski-circumcenter-marker').forEach((marker) => marker.classList.add('is-active'))
  }

  vertices.forEach((point, index) => {
    const displayed = isParabolicMode(mode) ? parabolicDisplayPoint(parabolicKappa, point, parabolicChart, parabolicView) : point
    if (displayed) {
      drawMarker(root, displayed, transform, `vertex vertex-${index}`, 6)
    }
  })

  if ((mode === 'minkowski' && overlays.euclideanCircumcircle) || (mode === 'euclidean' && overlays.minkowskiCircumcircle)) {
    drawRingMarker(root, [0, 0], transform, `boost-toggle${boostActive ? ' is-active' : ''}`)
  }

  return transform
}
