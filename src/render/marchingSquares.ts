import type { Vec2 } from '../math/types'

export interface Bounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

interface Segment {
  start: Vec2
  end: Vec2
}

const edges: Record<number, Array<[number, number]>> = {
  0: [],
  1: [[3, 0]],
  2: [[0, 1]],
  3: [[3, 1]],
  4: [[1, 2]],
  5: [[3, 0], [1, 2]],
  6: [[0, 2]],
  7: [[3, 2]],
  8: [[2, 3]],
  9: [[0, 2]],
  10: [[0, 1], [2, 3]],
  11: [[1, 2]],
  12: [[1, 3]],
  13: [[0, 1]],
  14: [[0, 3]],
  15: []
}

const cellEdges = (state: number, values: number[]): Array<[number, number]> => {
  if (state !== 5 && state !== 10) {
    return edges[state]!
  }

  const center = (values[0]! + values[1]! + values[2]! + values[3]!) / 4
  if (state === 5) {
    return center > 0 ? [[0, 1], [2, 3]] : [[3, 0], [1, 2]]
  }
  return center > 0 ? [[3, 0], [1, 2]] : [[0, 1], [2, 3]]
}

const interpolate = (first: Vec2, second: Vec2, firstValue: number, secondValue: number): Vec2 => {
  const denominator = firstValue - secondValue
  const t = Math.abs(denominator) <= 1e-14 ? 0.5 : firstValue / denominator
  return [first[0] + (second[0] - first[0]) * t, first[1] + (second[1] - first[1]) * t]
}

const pointKey = ([x, y]: Vec2) => `${Math.round(x * 1e8)},${Math.round(y * 1e8)}`

const samePoint = (first: Vec2, second: Vec2) => pointKey(first) === pointKey(second)

const joinSegments = (segments: Segment[]): Vec2[][] => {
  const remaining = segments.slice()
  const lines: Vec2[][] = []

  while (remaining.length > 0) {
    const segment = remaining.pop()!
    const line = [segment.start, segment.end]
    let extended = true

    while (extended) {
      extended = false

      for (let index = remaining.length - 1; index >= 0; index -= 1) {
        const candidate = remaining[index]!

        if (samePoint(candidate.start, line[line.length - 1]!)) {
          line.push(candidate.end)
        } else if (samePoint(candidate.end, line[line.length - 1]!)) {
          line.push(candidate.start)
        } else if (samePoint(candidate.end, line[0]!)) {
          line.unshift(candidate.start)
        } else if (samePoint(candidate.start, line[0]!)) {
          line.unshift(candidate.end)
        } else {
          continue
        }

        remaining.splice(index, 1)
        extended = true
        break
      }
    }

    lines.push(line)
  }

  return lines
}

export const marchingSquares = (sample: (point: Vec2) => number, bounds: Bounds, columns = 180, rows = 140): Vec2[][] => {
  const xStep = (bounds.maxX - bounds.minX) / columns
  const yStep = (bounds.maxY - bounds.minY) / rows
  const values: number[][] = []

  for (let row = 0; row <= rows; row += 1) {
    const sampleRow: number[] = []
    for (let column = 0; column <= columns; column += 1) {
      sampleRow.push(sample([bounds.minX + column * xStep, bounds.minY + row * yStep]))
    }
    values.push(sampleRow)
  }

  const segments: Segment[] = []

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const topLeft: Vec2 = [bounds.minX + column * xStep, bounds.minY + row * yStep]
      const topRight: Vec2 = [topLeft[0] + xStep, topLeft[1]]
      const bottomRight: Vec2 = [topLeft[0] + xStep, topLeft[1] + yStep]
      const bottomLeft: Vec2 = [topLeft[0], topLeft[1] + yStep]
      const cellValues = [
        values[row]![column]!,
        values[row]![column + 1]!,
        values[row + 1]![column + 1]!,
        values[row + 1]![column]!
      ]

      if (cellValues.some((value) => !Number.isFinite(value))) {
        continue
      }

      const cellPoints = [topLeft, topRight, bottomRight, bottomLeft]
      const state =
        (cellValues[0]! >= 0 ? 1 : 0) |
        (cellValues[1]! >= 0 ? 2 : 0) |
        (cellValues[2]! >= 0 ? 4 : 0) |
        (cellValues[3]! >= 0 ? 8 : 0)
      const edgePoints: Vec2[] = [
        interpolate(cellPoints[0]!, cellPoints[1]!, cellValues[0]!, cellValues[1]!),
        interpolate(cellPoints[1]!, cellPoints[2]!, cellValues[1]!, cellValues[2]!),
        interpolate(cellPoints[2]!, cellPoints[3]!, cellValues[2]!, cellValues[3]!),
        interpolate(cellPoints[3]!, cellPoints[0]!, cellValues[3]!, cellValues[0]!)
      ]

      for (const [start, end] of cellEdges(state, cellValues)) {
        segments.push({ start: edgePoints[start]!, end: edgePoints[end]! })
      }
    }
  }

  return joinSegments(segments)
}
