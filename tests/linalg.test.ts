import { describe, expect, it } from 'vitest'
import { complex, realify } from '../src/math/complex'
import { bilinear2, bilinear3, cross3, solve2, solve3 } from '../src/math/linalg'

describe('linear algebra', () => {
  it('keeps both bilinear forms symmetric', () => {
    expect(bilinear2(-1, [2, -3], [-5, 7])).toBe(bilinear2(-1, [-5, 7], [2, -3]))
    expect(bilinear3([1, 1, -1], [2, -3, 5], [-7, 11, 13])).toBe(bilinear3([1, 1, -1], [-7, 11, 13], [2, -3, 5]))
  })

  it('solves non-singular systems', () => {
    expect(solve2([[2, 1], [1, -1]], [5, 1])).toEqual([2, 1])
    expect(solve3([[2, 1, -1], [1, -1, 2], [3, 2, 1]], [3, 1, 8])).toEqual([1, 2, 1])
  })

  it('builds a cross-product null direction', () => {
    const direction = cross3([1, 2, 3], [-2, 1, 4])
    expect(direction[0] + 2 * direction[1] + 3 * direction[2]).toBeCloseTo(0, 12)
    expect(-2 * direction[0] + direction[1] + 4 * direction[2]).toBeCloseTo(0, 12)
  })

  it('realifies a common complex phase', () => {
    const result = realify([complex(0, 2), complex(0, -3), complex(0, 5)])
    expect(result.values).not.toBeNull()
    expect(result.values![0]).toBeCloseTo(2, 12)
    expect(result.values![1]).toBeCloseTo(-3, 12)
    expect(result.values![2]).toBeCloseTo(5, 12)
  })
})
