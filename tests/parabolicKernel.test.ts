import { describe, expect, it } from 'vitest'
import { presets } from '../src/data/presets'
import {
  Ck,
  Pk,
  Sk,
  Tk,
  buildParabolicState,
  canonicalParabolicParameters,
  parabolicChartPoint,
  parabolicDisplayPoint,
  parabolicPointFromChart,
  parabolicPointFromDisplay,
  parabolicArea,
  parabolicCycleValue,
  parabolicDefaultKappa,
  parabolicModeForKappa,
  parabolicSideValue,
  parabolicStripArea
} from '../src/math/parabolicKernel'

const vertices: [[number, number], [number, number], [number, number]] = [[0, 0], [1, 0.7], [1.7, 0.2]]
const kappas = [-0.36, 0, 0.36]

const maxResidual = (state: ReturnType<typeof buildParabolicState>) => Math.max(...Object.values(state.residuals))

describe('parabolic Cayley–Klein kernel', () => {
  it('keeps the generalized trigonometric identities across zero', () => {
    kappas.forEach((kappa) => {
      ;[-1.1, -0.4, 0, 0.7, 1.5].forEach((time) => {
        expect(Ck(kappa, time) ** 2 - kappa * Sk(kappa, time) ** 2).toBeCloseTo(1, 11)
      })
      expect(Ck(kappa, 1.2 - 0.4)).toBeCloseTo(Ck(kappa, 1.2) * Ck(kappa, 0.4) - kappa * Sk(kappa, 1.2) * Sk(kappa, 0.4), 11)
    })
    expect(Pk(1e-9, 0.8)).toBeCloseTo(0.32, 9)
    expect(Pk(-1e-9, 0.8)).toBeCloseTo(0.32, 9)
  })

  it('keeps a raw draggable triangle equivalent to its canonical gauge', () => {
    const raw: [[number, number], [number, number], [number, number]] = [[-0.5, -0.2], [0.5, 0.7 - 0.2 * Ck(0.36, 1)], [1.2, 0.2 - 0.2 * Ck(0.36, 1.7)]]
    const parameters = canonicalParabolicParameters(raw, 0.36)
    expect(parameters.u).toBeCloseTo(1, 12)
    expect(parameters.v).toBeCloseTo(0.7, 12)
    expect(parameters.p).toBeCloseTo(0.7, 12)
    expect(parameters.q).toBeCloseTo(0.2, 12)
  })

  it('maps natural NH coordinates projectively to and from the Beltrami chart', () => {
    kappas.forEach((kappa) => {
      const point: [number, number] = [0.8, -0.45]
      const chart = parabolicChartPoint(kappa, point, 'beltrami')
      expect(chart).not.toBeNull()
      const recovered = parabolicPointFromChart(kappa, chart!, 'beltrami', point[0])
      expect(recovered).not.toBeNull()
      expect(recovered![0]).toBeCloseTo(point[0], 12)
      expect(recovered![1]).toBeCloseTo(point[1], 12)

      const side = { a: 0.25, b: -0.7 }
      const first = parabolicChartPoint(kappa, [0.2, parabolicSideValue(side, kappa, 0.2)], 'beltrami')!
      const second = parabolicChartPoint(kappa, [1.1, parabolicSideValue(side, kappa, 1.1)], 'beltrami')!
      expect((second[1] - first[1]) / (second[0] - first[0])).toBeCloseTo(side.b, 11)
    })
    const wrapped: [number, number] = [3.1, 0.4]
    const wrappedChart = parabolicChartPoint(-0.36, wrapped, 'beltrami')
    expect(wrappedChart).not.toBeNull()
    const wrappedRecovered = parabolicPointFromChart(-0.36, wrappedChart!, 'beltrami', wrapped[0])
    expect(wrappedRecovered).not.toBeNull()
    expect(wrappedRecovered![0]).toBeCloseTo(wrapped[0], 12)
    expect(wrappedRecovered![1]).toBeCloseTo(wrapped[1], 12)
    expect(parabolicChartPoint(-0.36, [Math.PI / 1.2, 0], 'beltrami')).toBeNull()
  })

  it('maps the Carroll dual by exchanging the displayed absolute axes', () => {
    kappas.forEach((kappa) => {
      const point: [number, number] = [0.8, -0.45]
      const galilei = parabolicDisplayPoint(kappa, point, 'beltrami', 'galilei')
      const carroll = parabolicDisplayPoint(kappa, point, 'beltrami', 'carroll')
      expect(galilei).not.toBeNull()
      expect(carroll).not.toBeNull()
      expect(carroll![0]).toBeCloseTo(galilei![1], 12)
      expect(carroll![1]).toBeCloseTo(galilei![0], 12)
      const recovered = parabolicPointFromDisplay(kappa, carroll!, 'beltrami', 'carroll', point[0])
      expect(recovered).not.toBeNull()
      expect(recovered![0]).toBeCloseTo(point[0], 12)
      expect(recovered![1]).toBeCloseTo(point[1], 12)
      expect(parabolicDisplayPoint(kappa, point, 'natural', 'carroll')).toEqual(point)
    })
  })

  it('matches signed area with numerical strip integration and reverses orientation', () => {
    kappas.forEach((kappa) => {
      const parameters = canonicalParabolicParameters(vertices, kappa)
      const exact = parabolicArea(kappa, [0, 0], [parameters.u, parameters.p], [parameters.w, parameters.q])
      expect(parabolicStripArea(kappa, parameters)).toBeCloseTo(exact, 7)
      expect(parabolicArea(kappa, [0, 0], [parameters.w, parameters.q], [parameters.u, parameters.p])).toBeCloseTo(-exact, 12)
    })
  })

  it('constructs the common pseudomedian, Euler, pseudoaltitude, and contact configuration', () => {
    kappas.forEach((kappa) => {
      const state = buildParabolicState(kappa, vertices)
      expect(state.valid).toBe(true)
      expect(state.feet).not.toBeNull()
      expect(state.euler).not.toBeNull()
      expect(state.tangent).not.toBeNull()
      expect(state.contact).not.toBeNull()
      expect(state.secondIntersections.every(Boolean)).toBe(true)
      expect(maxResidual(state)).toBeLessThan(1e-6)
      state.feet!.forEach((foot, index) => {
        expect(foot.vector[0] ** 2 - kappa * foot.vector[2] ** 2).toBeCloseTo(1, 10)
        expect(parabolicCycleValue(state.euler!, kappa, foot.T)).toBeCloseTo(foot.y, 9)
        expect(parabolicSideValue(state.sides[index]!, kappa, foot.T)).toBeCloseTo(foot.y, 9)
      })
      const area = parabolicArea(kappa, [0, 0], [state.parameters.u, state.parameters.p], [state.parameters.w, state.parameters.q])
      const points: [[number, number], [number, number], [number, number]] = [[0, 0], [state.parameters.u, state.parameters.p], [state.parameters.w, state.parameters.q]]
      const pairs: Array<[[number, number], [number, number], [number, number], [number, number]]> = [
        [points[0], points[1], [state.feet![0].T, state.feet![0].y], points[2]],
        [points[1], points[2], [state.feet![1].T, state.feet![1].y], points[0]],
        [points[2], points[0], [state.feet![2].T, state.feet![2].y], points[1]]
      ]
      pairs.forEach(([first, second, foot, third]) => {
        expect(parabolicArea(kappa, first, second, foot)).toBeCloseTo(area / 2, 9)
        expect(parabolicArea(kappa, first, foot, third)).toBeCloseTo(area / 2, 9)
      })
      state.secondIntersections.forEach((point, index) => {
        expect(parabolicCycleValue(state.euler!, kappa, point!.T)).toBeCloseTo(point!.y, 9)
        expect(parabolicSideValue(state.sides[index]!, kappa, point!.T)).toBeCloseTo(point!.y, 9)
        expect(point!.T).toBeCloseTo([0, state.parameters.u, state.parameters.w][index]!, 9)
        expect(Math.abs(point!.T - state.feet![index]!.T)).toBeGreaterThan(1e-4)
      })
    })
  })

  it('keeps the finite tangent branch doubly tangent to every sideline', () => {
    kappas.forEach((kappa) => {
      const state = buildParabolicState(kappa, vertices)
      state.sides.forEach((side) => {
        const a2 = 2 * state.tangent!.A - kappa * (side.a + state.tangent!.c)
        const a1 = 2 * (state.tangent!.b - side.b)
        const a0 = state.tangent!.c - side.a
        const scale = Math.max(1, a1 * a1, Math.abs(4 * a2 * a0))
        expect(Math.abs(a1 * a1 - 4 * a2 * a0) / scale).toBeLessThan(1e-10)
      })
    })
  })

  it('recovers the certified contact coordinates and Galilean coefficient ratio', () => {
    const expected = [-0.36, 0, 0.36].map((kappa, index) => [kappa, [0.8907553323, 0.9, 0.9081051948][index]!, [0.1114925726, 0.0975630252, 0.0864465365][index]!])
    expected.forEach(([kappa, time, y]) => {
      const state = buildParabolicState(kappa!, vertices)
      expect(state.contactTime).toBeCloseTo(time!, 9)
      expect(state.contact!.y).toBeCloseTo(y!, 9)
    })
    const galilei = buildParabolicState(0, vertices)
    expect(galilei.euler!.A / galilei.tangent!.A).toBeCloseTo(-8, 12)
  })

  it('uses the three singular fibers and enforces the principal NH− domain', () => {
    const state = buildParabolicState(-0.36, vertices)
    expect(state.fibers).toEqual([0, 1, 1.7])
    state.fibers.forEach((time) => {
      expect(Ck(-0.36, 0)).toBeCloseTo(1, 12)
      expect(-0.36 * Sk(-0.36, 0)).toBeCloseTo(0, 12)
      expect(time).toBeGreaterThanOrEqual(0)
    })
    expect(buildParabolicState(-0.36, [[0, 0], [3, 0.7], [6, 0.2]]).valid).toBe(false)
  })

  it('converges symmetrically to the exact Galilean configuration', () => {
    const zero = buildParabolicState(0, vertices)
    ;[1e-4, 1e-6].forEach((epsilon) => {
      const negative = buildParabolicState(-epsilon, vertices)
      const positive = buildParabolicState(epsilon, vertices)
      expect((negative.contactTime! + positive.contactTime!) / 2).toBeCloseTo(zero.contactTime!, epsilon === 1e-4 ? 6 : 8)
      expect((negative.euler!.A + positive.euler!.A) / 2).toBeCloseTo(zero.euler!.A, epsilon === 1e-4 ? 6 : 8)
      expect((negative.tangent!.A + positive.tangent!.A) / 2).toBeCloseTo(zero.tangent!.A, epsilon === 1e-4 ? 6 : 8)
    })
  })

  it('maps selector modes to a continuous signed curvature family', () => {
    expect(parabolicModeForKappa(-0.1)).toBe('nhnegative')
    expect(parabolicModeForKappa(0)).toBe('galilei')
    expect(parabolicModeForKappa(0.1)).toBe('nhpositive')
    expect(parabolicDefaultKappa('nhnegative')).toBe(-0.36)
    expect(parabolicDefaultKappa('galilei')).toBe(0)
    expect(parabolicDefaultKappa('nhpositive')).toBe(0.36)
    expect(Tk(0, 0.7)).toBeCloseTo(0.7, 12)
  })

  it('keeps the parabolic defaults in the Beltrami viewport without changing their canonical data', () => {
    ;([['nhnegative-default', -0.36], ['galilei-default', 0], ['nhpositive-default', 0.36]] as Array<[string, number]>).forEach(([id, kappa]) => {
      const preset = presets.find((candidate) => candidate.id === id)!
      const parameters = canonicalParabolicParameters(preset.vertices, kappa)
      expect([parameters.u, parameters.v, parameters.p, parameters.q]).toEqual([1, 0.7, 0.7, 0.2])
      preset.vertices.forEach((point) => {
        const displayed = parabolicChartPoint(kappa, point, 'beltrami')!
        expect(Math.abs(displayed[0])).toBeLessThan(2.2)
        expect(Math.abs(displayed[1])).toBeLessThan(2.2)
      })
    })
  })
})
