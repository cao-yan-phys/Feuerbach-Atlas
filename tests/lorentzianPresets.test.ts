import { describe, expect, it } from 'vitest'
import { presets } from '../src/data/presets'
import { liftForMode, metricForMode } from '../src/geometry/modes'
import { buildCurvedState } from '../src/math/curvedKernel'
import { bilinear3 } from '../src/math/linalg'
import type { Vec3 } from '../src/math/types'

const preset = (id: string) => presets.find((candidate) => candidate.id === id)!

const curvedPreset = (id: string) => {
  const selected = preset(id)
  const metric = metricForMode(selected.mode)!
  const lifted = selected.vertices.map((vertex) => liftForMode(selected.mode, vertex))
  return {
    metric,
    state: buildCurvedState(metric, [lifted[0]!, lifted[1]!, lifted[2]!] as [Vec3, Vec3, Vec3])
  }
}

describe('Lorentzian presets', () => {
  it('keeps all AdS finite-sample contacts finite', () => {
    const { state } = curvedPreset('ads-finite')
    expect(state.branches).toHaveLength(4)
    expect(state.branches.every((branch) => branch.isReal)).toBe(true)
    expect(state.branches.every((branch) => branch.contact?.kind === 'finite')).toBe(true)
    expect(state.branches.every((branch) => Math.max(...Object.values(branch.contact!.residuals)) < 1e-9)).toBe(true)
  })

  it('detects the AdS ideal contact', () => {
    const { metric, state } = curvedPreset('ads-ideal')
    const contact = state.branches[2]!.contact
    expect(contact?.kind).toBe('ideal')
    expect(contact?.point).not.toBeNull()
    expect(Math.abs(bilinear3(metric, contact!.point!, contact!.point!))).toBeLessThan(1e-9)
    expect(Math.max(...Object.values(contact!.residuals))).toBeLessThan(1e-9)
  })

  it('crosses the stored de Sitter center transition', () => {
    const positive = curvedPreset('ds-center-positive').state.branches[2]!.centerNorm!
    const nullValue = curvedPreset('ds-center-null').state.branches[2]!.centerNorm!
    const negative = curvedPreset('ds-center-negative').state.branches[2]!.centerNorm!
    expect(positive).toBeGreaterThan(0)
    expect(Math.abs(nullValue)).toBeLessThan(1e-9)
    expect(negative).toBeLessThan(0)
  })

  it('keeps the displayed Lorentzian pseudomedian concurrence point real', () => {
    ;['ads-publication', 'ads-finite', 'ads-ideal', 'ds-center-positive', 'ds-center-null', 'ds-center-negative'].forEach((id) => {
      const { metric, state } = curvedPreset(id)
      expect(state.pseudomedianDirection).not.toBeNull()
      expect(bilinear3(metric, state.pseudomedianDirection!, state.pseudomedianDirection!)).toBeGreaterThan(1e-10)
      expect(state.pseudoaltitudeDirection).not.toBeNull()
      expect(Math.hypot(...state.pseudoaltitudeDirection!)).toBeGreaterThan(1e-10)
    })
  })
})
