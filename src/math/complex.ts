export interface Complex {
  re: number
  im: number
}

export const complex = (re = 0, im = 0): Complex => ({ re, im })

export const cAdd = (left: Complex, right: Complex): Complex => ({
  re: left.re + right.re,
  im: left.im + right.im
})

export const cSubtract = (left: Complex, right: Complex): Complex => ({
  re: left.re - right.re,
  im: left.im - right.im
})

export const cScale = (value: Complex, scalar: number): Complex => ({
  re: value.re * scalar,
  im: value.im * scalar
})

export const cMultiply = (left: Complex, right: Complex): Complex => ({
  re: left.re * right.re - left.im * right.im,
  im: left.re * right.im + left.im * right.re
})

export const cSquare = (value: Complex) => cMultiply(value, value)

export const cConjugate = (value: Complex): Complex => ({ re: value.re, im: -value.im })

export const cAbs = (value: Complex) => Math.hypot(value.re, value.im)

export const cSqrtReal = (value: number): Complex =>
  value >= 0 ? complex(Math.sqrt(value), 0) : complex(0, Math.sqrt(-value))

export interface Realified {
  values: number[] | null
  factor: Complex
  magnitude: number
}

export const realify = (values: readonly Complex[], tolerance = 1e-9): Realified => {
  let selected = complex(0, 0)
  let magnitude = 0

  for (const value of values) {
    const candidate = cAbs(value)

    if (candidate > magnitude) {
      magnitude = candidate
      selected = value
    }
  }

  if (magnitude === 0) {
    return { values: null, factor: complex(1, 0), magnitude: 0 }
  }

  const factor = cScale(cConjugate(selected), 1 / magnitude)
  const rotated = values.map((value) => cMultiply(value, factor))
  const acceptable = rotated.every((value) => Math.abs(value.im) <= tolerance * magnitude)

  return {
    values: acceptable ? rotated.map((value) => value.re) : null,
    factor,
    magnitude
  }
}
