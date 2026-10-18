# Vectorisation validation

The original package's loop-based `assemble_system` and this package's
vectorised version were compared on all 16 combinations of
`N = 2, 3, 8, 64`, two boundary types and two closures. The largest
matrix-entry difference was exactly zero; the largest right-hand-side
difference was `1.42e-14` (floating-point summation order).

On the validation machine, five warm-up-excluded medians for Neumann/FO
assembly were approximately 53 ms versus 2.7 ms at `N=128`, and 226 ms
versus 9.4 ms at `N=256` (old versus vectorised). These are indicative
machine-specific measurements, not claims about solver speed or universal
complexity. The cost study uses fresh measurements of the vectorised
assembly and direct solve separately.
