//! First bounded NeuralOperator Rust kernel: isotropic conservative Darcy residual.
//!
//! Ports `neural_operator_reference.darcy.darcy_residual_2d`, preserving the
//! harmonic-face five-point stencil and supplied boundary values. This is not
//! a solver, trained neural operator, or a complete SciRust integration.
//! Required Notice: Copyright 2026 Tarek Zekriti (https://github.com/Memorithm/)
#![forbid(unsafe_code)]

/// Hard ceiling independent of caller-supplied admission policy.
pub const MAX_GRID_POINTS: usize = 1_000_000;

/// Row-major grid, including boundary nodes.
#[derive(Clone, Copy, Debug)]
pub struct Grid {
    pub rows: usize,
    pub columns: usize,
    pub length_x: f64,
    pub length_y: f64,
}

/// Admission limits, checked before allocating the output.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub max_grid_points: usize,
    pub max_scalar_ops: usize,
    pub max_output_bytes: usize,
}

/// Only scalar or exact full-grid forcing is admitted; no implicit broadcasting.
#[derive(Clone, Copy, Debug)]
pub enum Forcing<'a> {
    Uniform(f64),
    Field(&'a [f64]),
}

/// Exact structural work/storage accounting, not a wall-clock or RSS estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    pub grid_points: usize,
    pub interior_points: usize,
    /// Four spacing operations plus 30 scalar arithmetic operations per cell.
    /// Validation, indexing, allocation and hardware instructions are excluded.
    pub scalar_ops: usize,
    pub output_bytes: usize,
}

/// Completed output only; failed computations never expose partial results.
#[derive(Debug)]
pub struct Residual {
    pub values: Vec<f64>,
    pub cost: Cost,
}

/// Fail-closed shape, resource and numerical failure classes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Shape,
    Overflow,
    Budget,
    NonFiniteInput,
    NonPositivePermeability,
    InvalidLength,
    InvalidSpacing,
    Allocation,
    NumericalBreakdown,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Darcy residual rejected: {self:?}")
    }
}

impl std::error::Error for Error {}

/// Conservatively compute `-div(k grad u) - f` on interior grid nodes.
///
/// Inputs are borrowed and never mutated. Their allocation belongs to the
/// caller; this API bounds its scan/work/output, not the caller's prior memory.
/// The returned vector has `(rows-2)*(columns-2)` row-major elements. Boundary
/// values are used as supplied, exactly like the independent Python residual.
/// Finite arithmetic breakdown is an error, never a successful NaN/Inf output.
pub fn darcy_residual(
    grid: Grid,
    solution: &[f64],
    permeability: &[f64],
    forcing: Forcing<'_>,
    budget: Budget,
) -> Result<Residual, Error> {
    if grid.rows < 3 || grid.columns < 3 {
        return Err(Error::Shape);
    }
    let points = grid.rows.checked_mul(grid.columns).ok_or(Error::Overflow)?;
    let interior = (grid.rows - 2)
        .checked_mul(grid.columns - 2)
        .ok_or(Error::Overflow)?;
    let scalar_ops = interior
        .checked_mul(30)
        .and_then(|n| n.checked_add(4))
        .ok_or(Error::Overflow)?;
    let output_bytes = interior
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(Error::Overflow)?;
    if points > MAX_GRID_POINTS
        || points > budget.max_grid_points
        || scalar_ops > budget.max_scalar_ops
        || output_bytes > budget.max_output_bytes
    {
        return Err(Error::Budget);
    }
    if solution.len() != points || permeability.len() != points {
        return Err(Error::Shape);
    }
    match forcing {
        Forcing::Uniform(value) if !value.is_finite() => return Err(Error::NonFiniteInput),
        Forcing::Field(values) if values.len() != points => return Err(Error::Shape),
        Forcing::Field(values) if values.iter().any(|x| !x.is_finite()) => {
            return Err(Error::NonFiniteInput);
        }
        _ => {}
    }
    if solution.iter().chain(permeability).any(|x| !x.is_finite()) {
        return Err(Error::NonFiniteInput);
    }
    if permeability.iter().any(|x| *x <= 0.0) {
        return Err(Error::NonPositivePermeability);
    }
    if !grid.length_x.is_finite()
        || grid.length_x <= 0.0
        || !grid.length_y.is_finite()
        || grid.length_y <= 0.0
    {
        return Err(Error::InvalidLength);
    }
    let dx = grid.length_x / (grid.columns - 1) as f64;
    let dy = grid.length_y / (grid.rows - 1) as f64;
    let dx2 = dx * dx;
    let dy2 = dy * dy;
    if !dx2.is_finite() || dx2 <= 0.0 || !dy2.is_finite() || dy2 <= 0.0 {
        return Err(Error::InvalidSpacing);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(interior)
        .map_err(|_| Error::Allocation)?;
    for row in 1..grid.rows - 1 {
        for column in 1..grid.columns - 1 {
            let index = row * grid.columns + column;
            let center = permeability[index];
            let harmonic = |other| 2.0 * center * other / (center + other);
            let east = harmonic(permeability[index + 1]);
            let west = harmonic(permeability[index - 1]);
            let north = harmonic(permeability[index + grid.columns]);
            let south = harmonic(permeability[index - grid.columns]);
            let u = solution[index];
            let horizontal =
                (east * (u - solution[index + 1]) + west * (u - solution[index - 1])) / dx2;
            let vertical = (north * (u - solution[index + grid.columns])
                + south * (u - solution[index - grid.columns]))
                / dy2;
            let rhs = match forcing {
                Forcing::Uniform(value) => value,
                Forcing::Field(values) => values[index],
            };
            let value = horizontal + vertical - rhs;
            if !value.is_finite() {
                return Err(Error::NumericalBreakdown);
            }
            values.push(value);
        }
    }
    Ok(Residual {
        values,
        cost: Cost {
            grid_points: points,
            interior_points: interior,
            scalar_ops,
            output_bytes,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid {
            rows: 3,
            columns: 3,
            length_x: 1.0,
            length_y: 1.0,
        }
    }
    fn budget() -> Budget {
        Budget {
            max_grid_points: 9,
            max_scalar_ops: 34,
            max_output_bytes: 8,
        }
    }
    #[test]
    fn constant_solution_and_exact_cost() {
        let u = [2.0; 9];
        let r = darcy_residual(grid(), &u, &[1.0; 9], Forcing::Uniform(5.0), budget()).unwrap();
        assert_eq!(r.values, [-5.0]);
        assert_eq!(
            r.cost,
            Cost {
                grid_points: 9,
                interior_points: 1,
                scalar_ops: 34,
                output_bytes: 8
            }
        );
        assert_eq!(u, [2.0; 9]);
    }
    #[test]
    fn quadratic_solution_uses_supplied_boundaries() {
        let u = [0.0, 0.25, 1.0, 0.25, 0.5, 1.25, 1.0, 1.25, 2.0];
        assert_eq!(
            darcy_residual(grid(), &u, &[1.0; 9], Forcing::Uniform(0.0), budget())
                .unwrap()
                .values,
            [-4.0]
        );
    }
    #[test]
    fn uniform_and_full_grid_forcing_agree() {
        let a = darcy_residual(
            grid(),
            &[0.0; 9],
            &[1.0; 9],
            Forcing::Uniform(3.0),
            budget(),
        )
        .unwrap();
        let b = darcy_residual(
            grid(),
            &[0.0; 9],
            &[1.0; 9],
            Forcing::Field(&[3.0; 9]),
            budget(),
        )
        .unwrap();
        assert_eq!(a.values, b.values);
    }
    #[test]
    fn budgets_are_checked_before_input_access() {
        for b in [
            Budget {
                max_grid_points: 8,
                ..budget()
            },
            Budget {
                max_scalar_ops: 33,
                ..budget()
            },
            Budget {
                max_output_bytes: 7,
                ..budget()
            },
        ] {
            assert_eq!(
                darcy_residual(grid(), &[], &[], Forcing::Uniform(0.0), b).unwrap_err(),
                Error::Budget
            );
        }
    }
    #[test]
    fn overflow_and_hard_ceiling_fail_closed() {
        let unlimited = Budget {
            max_grid_points: usize::MAX,
            max_scalar_ops: usize::MAX,
            max_output_bytes: usize::MAX,
        };
        assert_eq!(
            darcy_residual(
                Grid {
                    rows: usize::MAX,
                    columns: 3,
                    ..grid()
                },
                &[],
                &[],
                Forcing::Uniform(0.0),
                unlimited
            )
            .unwrap_err(),
            Error::Overflow
        );
        assert_eq!(
            darcy_residual(
                Grid {
                    rows: 1001,
                    columns: 1001,
                    ..grid()
                },
                &[],
                &[],
                Forcing::Uniform(0.0),
                unlimited
            )
            .unwrap_err(),
            Error::Budget
        );
    }
    #[test]
    fn invalid_shapes_fail_closed() {
        assert_eq!(
            darcy_residual(
                Grid { rows: 2, ..grid() },
                &[],
                &[],
                Forcing::Uniform(0.0),
                budget()
            )
            .unwrap_err(),
            Error::Shape
        );
        assert_eq!(
            darcy_residual(
                grid(),
                &[0.0; 8],
                &[1.0; 9],
                Forcing::Uniform(0.0),
                budget()
            )
            .unwrap_err(),
            Error::Shape
        );
        assert_eq!(
            darcy_residual(
                grid(),
                &[0.0; 9],
                &[1.0; 9],
                Forcing::Field(&[1.0; 8]),
                budget()
            )
            .unwrap_err(),
            Error::Shape
        );
    }
    #[test]
    fn nonfinite_inputs_and_nonpositive_permeability_are_rejected() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                darcy_residual(
                    grid(),
                    &[bad; 9],
                    &[1.0; 9],
                    Forcing::Uniform(0.0),
                    budget()
                )
                .unwrap_err(),
                Error::NonFiniteInput
            );
            assert_eq!(
                darcy_residual(
                    grid(),
                    &[0.0; 9],
                    &[bad; 9],
                    Forcing::Uniform(0.0),
                    budget()
                )
                .unwrap_err(),
                Error::NonFiniteInput
            );
            assert_eq!(
                darcy_residual(
                    grid(),
                    &[0.0; 9],
                    &[1.0; 9],
                    Forcing::Uniform(bad),
                    budget()
                )
                .unwrap_err(),
                Error::NonFiniteInput
            );
        }
        for bad in [0.0, -1.0] {
            assert_eq!(
                darcy_residual(
                    grid(),
                    &[0.0; 9],
                    &[bad; 9],
                    Forcing::Uniform(0.0),
                    budget()
                )
                .unwrap_err(),
                Error::NonPositivePermeability
            );
        }
    }
    #[test]
    fn lengths_spacing_and_arithmetic_breakdown_are_explicit() {
        for length in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                darcy_residual(
                    Grid {
                        length_x: length,
                        ..grid()
                    },
                    &[0.0; 9],
                    &[1.0; 9],
                    Forcing::Uniform(0.0),
                    budget()
                )
                .unwrap_err(),
                Error::InvalidLength
            );
        }
        assert_eq!(
            darcy_residual(
                Grid {
                    length_x: f64::MIN_POSITIVE,
                    ..grid()
                },
                &[0.0; 9],
                &[1.0; 9],
                Forcing::Uniform(0.0),
                budget()
            )
            .unwrap_err(),
            Error::InvalidSpacing
        );
        assert_eq!(
            darcy_residual(
                grid(),
                &[0.0; 9],
                &[f64::MAX; 9],
                Forcing::Uniform(0.0),
                budget()
            )
            .unwrap_err(),
            Error::NumericalBreakdown
        );
    }
}
