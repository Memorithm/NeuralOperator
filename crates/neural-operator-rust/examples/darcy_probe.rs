//! Bounded fixed-format fixture runner; not a production dataset loader.
use neural_operator_rust::{Budget, Forcing, Grid, darcy_residual};
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .take(1_048_577)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1_048_576 {
        return Err("fixture exceeds 1 MiB".into());
    }
    let text = std::str::from_utf8(&bytes)?;
    let mut words = text.split_whitespace();
    let rows: usize = words.next().ok_or("missing rows")?.parse()?;
    let columns: usize = words.next().ok_or("missing columns")?.parse()?;
    let length_x: f64 = words.next().ok_or("missing length_x")?.parse()?;
    let length_y: f64 = words.next().ok_or("missing length_y")?.parse()?;
    let points = rows.checked_mul(columns).ok_or("shape overflow")?;
    if points > 4096 {
        return Err("fixture exceeds 4096 points".into());
    }
    let mut read_field = || -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        (0..points)
            .map(|_| Ok(words.next().ok_or("short field")?.parse()?))
            .collect()
    };
    let solution = read_field()?;
    let permeability = read_field()?;
    let forcing = read_field()?;
    if words.next().is_some() {
        return Err("trailing fixture values".into());
    }
    let grid = Grid {
        rows,
        columns,
        length_x,
        length_y,
    };
    let budget = Budget {
        max_grid_points: 4096,
        max_scalar_ops: 30 * 4096 + 4,
        max_output_bytes: 8 * 4096,
    };
    let run = || {
        darcy_residual(
            grid,
            &solution,
            &permeability,
            Forcing::Field(&forcing),
            budget,
        )
    };
    let result = run()?;
    for value in &result.values {
        print!("{value:.17e} ");
    }
    println!();
    for _ in 0..5 {
        std::hint::black_box(run()?);
    }
    let mut times = Vec::with_capacity(31);
    for _ in 0..31 {
        let start = std::time::Instant::now();
        std::hint::black_box(run()?);
        times.push(start.elapsed().as_nanos());
    }
    times.sort_unstable();
    println!(
        "{{\"grid_points\":{},\"interior_points\":{},\"scalar_ops\":{},\"output_bytes\":{},\"warmups\":5,\"repetitions\":31,\"median_ns\":{},\"p95_ns\":{},\"timing_scope\":\"hosted_CPU_API_validation_allocation_and_kernel_not_stable_hardware\"}}",
        result.cost.grid_points,
        result.cost.interior_points,
        result.cost.scalar_ops,
        result.cost.output_bytes,
        times[15],
        times[29]
    );
    Ok(())
}
