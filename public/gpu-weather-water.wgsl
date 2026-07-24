struct WeatherParams {
  viewport: vec4<f32>,
  water: vec4<f32>,
  water_grid: vec4<f32>,
  pointer: vec4<f32>,
  interaction: vec4<f32>,
  lifecycle: vec4<f32>,
  cloud: vec4<f32>,
  cloud_dynamics: vec4<f32>,
}

struct ActiveList {
  count: u32,
  dispatch_x: u32,
  dispatch_y: u32,
  dispatch_z: u32,
  indices: array<u32>,
}

@group(0) @binding(0)
var<storage, read_write> positions: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read> previous_positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> velocities: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read_write> lambdas: array<f32>;

@group(0) @binding(4)
var<storage, read_write> corrections: array<vec2<f32>>;

@group(0) @binding(5)
var<storage, read_write> grid_heads: array<atomic<u32>>;

@group(0) @binding(6)
var<storage, read_write> grid_next: array<u32>;

@group(0) @binding(7)
var<storage, read> active_list: ActiveList;

@group(0) @binding(8)
var<uniform> params: WeatherParams;

const PI: f32 = 3.141592653589793;

fn smoothing_radius() -> f32 {
  return params.viewport.z;
}

fn master_index(local_index: u32) -> u32 {
  return active_list.indices[local_index];
}

fn poly6_kernel(radius_squared: f32) -> f32 {
  let h = smoothing_radius();
  let h2 = h * h;
  if (radius_squared <= 0.0 || radius_squared >= h2) {
    return 0.0;
  }
  let difference = h2 - radius_squared;
  let h4 = h2 * h2;
  let h8 = h4 * h4;
  return
    (4.0 / (PI * h8)) *
    difference *
    difference *
    difference;
}

fn spiky_gradient(radius: vec2<f32>) -> vec2<f32> {
  let distance = length(radius);
  let h = smoothing_radius();
  if (distance <= 0.0001 || distance >= h) {
    return vec2<f32>(0.0);
  }
  let h2 = h * h;
  let h5 = h2 * h2 * h;
  let difference = h - distance;
  let scale =
    (-30.0 / (PI * h5)) *
    difference *
    difference /
    distance;
  return radius * scale;
}

fn cell_coordinates(position: vec2<f32>) -> vec2<i32> {
  let cell = vec2<i32>(floor(position / smoothing_radius()));
  return vec2<i32>(
    clamp(cell.x, 0, i32(params.water_grid.x) - 1),
    clamp(cell.y, 0, i32(params.water_grid.y) - 1),
  );
}

fn cell_index(cell: vec2<i32>) -> u32 {
  return
    u32(cell.y) * u32(params.water_grid.x) +
    u32(cell.x);
}

fn project_boundary(position: vec2<f32>) -> vec2<f32> {
  let padding = params.interaction.w;
  return clamp(
    position,
    vec2<f32>(padding),
    params.viewport.xy - vec2<f32>(padding),
  );
}

@compute @workgroup_size(128)
fn build_hash(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  let cell = cell_coordinates(positions[index]);
  let previous_head = atomicExchange(
    &grid_heads[cell_index(cell)],
    index + 1u,
  );
  grid_next[index] = previous_head;
}

@compute @workgroup_size(128)
fn solve_lambda(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  let position = positions[index];
  let base_cell = cell_coordinates(position);
  let rest_density = params.water.x;
  var density = 0.0;
  var gradient_i = vec2<f32>(0.0);
  var sum_gradient_squared = 0.0;

  for (var offset_y = -1; offset_y <= 1; offset_y += 1) {
    let cell_y = base_cell.y + offset_y;
    if (cell_y < 0 || cell_y >= i32(params.water_grid.y)) {
      continue;
    }
    for (var offset_x = -1; offset_x <= 1; offset_x += 1) {
      let cell_x = base_cell.x + offset_x;
      if (cell_x < 0 || cell_x >= i32(params.water_grid.x)) {
        continue;
      }
      var node = atomicLoad(
        &grid_heads[cell_index(vec2<i32>(cell_x, cell_y))],
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let radius = position - positions[other];
          let radius_squared = dot(radius, radius);
          if (
            radius_squared > 0.0 &&
            radius_squared <
              smoothing_radius() * smoothing_radius()
          ) {
            density += poly6_kernel(radius_squared);
            let gradient_j =
              -spiky_gradient(radius) / rest_density;
            sum_gradient_squared +=
              dot(gradient_j, gradient_j);
            gradient_i += gradient_j;
          }
        }
        node = grid_next[other];
      }
    }
  }

  sum_gradient_squared += dot(gradient_i, gradient_i);
  let constraint = density / rest_density - 1.0;
  lambdas[index] =
    -constraint /
    (sum_gradient_squared + params.water.y);
}

@compute @workgroup_size(128)
fn solve_correction(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  let position = positions[index];
  let base_cell = cell_coordinates(position);
  let rest_density = params.water.x;
  let h = smoothing_radius();
  let reference_kernel = max(
    poly6_kernel(0.09 * h * h),
    0.0000001,
  );
  var correction = vec2<f32>(0.0);

  for (var offset_y = -1; offset_y <= 1; offset_y += 1) {
    let cell_y = base_cell.y + offset_y;
    if (cell_y < 0 || cell_y >= i32(params.water_grid.y)) {
      continue;
    }
    for (var offset_x = -1; offset_x <= 1; offset_x += 1) {
      let cell_x = base_cell.x + offset_x;
      if (cell_x < 0 || cell_x >= i32(params.water_grid.x)) {
        continue;
      }
      var node = atomicLoad(
        &grid_heads[cell_index(vec2<i32>(cell_x, cell_y))],
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let radius = position - positions[other];
          let distance = length(radius);
          if (distance > 0.0001 && distance < h) {
            let kernel_ratio =
              poly6_kernel(distance * distance) /
              reference_kernel;
            let ratio_squared = kernel_ratio * kernel_ratio;
            let artificial_pressure =
              -0.0018 * ratio_squared * ratio_squared;
            correction +=
              (
                lambdas[index] +
                lambdas[other] +
                artificial_pressure
              ) *
              spiky_gradient(radius);
          }
        }
        node = grid_next[other];
      }
    }
  }

  correction /= rest_density;
  let correction_length = length(correction);
  let maximum_correction = params.interaction.w * 0.34;
  if (correction_length > maximum_correction) {
    correction *= maximum_correction / correction_length;
  }
  corrections[index] = correction;
}

@compute @workgroup_size(128)
fn apply_correction(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  positions[index] =
    project_boundary(positions[index] + corrections[index]);
}

@compute @workgroup_size(128)
fn reconstruct_velocity(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  let padding = params.interaction.w;
  let position = positions[index];
  var velocity =
    (position - previous_positions[index]) / params.viewport.w;

  if (position.x <= padding + 0.01 && velocity.x < 0.0) {
    velocity.x *= -0.18;
  }
  if (
    position.x >= params.viewport.x - padding - 0.01 &&
    velocity.x > 0.0
  ) {
    velocity.x *= -0.18;
  }
  if (position.y <= padding + 0.01 && velocity.y < 0.0) {
    velocity.y *= -0.12;
  }
  if (
    position.y >= params.viewport.y - padding - 0.01 &&
    velocity.y > 0.0
  ) {
    velocity.y *= -0.12;
  }
  let speed = length(velocity);
  if (speed > params.water_grid.w) {
    velocity *= params.water_grid.w / speed;
  }
  velocities[index] = velocity * 0.998;
}

@compute @workgroup_size(128)
fn solve_viscosity(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  let velocity = velocities[index];
  let position = positions[index];
  let base_cell = cell_coordinates(position);
  let rest_density = params.water.x;
  var viscosity_correction = vec2<f32>(0.0);

  for (var offset_y = -1; offset_y <= 1; offset_y += 1) {
    let cell_y = base_cell.y + offset_y;
    if (cell_y < 0 || cell_y >= i32(params.water_grid.y)) {
      continue;
    }
    for (var offset_x = -1; offset_x <= 1; offset_x += 1) {
      let cell_x = base_cell.x + offset_x;
      if (cell_x < 0 || cell_x >= i32(params.water_grid.x)) {
        continue;
      }
      var node = atomicLoad(
        &grid_heads[cell_index(vec2<i32>(cell_x, cell_y))],
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let radius = position - positions[other];
          let radius_squared = dot(radius, radius);
          if (
            radius_squared > 0.0 &&
            radius_squared <
              smoothing_radius() * smoothing_radius()
          ) {
            let weight =
              poly6_kernel(radius_squared) / rest_density;
            viscosity_correction +=
              (velocities[other] - velocity) * weight;
          }
        }
        node = grid_next[other];
      }
    }
  }
  corrections[index] =
    velocity + 0.06 * viscosity_correction;
}

@compute @workgroup_size(128)
fn apply_viscosity(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  velocities[index] = corrections[index];
}
