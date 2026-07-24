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
var<storage, read> positions: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> velocities: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read> states: array<vec4<f32>>;

@group(0) @binding(3)
var<storage, read_write> velocity_targets: array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read_write> grid_heads: array<atomic<u32>>;

@group(0) @binding(5)
var<storage, read_write> grid_next: array<u32>;

@group(0) @binding(6)
var<storage, read> active_list: ActiveList;

@group(0) @binding(7)
var<uniform> params: WeatherParams;

fn master_index(local_index: u32) -> u32 {
  return active_list.indices[local_index];
}

fn cell_coordinates(position: vec2<f32>) -> vec2<i32> {
  let cell = vec2<i32>(floor(position / params.cloud.x));
  return vec2<i32>(
    clamp(cell.x, 0, i32(params.cloud.y) - 1),
    clamp(cell.y, 0, i32(params.cloud.z) - 1),
  );
}

fn cell_index(cell: vec2<i32>) -> u32 {
  return u32(cell.y) * u32(params.cloud.y) + u32(cell.x);
}

fn is_condensed(state: vec4<f32>) -> bool {
  return
    state.w > 0.5 &&
    state.y >= 0.0 &&
    state.x > 0.22 &&
    state.x < 1.2;
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
  if (!is_condensed(states[index])) {
    grid_next[index] = 0u;
    return;
  }
  let cell = cell_coordinates(positions[index]);
  let previous_head = atomicExchange(
    &grid_heads[cell_index(cell)],
    index + 1u,
  );
  grid_next[index] = previous_head;
}

@compute @workgroup_size(128)
fn solve_cloud_velocity(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  let state = states[index];
  let velocity = velocities[index];
  if (!is_condensed(state)) {
    velocity_targets[index] = velocity;
    return;
  }

  let position = positions[index];
  let base_cell = cell_coordinates(position);
  let radius = params.cloud.x;
  var position_sum = vec2<f32>(0.0);
  var velocity_sum = vec2<f32>(0.0);
  var weight_sum = 0.0;
  var neighbor_count = 0u;

  for (var offset_y = -1; offset_y <= 1; offset_y += 1) {
    let cell_y = base_cell.y + offset_y;
    if (cell_y < 0 || cell_y >= i32(params.cloud.z)) {
      continue;
    }
    for (var offset_x = -1; offset_x <= 1; offset_x += 1) {
      let cell_x = base_cell.x + offset_x;
      if (cell_x < 0 || cell_x >= i32(params.cloud.y)) {
        continue;
      }
      var node = atomicLoad(
        &grid_heads[cell_index(vec2<i32>(cell_x, cell_y))],
      );
      while (node != 0u && neighbor_count < 64u) {
        let other = node - 1u;
        if (other != index) {
          let delta = positions[other] - position;
          let distance = length(delta);
          if (distance > 0.0001 && distance < radius) {
            let weight =
              (1.0 - distance / radius) *
              smoothstep(0.18, 0.88, states[other].x);
            position_sum += delta * weight;
            velocity_sum += velocities[other] * weight;
            weight_sum += weight;
            neighbor_count += 1u;
          }
        }
        node = grid_next[other];
      }
    }
  }

  var target_velocity = velocity;
  if (weight_sum > 0.0001) {
    let average_offset = position_sum / weight_sum;
    let average_velocity = velocity_sum / weight_sum;
    let phase_strength = smoothstep(0.2, 0.9, state.x);
    target_velocity +=
      average_offset *
      params.cloud.w *
      phase_strength *
      params.viewport.w;
    target_velocity = mix(
      target_velocity,
      average_velocity,
      params.cloud_dynamics.y * phase_strength,
    );
  }

  let speed = length(target_velocity);
  if (speed > params.water_grid.w) {
    target_velocity *= params.water_grid.w / speed;
  }
  velocity_targets[index] = target_velocity;
}

@compute @workgroup_size(128)
fn apply_cloud_velocity(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let local_index = invocation.x;
  if (local_index >= active_list.count) {
    return;
  }
  let index = master_index(local_index);
  velocities[index] = velocity_targets[index];
}
