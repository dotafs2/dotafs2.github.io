struct CloudParams {
  viewport: vec4<f32>,
  simulation: vec4<f32>,
  grid: vec4<f32>,
  pointer: vec4<f32>,
  interaction: vec4<f32>,
  lifecycle: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read_write> positions: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> previous_positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> velocities: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read_write> states: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read_write> lambdas: array<f32>;

@group(0) @binding(5)
var<storage, read_write> corrections: array<vec2<f32>>;

@group(0) @binding(6)
var<storage, read_write> grid_heads: array<atomic<u32>>;

@group(0) @binding(7)
var<storage, read_write> grid_next: array<u32>;

@group(0) @binding(8)
var<uniform> params: CloudParams;

const PI: f32 = 3.141592653589793;

fn particle_count() -> u32 {
  return u32(params.simulation.z);
}

fn smoothing_radius() -> f32 {
  return params.viewport.z;
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
  let cell = vec2<i32>(
    floor(position / smoothing_radius()),
  );
  return vec2<i32>(
    clamp(cell.x, 0, i32(params.grid.x) - 1),
    clamp(cell.y, 0, i32(params.grid.y) - 1),
  );
}

fn cell_index(cell: vec2<i32>) -> u32 {
  return u32(cell.y) * u32(params.grid.x) + u32(cell.x);
}

fn spawn_position(seed: f32) -> vec2<f32> {
  let time = params.simulation.w;
  let horizontal_wave =
    sin(time * 0.61 + seed * 31.0) * 0.028;
  return vec2<f32>(
    params.viewport.x *
      (0.5 + (seed - 0.5) * 0.31 + horizontal_wave),
    params.viewport.y * 0.57 - params.interaction.w * 0.5,
  );
}

fn condensed_phase(raw_phase: f32) -> f32 {
  return select(
    clamp(raw_phase, 0.0, 1.0),
    0.0,
    raw_phase > 1.2,
  );
}

fn project_cloud_boundary(position: vec2<f32>) -> vec2<f32> {
  let padding = params.interaction.w;
  return vec2<f32>(
    clamp(
      position.x,
      padding,
      params.viewport.x - padding,
    ),
    clamp(
      position.y,
      padding,
      params.viewport.y + padding * 3.0,
    ),
  );
}

fn project_water_boundary(position: vec2<f32>) -> vec2<f32> {
  let padding = params.interaction.w;
  return clamp(
    position,
    vec2<f32>(padding),
    params.viewport.xy - vec2<f32>(padding),
  );
}

@compute @workgroup_size(128)
fn integrate(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }

  let dt = params.viewport.w;
  let time = params.simulation.w;
  var position = positions[index];
  var velocity = velocities[index];
  var state = states[index];
  var phase = state.x;
  var age = state.y;
  let seed = state.z;
  var active_flag = state.w;
  previous_positions[index] = position;

  if (phase < -0.5) {
    velocity.y += params.lifecycle.y * dt;

    if (params.interaction.z > 0.5) {
      let radius = params.interaction.x;
      let from_pointer = position - params.pointer.xy;
      let distance = length(from_pointer);
      if (distance < radius) {
        let falloff = 1.0 - distance / radius;
        let weighted_falloff = falloff * falloff;
        var direction = vec2<f32>(0.0, -1.0);
        if (distance > 0.001) {
          direction = from_pointer / distance;
        }
        velocity +=
          params.pointer.zw *
          (0.24 * weighted_falloff);
        velocity +=
          direction *
          params.interaction.y *
          weighted_falloff;
      }
    }

    let emission_tick = floor(time * 2.0);
    let emission_random = fract(
      sin(seed * 91.37 + emission_tick * 17.23) *
      43758.5453,
    );
    let should_evaporate =
      params.lifecycle.z > 0.5 &&
      position.y < params.viewport.y * 0.64 &&
      emission_random < 0.045;
    if (should_evaporate) {
      phase = 0.0;
      age = 0.0;
      velocity.y = -118.0 - seed * 62.0;
      velocity.x += (seed - 0.5) * 54.0;
      position = project_cloud_boundary(
        position + velocity * dt,
      );
    } else {
      position = project_water_boundary(
        position + velocity * dt,
      );
    }

    positions[index] = position;
    velocities[index] = velocity;
    states[index] = vec4<f32>(
      phase,
      age,
      seed,
      active_flag,
    );
    return;
  }

  if (active_flag < 0.5) {
    if (params.lifecycle.z < 0.5) {
      velocities[index] = vec2<f32>(0.0);
      lambdas[index] = 0.0;
      corrections[index] = vec2<f32>(0.0);
      return;
    }
    active_flag = 1.0;
    age = -0.25 - seed * 1.55;
    phase = 0.0;
    position = spawn_position(seed);
    velocity = vec2<f32>(0.0);
    previous_positions[index] = position;
  }

  age += dt;
  if (age < 0.0) {
    position = spawn_position(seed);
    velocity = vec2<f32>(0.0);
    positions[index] = position;
    previous_positions[index] = position;
    velocities[index] = velocity;
    states[index] = vec4<f32>(
      0.0,
      age,
      seed,
      active_flag,
    );
    return;
  }

  if (phase > 1.5) {
    velocity.y += params.lifecycle.y * 4.2 * dt;
    velocity.x +=
      sin(time * 1.7 + seed * 27.0) * 7.0 * dt;
    velocity *= pow(0.998, dt * 60.0);
    position += velocity * dt;

    if (position.y >= params.viewport.y * 0.575) {
      active_flag = 1.0;
      age = 0.0;
      phase = -1.0;
      position.y = params.viewport.y * 0.575;
      velocity.y = min(velocity.y, 260.0);
    }

    positions[index] = position;
    velocities[index] = velocity;
    states[index] = vec4<f32>(
      phase,
      age,
      seed,
      active_flag,
    );
    return;
  }

  let condensation_start = 0.9 + seed * 0.38;
  let condensation_end = 3.15 + seed * 0.62;
  let target_phase = smoothstep(
    condensation_start,
    condensation_end,
    age,
  );
  let phase_response =
    1.0 - exp(-params.lifecycle.w * dt);
  phase = mix(phase, target_phase, phase_response);

  let smoke_amount =
    1.0 - smoothstep(0.28, 0.78, phase);
  let lift = -params.lifecycle.x * smoke_amount;
  let cloud_target_y =
    params.viewport.y * (0.17 + seed * 0.13);
  let cloud_anchor =
    (cloud_target_y - position.y) *
    smoothstep(0.28, 0.92, phase) *
    0.72;
  velocity.y += (lift + cloud_anchor) * dt;

  let curl_angle =
    time * (0.72 + seed * 0.31) +
    position.x * 0.011 -
    position.y * 0.008 +
    seed * 13.0;
  let curl_force = vec2<f32>(
    sin(curl_angle * 1.17),
    cos(curl_angle * 0.83) * 0.34,
  ) * (18.0 + 24.0 * smoke_amount);
  velocity += curl_force * dt;

  if (params.interaction.z > 0.5) {
    let radius = params.interaction.x;
    let from_pointer = position - params.pointer.xy;
    let distance = length(from_pointer);
    if (distance < radius) {
      let falloff = 1.0 - distance / radius;
      let weighted_falloff = falloff * falloff;
      var direction = vec2<f32>(0.0, -1.0);
      if (distance > 0.001) {
        direction = from_pointer / distance;
      }
      let pointer_speed = length(params.pointer.zw);
      velocity +=
        params.pointer.zw *
        (0.26 * weighted_falloff);
      velocity +=
        direction *
        params.interaction.y *
        0.32 *
        weighted_falloff;

      if (
        (pointer_speed > 360.0 ||
          params.interaction.y > 1.0) &&
        falloff > 0.14
      ) {
        age = min(age, 0.42 + seed * 0.48);
        phase = min(phase, 0.08);
        velocity.y -=
          (42.0 + min(pointer_speed * 0.04, 92.0)) *
          weighted_falloff;
      }
    }
  }

  let drag = mix(0.988, 0.982, phase);
  velocity *= pow(drag, dt * 60.0);
  let speed = length(velocity);
  if (speed > params.grid.w) {
    velocity *= params.grid.w / speed;
  }

  position += velocity * dt;
  position = project_cloud_boundary(position);

  let rain_start = 6.2 + seed * 2.35;
  if (age > rain_start && phase > 0.86) {
    phase = 2.0;
    velocity.x *= 0.28;
    velocity.y = 38.0 + seed * 34.0;
  }

  positions[index] = position;
  velocities[index] = velocity;
  states[index] = vec4<f32>(
    phase,
    age,
    seed,
    active_flag,
  );
}

@compute @workgroup_size(128)
fn build_hash(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }
  let state = states[index];
  if (
    state.w < 0.5 ||
    state.y < 0.0 ||
    state.x > 1.2
  ) {
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
fn solve_lambda(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }

  let state = states[index];
  let is_water = state.x < -0.5;
  let phase = select(
    condensed_phase(state.x),
    1.0,
    is_water,
  );
  if (
    state.w < 0.5 ||
    state.y < 0.0 ||
    phase < 0.08
  ) {
    lambdas[index] = 0.0;
    return;
  }

  let position = positions[index];
  let base_cell = cell_coordinates(position);
  let rest_density = params.simulation.x;
  var density = 0.0;
  var gradient_i = vec2<f32>(0.0);
  var sum_gradient_squared = 0.0;

  for (
    var offset_y = -1;
    offset_y <= 1;
    offset_y = offset_y + 1
  ) {
    let cell_y = base_cell.y + offset_y;
    if (cell_y < 0 || cell_y >= i32(params.grid.y)) {
      continue;
    }
    for (
      var offset_x = -1;
      offset_x <= 1;
      offset_x = offset_x + 1
    ) {
      let cell_x = base_cell.x + offset_x;
      if (cell_x < 0 || cell_x >= i32(params.grid.x)) {
        continue;
      }

      var node = atomicLoad(
        &grid_heads[
          cell_index(vec2<i32>(cell_x, cell_y))
        ],
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let other_state = states[other];
          let other_phase = select(
            condensed_phase(other_state.x),
            select(0.0, 1.0, other_state.x < -0.5),
            is_water,
          );
          let phase_weight =
            phase *
            other_phase *
            other_state.w *
            select(0.0, 1.0, other_state.y >= 0.0);
          let radius = position - positions[other];
          let radius_squared = dot(radius, radius);
          if (
            phase_weight > 0.0 &&
            radius_squared > 0.0 &&
            radius_squared <
              smoothing_radius() * smoothing_radius()
          ) {
            density +=
              poly6_kernel(radius_squared) *
              phase_weight;
            let gradient_j =
              -spiky_gradient(radius) *
              phase_weight /
              rest_density;
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
  let compression = max(
    density / rest_density - 1.0,
    0.0,
  );
  let constraint = select(
    compression,
    density / rest_density - 1.0,
    is_water,
  );
  lambdas[index] =
    -constraint /
    (sum_gradient_squared + params.simulation.y);
}

@compute @workgroup_size(128)
fn solve_correction(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }

  let state = states[index];
  let is_water = state.x < -0.5;
  let phase = select(
    condensed_phase(state.x),
    1.0,
    is_water,
  );
  if (
    state.w < 0.5 ||
    state.y < 0.0 ||
    phase < 0.06
  ) {
    corrections[index] = vec2<f32>(0.0);
    return;
  }

  let position = positions[index];
  let base_cell = cell_coordinates(position);
  let rest_density = params.simulation.x;
  let h = smoothing_radius();
  let reference_kernel = max(
    poly6_kernel(0.09 * h * h),
    0.0000001,
  );
  var pressure_correction = vec2<f32>(0.0);
  var cohesion_direction = vec2<f32>(0.0);
  var cohesion_weight = 0.0;

  for (
    var offset_y = -1;
    offset_y <= 1;
    offset_y = offset_y + 1
  ) {
    let cell_y = base_cell.y + offset_y;
    if (cell_y < 0 || cell_y >= i32(params.grid.y)) {
      continue;
    }
    for (
      var offset_x = -1;
      offset_x <= 1;
      offset_x = offset_x + 1
    ) {
      let cell_x = base_cell.x + offset_x;
      if (cell_x < 0 || cell_x >= i32(params.grid.x)) {
        continue;
      }

      var node = atomicLoad(
        &grid_heads[
          cell_index(vec2<i32>(cell_x, cell_y))
        ],
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let other_state = states[other];
          let other_phase = select(
            condensed_phase(other_state.x),
            select(0.0, 1.0, other_state.x < -0.5),
            is_water,
          );
          let phase_weight =
            phase *
            other_phase *
            other_state.w *
            select(0.0, 1.0, other_state.y >= 0.0);
          let radius = position - positions[other];
          let distance = length(radius);
          if (
            phase_weight > 0.0 &&
            distance > 0.0001 &&
            distance < h
          ) {
            let kernel_ratio =
              poly6_kernel(distance * distance) /
              reference_kernel;
            let ratio_squared =
              kernel_ratio * kernel_ratio;
            let artificial_pressure = select(
              0.0,
              -0.0018 * ratio_squared * ratio_squared,
              is_water,
            );
            pressure_correction +=
              (
                lambdas[index] +
                lambdas[other] +
                artificial_pressure
              ) *
              spiky_gradient(radius) *
              phase_weight;
            if (!is_water) {
              let attraction =
                (1.0 - distance / h) *
                phase_weight;
              cohesion_direction +=
                (-radius / distance) * attraction;
              cohesion_weight += attraction;
            }
          }
        }
        node = grid_next[other];
      }
    }
  }

  pressure_correction /= rest_density;
  var correction = pressure_correction;
  if (cohesion_weight > 0.0001) {
    correction +=
      cohesion_direction /
      cohesion_weight *
      params.grid.z *
      smoothstep(0.1, 0.82, phase);
  }

  let correction_length = length(correction);
  let maximum_correction = params.interaction.w * 0.34;
  if (correction_length > maximum_correction) {
    correction *=
      maximum_correction / correction_length;
  }
  corrections[index] = correction;
}

@compute @workgroup_size(128)
fn apply_correction(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }
  let corrected_position =
    positions[index] + corrections[index];
  positions[index] = select(
    project_cloud_boundary(corrected_position),
    project_water_boundary(corrected_position),
    states[index].x < -0.5,
  );
}

@compute @workgroup_size(128)
fn reconstruct_velocity(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }

  let state = states[index];
  if (state.w < 0.5 || state.y < 0.0) {
    velocities[index] = vec2<f32>(0.0);
    return;
  }

  let padding = params.interaction.w;
  let position = positions[index];
  var velocity =
    (position - previous_positions[index]) /
    params.viewport.w;

  if (
    position.x <= padding + 0.01 &&
    velocity.x < 0.0
  ) {
    velocity.x *= -0.18;
  }
  if (
    position.x >=
      params.viewport.x - padding - 0.01 &&
    velocity.x > 0.0
  ) {
    velocity.x *= -0.18;
  }
  if (
    position.y <= padding + 0.01 &&
    velocity.y < 0.0
  ) {
    velocity.y *= -0.12;
  }
  if (
    state.x < -0.5 &&
    position.y >= params.viewport.y - padding - 0.01 &&
    velocity.y > 0.0
  ) {
    velocity.y *= -0.12;
  }

  let speed = length(velocity);
  if (speed > params.grid.w) {
    velocity *= params.grid.w / speed;
  }
  let damping = select(
    mix(0.991, 0.997, clamp(state.x, 0.0, 1.0)),
    0.998,
    state.x < -0.5,
  );
  velocities[index] = velocity * damping;
}

@compute @workgroup_size(128)
fn solve_viscosity(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }

  let state = states[index];
  let velocity = velocities[index];
  if (state.x >= -0.5) {
    corrections[index] = velocity;
    return;
  }

  let position = positions[index];
  let base_cell = cell_coordinates(position);
  let rest_density = params.simulation.x;
  var viscosity_correction = vec2<f32>(0.0);

  for (
    var offset_y = -1;
    offset_y <= 1;
    offset_y = offset_y + 1
  ) {
    let cell_y = base_cell.y + offset_y;
    if (cell_y < 0 || cell_y >= i32(params.grid.y)) {
      continue;
    }
    for (
      var offset_x = -1;
      offset_x <= 1;
      offset_x = offset_x + 1
    ) {
      let cell_x = base_cell.x + offset_x;
      if (cell_x < 0 || cell_x >= i32(params.grid.x)) {
        continue;
      }

      var node = atomicLoad(
        &grid_heads[
          cell_index(vec2<i32>(cell_x, cell_y))
        ],
      );
      while (node != 0u) {
        let other = node - 1u;
        if (
          other != index &&
          states[other].x < -0.5
        ) {
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
  let index = invocation.x;
  if (index >= particle_count()) {
    return;
  }
  velocities[index] = corrections[index];
}
