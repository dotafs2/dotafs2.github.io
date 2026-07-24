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

@group(0) @binding(0)
var<storage, read_write> positions: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read_write> previous_positions: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> velocities: array<vec2<f32>>;

@group(0) @binding(3)
var<storage, read_write> states: array<vec4<f32>>;

@group(0) @binding(4)
var<uniform> params: WeatherParams;

fn particle_count() -> u32 {
  return u32(params.water.z);
}

fn project_cloud_boundary(position: vec2<f32>) -> vec2<f32> {
  let padding = params.interaction.w;
  return vec2<f32>(
    clamp(position.x, padding, params.viewport.x - padding),
    clamp(position.y, padding, params.viewport.y + padding * 3.0),
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

fn cloud_target(seed: f32, time: f32) -> vec2<f32> {
  let lobe = floor(seed * 3.0);
  let center_x =
    0.24 + lobe * 0.26 +
    sin(time * 0.12 + seed * 19.0) * 0.018;
  let center_y =
    0.17 + fract(seed * 7.13) * 0.105 +
    sin(time * 0.18 + seed * 31.0) * 0.012;
  return params.viewport.xy * vec2<f32>(center_x, center_y);
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
  let time = params.water.w;
  var position = positions[index];
  var velocity = velocities[index];
  var state = states[index];
  var phase = state.x;
  var age = state.y;
  let seed = state.z;
  let active_flag = state.w;
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
        velocity += params.pointer.zw * (0.24 * weighted_falloff);
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
      position = project_cloud_boundary(position + velocity * dt);
    } else {
      position = project_water_boundary(position + velocity * dt);
    }

    positions[index] = position;
    velocities[index] = velocity;
    states[index] = vec4<f32>(phase, age, seed, active_flag);
    return;
  }

  age += dt;
  if (phase > 1.5) {
    velocity.y += params.lifecycle.y * 4.2 * dt;
    velocity.x += sin(time * 1.7 + seed * 27.0) * 7.0 * dt;
    velocity *= pow(0.998, dt * 60.0);
    position += velocity * dt;

    if (position.y >= params.cloud_dynamics.z) {
      phase = -1.0;
      age = 0.0;
      position.y = params.cloud_dynamics.z;
      velocity.y = min(velocity.y, 260.0);
    }

    positions[index] = position;
    velocities[index] = velocity;
    states[index] = vec4<f32>(phase, age, seed, active_flag);
    return;
  }

  let condensation_start = 0.9 + seed * 0.38;
  let condensation_end = 3.15 + seed * 0.62;
  let target_phase = smoothstep(
    condensation_start,
    condensation_end,
    age,
  );
  let phase_response = 1.0 - exp(-params.lifecycle.w * dt);
  phase = mix(phase, target_phase, phase_response);

  let smoke_amount = 1.0 - smoothstep(0.28, 0.78, phase);
  let cloud_anchor = cloud_target(seed, time);
  let condensed = smoothstep(0.24, 0.9, phase);
  let anchor_force =
    (cloud_anchor - position) *
    params.cloud_dynamics.x *
    condensed;
  velocity += anchor_force * dt;
  velocity.y -= params.lifecycle.x * smoke_amount * dt;

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
      velocity += params.pointer.zw * (0.26 * weighted_falloff);
      velocity +=
        direction *
        params.interaction.y *
        0.32 *
        weighted_falloff;

      if (
        (pointer_speed > 360.0 || params.interaction.y > 1.0) &&
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

  velocity *= pow(mix(0.988, 0.982, phase), dt * 60.0);
  let speed = length(velocity);
  if (speed > params.water_grid.w) {
    velocity *= params.water_grid.w / speed;
  }
  position = project_cloud_boundary(position + velocity * dt);

  let rain_start = 6.8 + seed * 2.8;
  if (age > rain_start && phase > 0.86) {
    phase = 2.0;
    velocity.x *= 0.28;
    velocity.y = 38.0 + seed * 34.0;
  }

  positions[index] = position;
  velocities[index] = velocity;
  states[index] = vec4<f32>(phase, age, seed, active_flag);
}
