struct CloudRenderParams {
  display: vec4<f32>,
  simulation: vec4<f32>,
  style: vec4<f32>,
}

@group(0) @binding(0)
var<storage, read> positions: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read> states: array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> params: CloudRenderParams;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) local_position: vec2<f32>,
  @location(1) phase: f32,
  @location(2) visibility: f32,
  @location(3) rain: f32,
  @location(4) water: f32,
}

@vertex
fn cloud_density_vertex(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
  );
  let state = states[instance_index];
  let water = select(0.0, 1.0, state.x < -0.5);
  let rain = select(0.0, 1.0, state.x > 1.5);
  let phase = clamp(state.x, 0.0, 1.0);
  let visibility =
    state.w * select(0.0, 1.0, state.y >= 0.0);
  let airborne_radius = mix(
    params.style.w,
    params.simulation.w,
    smoothstep(0.08, 0.78, phase),
  );
  let radius = mix(
    airborne_radius,
    params.simulation.z,
    water,
  );
  let local_position = corners[vertex_index];
  let particle_scale = mix(
    vec2<f32>(radius),
    vec2<f32>(radius * 0.28, radius * 1.62),
    rain,
  );
  let pixel_position =
    positions[instance_index] +
    local_position * particle_scale;
  let clip_position = vec2<f32>(
    pixel_position.x / params.simulation.x * 2.0 - 1.0,
    1.0 - pixel_position.y / params.simulation.y * 2.0,
  );

  var output: VertexOutput;
  output.position = vec4<f32>(clip_position, 0.0, 1.0);
  output.local_position = local_position;
  output.phase = phase;
  output.visibility = visibility;
  output.rain = rain;
  output.water = water;
  return output;
}

@fragment
fn cloud_density_fragment(
  input: VertexOutput,
) -> @location(0) vec4<f32> {
  let radius_squared =
    dot(input.local_position, input.local_position);
  if (radius_squared >= 1.0 || input.visibility < 0.5) {
    discard;
  }
  let falloff = 1.0 - radius_squared;
  let kernel = falloff * falloff * falloff;
  if (input.water > 0.5) {
    let water_density = kernel * 0.72;
    return vec4<f32>(water_density, 0.0, 0.0, 0.0);
  }
  if (input.rain > 0.5) {
    return vec4<f32>(0.0, 0.0, 0.0, kernel * 0.94);
  }
  let weight =
    kernel *
    mix(0.26, 0.86, smoothstep(0.05, 0.86, input.phase));
  let condensed = weight * input.phase;
  let vapor = weight * (1.0 - input.phase);
  return vec4<f32>(weight, condensed, vapor, 0.0);
}

@fragment
fn unified_particle_fragment(
  input: VertexOutput,
) -> @location(0) vec4<f32> {
  let radius_squared =
    dot(input.local_position, input.local_position);
  if (radius_squared >= 1.0 || input.visibility < 0.5) {
    discard;
  }
  let falloff = 1.0 - radius_squared;
  let alpha = smoothstep(0.0, 0.72, falloff);

  if (input.water > 0.5) {
    let water_color = mix(
      vec3<f32>(0.01, 0.24, 0.48),
      vec3<f32>(0.22, 0.82, 1.0),
      falloff,
    );
    return vec4<f32>(water_color, alpha * 0.86);
  }
  if (input.rain > 0.5) {
    let rain_color = mix(
      vec3<f32>(0.12, 0.54, 0.82),
      vec3<f32>(0.76, 0.96, 1.0),
      falloff,
    );
    return vec4<f32>(rain_color, alpha * 0.92);
  }

  let vapor_color = vec3<f32>(0.2, 0.58, 0.78);
  let cloud_color = vec3<f32>(0.82, 0.94, 0.98);
  let particle_color = mix(
    vapor_color,
    cloud_color,
    smoothstep(0.1, 0.82, input.phase),
  );
  return vec4<f32>(
    particle_color * (0.74 + falloff * 0.26),
    alpha * mix(0.42, 0.88, input.phase),
  );
}
