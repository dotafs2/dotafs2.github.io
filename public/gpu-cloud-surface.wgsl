struct CloudRenderParams {
  display: vec4<f32>,
  simulation: vec4<f32>,
  style: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> params: CloudRenderParams;

@group(0) @binding(1)
var density_texture: texture_2d<f32>;

@group(0) @binding(2)
var density_sampler: sampler;

@vertex
fn cloud_fullscreen_vertex(
  @builtin(vertex_index) vertex_index: u32,
) -> @builtin(position) vec4<f32> {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

fn sample_density(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(
    density_texture,
    density_sampler,
    clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)),
    0.0,
  );
}

@fragment
fn cloud_surface_fragment(
  @builtin(position) fragment_position: vec4<f32>,
) -> @location(0) vec4<f32> {
  let uv = fragment_position.xy / params.display.xy;
  let field = sample_density(uv);
  let water = field.r;
  let condensed = field.g;
  let vapor = field.b;
  let rain = field.a;
  let threshold = params.style.x;
  let feather = params.style.y;
  let vapor_mask = smoothstep(0.018, 0.17, vapor);
  let cloud_mask = smoothstep(
    threshold - feather,
    threshold + feather,
    condensed,
  );
  let coverage = max(vapor_mask * 0.72, cloud_mask);
  let rain_mask = smoothstep(0.025, 0.24, rain);
  let combined_coverage = max(coverage, rain_mask);
  let background =
    vec3<f32>(0.002, 0.006, 0.009) +
    vec3<f32>(0.0, 0.014, 0.024) * (1.0 - uv.y);

  let texel = 1.0 / params.display.zw;
  let left_field = sample_density(
    uv - vec2<f32>(texel.x, 0.0),
  );
  let right_field = sample_density(
    uv + vec2<f32>(texel.x, 0.0),
  );
  let top_field = sample_density(
    uv - vec2<f32>(0.0, texel.y),
  );
  let bottom_field = sample_density(
    uv + vec2<f32>(0.0, texel.y),
  );
  let gradient = vec2<f32>(
    right_field.g - left_field.g,
    bottom_field.g - top_field.g,
  );
  let gradient_length = length(gradient);
  let light_direction = normalize(
    vec2<f32>(-0.62, -0.78),
  );
  let lighting = clamp(
    0.5 +
      dot(
        normalize(gradient + vec2<f32>(0.0001)),
        light_direction,
      ) *
      0.5,
    0.0,
    1.0,
  );

  let inner = smoothstep(
    threshold + feather * 0.5,
    threshold + 0.34,
    condensed,
  );
  let core = smoothstep(
    threshold + 0.42,
    threshold + 0.92,
    condensed,
  );
  let rim =
    cloud_mask *
    (1.0 - smoothstep(
      threshold + 0.03,
      threshold + 0.2,
      condensed,
    ));
  let lit_band =
    smoothstep(0.62, 0.78, lighting) *
    smoothstep(0.025, 0.16, gradient_length);

  let outline_color = vec3<f32>(0.035, 0.12, 0.2);
  let shadow_color = vec3<f32>(0.24, 0.46, 0.64);
  let middle_color = vec3<f32>(0.52, 0.72, 0.84);
  let light_color = vec3<f32>(0.86, 0.95, 0.98);
  var cloud_color = mix(shadow_color, middle_color, inner);
  cloud_color = mix(cloud_color, light_color, max(core, lit_band));
  cloud_color = mix(cloud_color, outline_color, rim * 0.92);

  let vapor_color =
    vec3<f32>(0.16, 0.48, 0.68) *
    (0.72 + lighting * 0.28);
  let phase_mix = smoothstep(
    0.04,
    0.32,
    condensed,
  );
  let body_color = mix(vapor_color, cloud_color, phase_mix);
  let rain_color = mix(
    vec3<f32>(0.18, 0.64, 0.88),
    vec3<f32>(0.72, 0.94, 1.0),
    smoothstep(0.04, 0.32, rain),
  );
  let final_alpha =
    1.0 - (1.0 - coverage) * (1.0 - rain_mask);
  let overlay_color =
    rain_color * rain_mask +
    body_color * coverage * (1.0 - rain_mask);

  let water_threshold = 0.43;
  let water_feather = 0.08;
  let water_mask = smoothstep(
    water_threshold - water_feather,
    water_threshold + water_feather,
    water,
  );
  let water_gradient = vec2<f32>(
    right_field.r - left_field.r,
    bottom_field.r - top_field.r,
  );
  let water_normal = normalize(
    vec3<f32>(-water_gradient * 7.0, 1.0),
  );
  let water_light_direction = normalize(
    vec3<f32>(-0.35, -0.55, 0.82),
  );
  let water_diffuse = max(
    dot(water_normal, water_light_direction),
    0.0,
  );
  let water_specular = pow(water_diffuse, 24.0);
  let depth_color = vec3<f32>(0.006, 0.095, 0.24);
  let shallow_color = vec3<f32>(0.015, 0.49, 0.86);
  let vertical_color = mix(
    shallow_color,
    depth_color,
    smoothstep(0.15, 1.0, uv.y),
  );
  let water_surface_band =
    1.0 -
    smoothstep(
      water_feather * 1.2,
      water_feather * 5.0,
      abs(water - water_threshold),
    );
  let water_edge_glow =
    smoothstep(0.015, 0.14, length(water_gradient)) *
    water_surface_band *
    water_mask;
  let water_color =
    vertical_color *
      (
        0.86 +
        water_diffuse * 0.16 * water_surface_band +
        water_specular * 0.72 * water_surface_band
      ) +
    vec3<f32>(0.18, 0.68, 0.9) * water_edge_glow;
  let base_color = mix(
    background,
    water_color,
    water_mask,
  );
  let final_color =
    overlay_color +
    base_color * (1.0 - final_alpha);
  return vec4<f32>(final_color, 1.0);
}
