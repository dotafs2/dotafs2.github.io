struct RenderParams {
  display: vec4<f32>,
  simulation: vec4<f32>,
  style: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: RenderParams;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) localPosition: vec2<f32>,
}

@vertex
fn densityVertex(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32,
) -> VertexOutput {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
  );
  let localPosition = corners[vertexIndex];
  let pixelPosition =
    positions[instanceIndex] + localPosition * params.simulation.z;
  let clipPosition = vec2<f32>(
    pixelPosition.x / params.simulation.x * 2.0 - 1.0,
    1.0 - pixelPosition.y / params.simulation.y * 2.0,
  );

  var output: VertexOutput;
  output.position = vec4<f32>(clipPosition, 0.0, 1.0);
  output.localPosition = localPosition;
  return output;
}

@fragment
fn densityFragment(input: VertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.localPosition, input.localPosition);
  if (radiusSquared >= 1.0) {
    discard;
  }
  let falloff = 1.0 - radiusSquared;
  let density = falloff * falloff * falloff * 0.72;
  return vec4<f32>(density, density, density, density);
}

@fragment
fn particleFragment(input: VertexOutput) -> @location(0) vec4<f32> {
  let radiusSquared = dot(input.localPosition, input.localPosition);
  if (radiusSquared >= 1.0) {
    discard;
  }
  let falloff = 1.0 - radiusSquared;
  let alpha = smoothstep(0.0, 0.72, falloff);
  let color = mix(
    vec3<f32>(0.01, 0.24, 0.48),
    vec3<f32>(0.22, 0.82, 1.0),
    falloff,
  );
  return vec4<f32>(color, alpha * 0.86);
}
