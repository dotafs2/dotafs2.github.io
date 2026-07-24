@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;

struct BlurParams {
  settings: vec4<f32>,
}

@group(0) @binding(2) var<uniform> blurParams: BlurParams;

@vertex
fn fullscreenVertex(@builtin(vertex_index) vertexIndex: u32)
  -> @builtin(position) vec4<f32> {
  let positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  return vec4<f32>(positions[vertexIndex], 0.0, 1.0);
}

fn sampleGaussian(uv: vec2<f32>, direction: vec2<f32>) -> vec4<f32> {
  let dimensions = vec2<f32>(textureDimensions(sourceTexture));
  let texel =
    direction * blurParams.settings.x / dimensions;
  var color =
    textureSample(sourceTexture, sourceSampler, uv) * 0.227027;
  color += textureSample(
    sourceTexture,
    sourceSampler,
    uv + texel * 1.384615,
  ) * 0.316216;
  color += textureSample(
    sourceTexture,
    sourceSampler,
    uv - texel * 1.384615,
  ) * 0.316216;
  color += textureSample(
    sourceTexture,
    sourceSampler,
    uv + texel * 3.230769,
  ) * 0.070270;
  color += textureSample(
    sourceTexture,
    sourceSampler,
    uv - texel * 3.230769,
  ) * 0.070270;
  return color;
}

@fragment
fn blurHorizontal(@builtin(position) fragmentPosition: vec4<f32>)
  -> @location(0) vec4<f32> {
  let dimensions = vec2<f32>(textureDimensions(sourceTexture));
  let uv = fragmentPosition.xy / dimensions;
  return sampleGaussian(uv, vec2<f32>(1.0, 0.0));
}

@fragment
fn blurVertical(@builtin(position) fragmentPosition: vec4<f32>)
  -> @location(0) vec4<f32> {
  let dimensions = vec2<f32>(textureDimensions(sourceTexture));
  let uv = fragmentPosition.xy / dimensions;
  return sampleGaussian(uv, vec2<f32>(0.0, 1.0));
}
