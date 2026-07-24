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

struct WritableActiveList {
  count: atomic<u32>,
  dispatch_x: u32,
  dispatch_y: u32,
  dispatch_z: u32,
  indices: array<u32>,
}

@group(0) @binding(0)
var<storage, read> states: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> water_active: WritableActiveList;

@group(0) @binding(2)
var<storage, read_write> cloud_active: WritableActiveList;

@group(0) @binding(3)
var<uniform> params: WeatherParams;

@compute @workgroup_size(128)
fn classify(
  @builtin(global_invocation_id) invocation: vec3<u32>,
) {
  let index = invocation.x;
  if (index >= u32(params.water.z)) {
    return;
  }

  if (states[index].x < -0.5) {
    let active_index = atomicAdd(&water_active.count, 1u);
    water_active.indices[active_index] = index;
  } else {
    let active_index = atomicAdd(&cloud_active.count, 1u);
    cloud_active.indices[active_index] = index;
  }
}

@compute @workgroup_size(1)
fn finalize_dispatch() {
  water_active.dispatch_x =
    (atomicLoad(&water_active.count) + 127u) / 128u;
  water_active.dispatch_y = 1u;
  water_active.dispatch_z = 1u;
  cloud_active.dispatch_x =
    (atomicLoad(&cloud_active.count) + 127u) / 128u;
  cloud_active.dispatch_y = 1u;
  cloud_active.dispatch_z = 1u;
}
