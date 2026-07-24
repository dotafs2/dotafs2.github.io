struct Params {
  viewport: vec4<f32>,
  simulation: vec4<f32>,
  grid: vec4<f32>,
  pointer: vec4<f32>,
  interaction: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> previousPositions: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> lambdas: array<f32>;
@group(0) @binding(4) var<storage, read_write> corrections: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> gridHeads: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> gridNext: array<u32>;
@group(0) @binding(7) var<storage, read_write> smoothedVelocities: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> params: Params;

const PI: f32 = 3.141592653589793;

fn particleCount() -> u32 {
  return u32(params.simulation.z);
}

fn smoothingRadius() -> f32 {
  return params.viewport.z;
}

fn poly6Kernel(radiusSquared: f32) -> f32 {
  let h = smoothingRadius();
  let h2 = h * h;
  if (radiusSquared <= 0.0 || radiusSquared >= h2) {
    return 0.0;
  }
  let difference = h2 - radiusSquared;
  let h4 = h2 * h2;
  let h8 = h4 * h4;
  return (4.0 / (PI * h8)) * difference * difference * difference;
}

fn spikyGradient(radius: vec2<f32>) -> vec2<f32> {
  let distance = length(radius);
  let h = smoothingRadius();
  if (distance <= 0.0001 || distance >= h) {
    return vec2<f32>(0.0);
  }
  let h2 = h * h;
  let h5 = h2 * h2 * h;
  let difference = h - distance;
  let scale = (-30.0 / (PI * h5)) * difference * difference / distance;
  return radius * scale;
}

fn cellCoordinates(position: vec2<f32>) -> vec2<i32> {
  let cell = vec2<i32>(floor(position / smoothingRadius()));
  return vec2<i32>(
    clamp(cell.x, 0, i32(params.grid.x) - 1),
    clamp(cell.y, 0, i32(params.grid.y) - 1),
  );
}

fn cellIndex(cell: vec2<i32>) -> u32 {
  return u32(cell.y) * u32(params.grid.x) + u32(cell.x);
}

fn projectBoundary(position: vec2<f32>) -> vec2<f32> {
  let padding = params.interaction.w;
  return clamp(
    position,
    vec2<f32>(padding),
    params.viewport.xy - vec2<f32>(padding),
  );
}

@compute @workgroup_size(128)
fn integrate(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }

  let dt = params.viewport.w;
  var position = positions[index];
  var velocity = velocities[index];
  previousPositions[index] = position;

  velocity.y += params.simulation.w * dt;

  if (params.interaction.z > 0.5) {
    let radius = params.interaction.x;
    let fromPointer = position - params.pointer.xy;
    let distance = length(fromPointer);
    if (distance < radius) {
      let falloff = (1.0 - distance / radius);
      let weightedFalloff = falloff * falloff;
      var direction = vec2<f32>(0.0, -1.0);
      if (distance > 0.001) {
        direction = fromPointer / distance;
      }
      velocity += params.pointer.zw * (0.24 * weightedFalloff);
      velocity += direction * params.interaction.y * weightedFalloff;
    }
  }

  position += velocity * dt;
  positions[index] = projectBoundary(position);
  velocities[index] = velocity;
}

@compute @workgroup_size(128)
fn buildHash(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }
  let cell = cellCoordinates(positions[index]);
  let previousHead = atomicExchange(&gridHeads[cellIndex(cell)], index + 1u);
  gridNext[index] = previousHead;
}

@compute @workgroup_size(128)
fn solveLambda(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }

  let position = positions[index];
  let baseCell = cellCoordinates(position);
  let restDensity = params.simulation.x;
  var density = 0.0;
  var gradientI = vec2<f32>(0.0);
  var sumGradientSquared = 0.0;

  for (var offsetY = -1; offsetY <= 1; offsetY = offsetY + 1) {
    let cellY = baseCell.y + offsetY;
    if (cellY < 0 || cellY >= i32(params.grid.y)) {
      continue;
    }
    for (var offsetX = -1; offsetX <= 1; offsetX = offsetX + 1) {
      let cellX = baseCell.x + offsetX;
      if (cellX < 0 || cellX >= i32(params.grid.x)) {
        continue;
      }

      var node = atomicLoad(
        &gridHeads[cellIndex(vec2<i32>(cellX, cellY))]
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let radius = position - positions[other];
          let radiusSquared = dot(radius, radius);
          if (radiusSquared > 0.0 && radiusSquared < smoothingRadius() * smoothingRadius()) {
            density += poly6Kernel(radiusSquared);
            let gradientJ = -spikyGradient(radius) / restDensity;
            sumGradientSquared += dot(gradientJ, gradientJ);
            gradientI += gradientJ;
          }
        }
        node = gridNext[other];
      }
    }
  }

  sumGradientSquared += dot(gradientI, gradientI);
  let constraint = density / restDensity - 1.0;
  lambdas[index] = -constraint / (sumGradientSquared + params.simulation.y);
}

@compute @workgroup_size(128)
fn solveCorrection(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }

  let position = positions[index];
  let baseCell = cellCoordinates(position);
  let restDensity = params.simulation.x;
  let h = smoothingRadius();
  let referenceKernel = max(poly6Kernel(0.09 * h * h), 0.0000001);
  var correction = vec2<f32>(0.0);

  for (var offsetY = -1; offsetY <= 1; offsetY = offsetY + 1) {
    let cellY = baseCell.y + offsetY;
    if (cellY < 0 || cellY >= i32(params.grid.y)) {
      continue;
    }
    for (var offsetX = -1; offsetX <= 1; offsetX = offsetX + 1) {
      let cellX = baseCell.x + offsetX;
      if (cellX < 0 || cellX >= i32(params.grid.x)) {
        continue;
      }

      var node = atomicLoad(
        &gridHeads[cellIndex(vec2<i32>(cellX, cellY))]
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let radius = position - positions[other];
          let radiusSquared = dot(radius, radius);
          if (radiusSquared > 0.0 && radiusSquared < h * h) {
            let kernelRatio = poly6Kernel(radiusSquared) / referenceKernel;
            let ratioSquared = kernelRatio * kernelRatio;
            let artificialPressure = -0.0018 * ratioSquared * ratioSquared;
            correction += (
              lambdas[index] + lambdas[other] + artificialPressure
            ) * spikyGradient(radius);
          }
        }
        node = gridNext[other];
      }
    }
  }

  correction /= restDensity;
  let correctionLength = length(correction);
  let maximumCorrection = params.interaction.w * 0.36;
  if (correctionLength > maximumCorrection) {
    correction *= maximumCorrection / correctionLength;
  }
  corrections[index] = correction;
}

@compute @workgroup_size(128)
fn applyCorrection(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }
  positions[index] = projectBoundary(positions[index] + corrections[index]);
}

@compute @workgroup_size(128)
fn reconstructVelocity(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }

  let padding = params.interaction.w;
  let position = positions[index];
  var velocity = (position - previousPositions[index]) / params.viewport.w;

  if (position.x <= padding + 0.01 && velocity.x < 0.0) {
    velocity.x *= -0.15;
  }
  if (position.x >= params.viewport.x - padding - 0.01 && velocity.x > 0.0) {
    velocity.x *= -0.15;
  }
  if (position.y <= padding + 0.01 && velocity.y < 0.0) {
    velocity.y *= -0.12;
  }
  if (position.y >= params.viewport.y - padding - 0.01 && velocity.y > 0.0) {
    velocity.y *= -0.12;
  }

  let speed = length(velocity);
  let maximumSpeed = params.grid.w;
  if (speed > maximumSpeed) {
    velocity *= maximumSpeed / speed;
  }
  velocities[index] = velocity * 0.998;
}

@compute @workgroup_size(128)
fn solveViscosity(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }

  let position = positions[index];
  let velocity = velocities[index];
  let baseCell = cellCoordinates(position);
  let restDensity = params.simulation.x;
  var correction = vec2<f32>(0.0);

  for (var offsetY = -1; offsetY <= 1; offsetY = offsetY + 1) {
    let cellY = baseCell.y + offsetY;
    if (cellY < 0 || cellY >= i32(params.grid.y)) {
      continue;
    }
    for (var offsetX = -1; offsetX <= 1; offsetX = offsetX + 1) {
      let cellX = baseCell.x + offsetX;
      if (cellX < 0 || cellX >= i32(params.grid.x)) {
        continue;
      }

      var node = atomicLoad(
        &gridHeads[cellIndex(vec2<i32>(cellX, cellY))]
      );
      while (node != 0u) {
        let other = node - 1u;
        if (other != index) {
          let radius = position - positions[other];
          let radiusSquared = dot(radius, radius);
          if (radiusSquared > 0.0 && radiusSquared < smoothingRadius() * smoothingRadius()) {
            let weight = poly6Kernel(radiusSquared) / restDensity;
            correction += (velocities[other] - velocity) * weight;
          }
        }
        node = gridNext[other];
      }
    }
  }

  smoothedVelocities[index] = velocity + params.grid.z * correction;
}

@compute @workgroup_size(128)
fn applyViscosity(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= particleCount()) {
    return;
  }
  velocities[index] = smoothedVelocities[index];
}
