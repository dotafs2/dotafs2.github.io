"use strict";

// 2D browser port of the Kuro PBF solver used by carDamageTest.
// The original density constraint, lambda solve, artificial pressure,
// position correction and XSPH viscosity are preserved in two dimensions.

const FIXED_DT = 1 / 60;
const H = 2.2;
const H2 = H * H;
const POLY6_COEFFICIENT = 4 / (Math.PI * Math.pow(H, 8));
const SPIKY_COEFFICIENT = -30 / (Math.PI * Math.pow(H, 5));
const RELAXATION_EPSILON = 0.001;
const SCORR_K = 0.0018;
const SCORR_N = 4;
const SCORR_DELTA_Q = 0.3;
const XSPH_VISCOSITY = 0.038;
const GRAVITY = 37;
const MAX_SPEED = 72;
const BOUNDARY_PADDING = 0.72;

let viewportWidth = 1;
let viewportHeight = 1;
let domainWidth = 1;
let domainHeight = 1;
let spacingPixels = 16;
let renderRadius = 4;
let particleCount = 0;
let generation = 0;
let reducedMotion = false;
let paused = false;
let initialized = false;
let solverIterations = 3;
let slowFrames = 0;
let lastTick = performance.now();
let accumulator = 0;
let timer = 0;

let positionX = new Float32Array(0);
let positionY = new Float32Array(0);
let previousX = new Float32Array(0);
let previousY = new Float32Array(0);
let velocityX = new Float32Array(0);
let velocityY = new Float32Array(0);
let lambda = new Float32Array(0);
let deltaX = new Float32Array(0);
let deltaY = new Float32Array(0);
let smoothedVelocityX = new Float32Array(0);
let smoothedVelocityY = new Float32Array(0);
let gridHead = new Int32Array(0);
let gridNext = new Int32Array(0);
let gridWidth = 1;
let gridHeight = 1;
let renderBuffers = [];

let pointerSequence = 0;
let lastAppliedPointerSequence = -1;
const pointer = {
  active: false,
  x: 0,
  y: 0,
  vx: 0,
  vy: 0,
  burst: 0,
};

function poly6KernelSquared(radiusSquared) {
  if (radiusSquared <= 0 || radiusSquared >= H2) return 0;
  const difference = H2 - radiusSquared;
  return (
    POLY6_COEFFICIENT *
    difference *
    difference *
    difference
  );
}

function calculateRestDensity() {
  let density = 0;
  const extent = Math.ceil(H);
  for (let y = -extent; y <= extent; y += 1) {
    for (let x = -extent; x <= extent; x += 1) {
      if (x === 0 && y === 0) continue;
      density += poly6KernelSquared(x * x + y * y);
    }
  }
  return Math.max(density, 0.000001);
}

const REST_DENSITY = calculateRestDensity();
const SCORR_REFERENCE = Math.max(
  poly6KernelSquared(Math.pow(SCORR_DELTA_Q * H, 2)),
  0.000001,
);

function spikyGradientScale(distance) {
  if (distance <= 0 || distance >= H) return 0;
  const difference = H - distance;
  return (
    (SPIKY_COEFFICIENT * difference * difference) /
    Math.max(distance, 0.000001)
  );
}

function projectBoundary(index) {
  const maximumX = domainWidth - BOUNDARY_PADDING;
  const maximumY = domainHeight - BOUNDARY_PADDING;
  positionX[index] = Math.max(
    BOUNDARY_PADDING,
    Math.min(maximumX, positionX[index]),
  );
  positionY[index] = Math.max(
    BOUNDARY_PADDING,
    Math.min(maximumY, positionY[index]),
  );
}

function allocateParticles(targetCount) {
  particleCount = targetCount;
  positionX = new Float32Array(particleCount);
  positionY = new Float32Array(particleCount);
  previousX = new Float32Array(particleCount);
  previousY = new Float32Array(particleCount);
  velocityX = new Float32Array(particleCount);
  velocityY = new Float32Array(particleCount);
  lambda = new Float32Array(particleCount);
  deltaX = new Float32Array(particleCount);
  deltaY = new Float32Array(particleCount);
  smoothedVelocityX = new Float32Array(particleCount);
  smoothedVelocityY = new Float32Array(particleCount);
  gridNext = new Int32Array(particleCount);

  gridWidth = Math.max(1, Math.ceil(domainWidth / H));
  gridHeight = Math.max(1, Math.ceil(domainHeight / H));
  gridHead = new Int32Array(gridWidth * gridHeight);

  renderBuffers = [
    new ArrayBuffer(particleCount * 3 * Float32Array.BYTES_PER_ELEMENT),
    new ArrayBuffer(particleCount * 3 * Float32Array.BYTES_PER_ELEMENT),
  ];
}

function initializeParticles() {
  if (viewportWidth <= 1 || viewportHeight <= 1) return;

  let targetCount;
  if (viewportWidth < 680) targetCount = 1250;
  else if (viewportWidth < 1180) targetCount = 2200;
  else if (viewportWidth < 2100) targetCount = 3200;
  else targetCount = 3900;
  if (reducedMotion) targetCount = Math.min(targetCount, 950);

  spacingPixels = Math.sqrt(
    (viewportWidth * viewportHeight * 0.52) / targetCount,
  );
  spacingPixels = Math.max(11, Math.min(23, spacingPixels));
  domainWidth = viewportWidth / spacingPixels;
  domainHeight = viewportHeight / spacingPixels;
  renderRadius = Math.max(2.7, spacingPixels * 0.29);

  const columns = Math.max(1, Math.floor(domainWidth - 2.2));
  const maximumRows = Math.max(1, Math.floor(domainHeight * 0.68));
  const rows = Math.min(maximumRows, Math.ceil(targetCount / columns));
  targetCount = Math.min(targetCount, columns * rows);

  generation += 1;
  solverIterations = reducedMotion ? 2 : 3;
  slowFrames = 0;
  allocateParticles(targetCount);

  let index = 0;
  for (let row = 0; row < rows && index < particleCount; row += 1) {
    for (
      let column = 0;
      column < columns && index < particleCount;
      column += 1
    ) {
      const rowOffset = row % 2 === 0 ? 0 : 0.48;
      const jitterX = (Math.random() - 0.5) * 0.025;
      const jitterY = (Math.random() - 0.5) * 0.025;
      positionX[index] =
        1.1 + column + rowOffset + jitterX;
      positionY[index] =
        domainHeight - 1.08 - row * 0.89 + jitterY;
      previousX[index] = positionX[index];
      previousY[index] = positionY[index];
      index += 1;
    }
  }

  self.postMessage({
    type: "ready",
    count: particleCount,
    generation,
  });
  emitFrame();
}

function cellIndex(x, y) {
  const cellX = Math.max(
    0,
    Math.min(gridWidth - 1, Math.floor(x / H)),
  );
  const cellY = Math.max(
    0,
    Math.min(gridHeight - 1, Math.floor(y / H)),
  );
  return cellY * gridWidth + cellX;
}

function buildSpatialGrid() {
  gridHead.fill(-1);
  for (let index = 0; index < particleCount; index += 1) {
    const cell = cellIndex(positionX[index], positionY[index]);
    gridNext[index] = gridHead[cell];
    gridHead[cell] = index;
  }
}

function solveLambdas() {
  for (let index = 0; index < particleCount; index += 1) {
    const x = positionX[index];
    const y = positionY[index];
    const baseCellX = Math.floor(x / H);
    const baseCellY = Math.floor(y / H);
    let density = 0;
    let sumGradientSquared = 0;
    let gradientIX = 0;
    let gradientIY = 0;

    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const cellY = baseCellY + offsetY;
      if (cellY < 0 || cellY >= gridHeight) continue;
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        const cellX = baseCellX + offsetX;
        if (cellX < 0 || cellX >= gridWidth) continue;
        let other = gridHead[cellY * gridWidth + cellX];
        while (other !== -1) {
          if (other !== index) {
            const differenceX = x - positionX[other];
            const differenceY = y - positionY[other];
            const radiusSquared =
              differenceX * differenceX +
              differenceY * differenceY;
            if (radiusSquared > 0 && radiusSquared < H2) {
              density += poly6KernelSquared(radiusSquared);
              const distance = Math.sqrt(radiusSquared);
              const gradientScale =
                spikyGradientScale(distance);
              const gradientJX =
                (-differenceX * gradientScale) / REST_DENSITY;
              const gradientJY =
                (-differenceY * gradientScale) / REST_DENSITY;
              sumGradientSquared +=
                gradientJX * gradientJX +
                gradientJY * gradientJY;
              gradientIX += gradientJX;
              gradientIY += gradientJY;
            }
          }
          other = gridNext[other];
        }
      }
    }

    sumGradientSquared +=
      gradientIX * gradientIX + gradientIY * gradientIY;
    const constraint = density / REST_DENSITY - 1;
    lambda[index] =
      -constraint /
      (sumGradientSquared + RELAXATION_EPSILON);
  }
}

function solvePositionDeltas() {
  for (let index = 0; index < particleCount; index += 1) {
    const x = positionX[index];
    const y = positionY[index];
    const baseCellX = Math.floor(x / H);
    const baseCellY = Math.floor(y / H);
    let correctionX = 0;
    let correctionY = 0;

    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const cellY = baseCellY + offsetY;
      if (cellY < 0 || cellY >= gridHeight) continue;
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        const cellX = baseCellX + offsetX;
        if (cellX < 0 || cellX >= gridWidth) continue;
        let other = gridHead[cellY * gridWidth + cellX];
        while (other !== -1) {
          if (other !== index) {
            const differenceX = x - positionX[other];
            const differenceY = y - positionY[other];
            const radiusSquared =
              differenceX * differenceX +
              differenceY * differenceY;
            if (radiusSquared > 0 && radiusSquared < H2) {
              const distance = Math.sqrt(radiusSquared);
              const kernelRatio =
                poly6KernelSquared(radiusSquared) /
                SCORR_REFERENCE;
              const ratioSquared = kernelRatio * kernelRatio;
              const artificialPressure =
                -SCORR_K *
                (SCORR_N === 4
                  ? ratioSquared * ratioSquared
                  : Math.pow(kernelRatio, SCORR_N));
              const factor =
                lambda[index] +
                lambda[other] +
                artificialPressure;
              const gradientScale =
                spikyGradientScale(distance);
              correctionX +=
                factor * differenceX * gradientScale;
              correctionY +=
                factor * differenceY * gradientScale;
            }
          }
          other = gridNext[other];
        }
      }
    }

    correctionX /= REST_DENSITY;
    correctionY /= REST_DENSITY;
    const correctionLength = Math.hypot(
      correctionX,
      correctionY,
    );
    if (correctionLength > 0.28) {
      const correctionScale = 0.28 / correctionLength;
      correctionX *= correctionScale;
      correctionY *= correctionScale;
    }
    deltaX[index] = correctionX;
    deltaY[index] = correctionY;
  }

  for (let index = 0; index < particleCount; index += 1) {
    positionX[index] += deltaX[index];
    positionY[index] += deltaY[index];
    projectBoundary(index);
  }
}

function applyPointerImpulse() {
  if (
    !pointer.active ||
    pointerSequence === lastAppliedPointerSequence
  ) {
    return;
  }
  lastAppliedPointerSequence = pointerSequence;

  const pointerX = pointer.x / spacingPixels;
  const pointerY = pointer.y / spacingPixels;
  const pointerVelocityX = pointer.vx / spacingPixels;
  const pointerVelocityY = pointer.vy / spacingPixels;
  const pointerSpeed = Math.hypot(
    pointerVelocityX,
    pointerVelocityY,
  );
  const radius = Math.max(4.2, 105 / spacingPixels);
  const radiusSquared = radius * radius;

  for (let index = 0; index < particleCount; index += 1) {
    const differenceX = positionX[index] - pointerX;
    const differenceY = positionY[index] - pointerY;
    const distanceSquared =
      differenceX * differenceX + differenceY * differenceY;
    if (distanceSquared >= radiusSquared) continue;

    const distance = Math.sqrt(Math.max(distanceSquared, 0.0001));
    const falloff = Math.pow(1 - distance / radius, 2);
    const normalX = differenceX / distance;
    const normalY = differenceY / distance;
    const radialStrength =
      Math.min(28, pointerSpeed * 0.18 + pointer.burst);
    velocityX[index] +=
      (pointerVelocityX * 0.34 + normalX * radialStrength) *
      falloff;
    velocityY[index] +=
      (pointerVelocityY * 0.34 + normalY * radialStrength) *
      falloff;
  }
  pointer.burst = 0;
}

function applyXSPHViscosity() {
  for (let index = 0; index < particleCount; index += 1) {
    const x = positionX[index];
    const y = positionY[index];
    const baseCellX = Math.floor(x / H);
    const baseCellY = Math.floor(y / H);
    let correctionX = 0;
    let correctionY = 0;

    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      const cellY = baseCellY + offsetY;
      if (cellY < 0 || cellY >= gridHeight) continue;
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        const cellX = baseCellX + offsetX;
        if (cellX < 0 || cellX >= gridWidth) continue;
        let other = gridHead[cellY * gridWidth + cellX];
        while (other !== -1) {
          if (other !== index) {
            const differenceX = x - positionX[other];
            const differenceY = y - positionY[other];
            const radiusSquared =
              differenceX * differenceX +
              differenceY * differenceY;
            if (radiusSquared > 0 && radiusSquared < H2) {
              const weight =
                poly6KernelSquared(radiusSquared) /
                REST_DENSITY;
              correctionX +=
                (velocityX[other] - velocityX[index]) *
                weight;
              correctionY +=
                (velocityY[other] - velocityY[index]) *
                weight;
            }
          }
          other = gridNext[other];
        }
      }
    }

    smoothedVelocityX[index] =
      velocityX[index] + XSPH_VISCOSITY * correctionX;
    smoothedVelocityY[index] =
      velocityY[index] + XSPH_VISCOSITY * correctionY;
  }

  for (let index = 0; index < particleCount; index += 1) {
    velocityX[index] = smoothedVelocityX[index];
    velocityY[index] = smoothedVelocityY[index];
  }
}

function simulateStep() {
  applyPointerImpulse();

  for (let index = 0; index < particleCount; index += 1) {
    previousX[index] = positionX[index];
    previousY[index] = positionY[index];
    velocityY[index] += GRAVITY * FIXED_DT;
    positionX[index] += velocityX[index] * FIXED_DT;
    positionY[index] += velocityY[index] * FIXED_DT;
    projectBoundary(index);
  }

  for (
    let iteration = 0;
    iteration < solverIterations;
    iteration += 1
  ) {
    buildSpatialGrid();
    solveLambdas();
    solvePositionDeltas();
  }

  for (let index = 0; index < particleCount; index += 1) {
    let nextVelocityX =
      (positionX[index] - previousX[index]) / FIXED_DT;
    let nextVelocityY =
      (positionY[index] - previousY[index]) / FIXED_DT;

    if (
      positionX[index] <= BOUNDARY_PADDING + 0.001 &&
      nextVelocityX < 0
    ) {
      nextVelocityX *= -0.16;
    }
    if (
      positionX[index] >=
        domainWidth - BOUNDARY_PADDING - 0.001 &&
      nextVelocityX > 0
    ) {
      nextVelocityX *= -0.16;
    }
    if (
      positionY[index] <= BOUNDARY_PADDING + 0.001 &&
      nextVelocityY < 0
    ) {
      nextVelocityY *= -0.12;
    }
    if (
      positionY[index] >=
        domainHeight - BOUNDARY_PADDING - 0.001 &&
      nextVelocityY > 0
    ) {
      nextVelocityY *= -0.12;
    }

    const speed = Math.hypot(
      nextVelocityX,
      nextVelocityY,
    );
    if (speed > MAX_SPEED) {
      const scale = MAX_SPEED / speed;
      nextVelocityX *= scale;
      nextVelocityY *= scale;
    }
    velocityX[index] = nextVelocityX * 0.999;
    velocityY[index] = nextVelocityY * 0.999;
  }

  buildSpatialGrid();
  applyXSPHViscosity();
}

function emitFrame() {
  if (renderBuffers.length === 0 || particleCount === 0) return;
  const buffer = renderBuffers.pop();
  const output = new Float32Array(buffer);
  for (let index = 0; index < particleCount; index += 1) {
    const offset = index * 3;
    output[offset] = positionX[index] * spacingPixels;
    output[offset + 1] = positionY[index] * spacingPixels;
    output[offset + 2] = Math.hypot(
      velocityX[index],
      velocityY[index],
    );
  }
  self.postMessage(
    {
      type: "frame",
      buffer: output.buffer,
      count: particleCount,
      radius: renderRadius,
      generation,
    },
    [output.buffer],
  );
}

function tick() {
  if (!initialized || paused || particleCount === 0) return;
  const start = performance.now();
  const elapsed = Math.min(
    0.05,
    Math.max(0, (start - lastTick) / 1000),
  );
  lastTick = start;
  accumulator += elapsed;

  let steps = 0;
  while (accumulator >= FIXED_DT && steps < 2) {
    simulateStep();
    accumulator -= FIXED_DT;
    steps += 1;
  }
  if (steps === 0) {
    simulateStep();
    accumulator = 0;
  }

  const solveTime = performance.now() - start;
  if (solveTime > 20) slowFrames += 1;
  else slowFrames = Math.max(0, slowFrames - 1);
  if (slowFrames > 28 && solverIterations > 2) {
    solverIterations = 2;
    slowFrames = 0;
  }
  emitFrame();
}

function startTimer() {
  if (timer) return;
  lastTick = performance.now();
  timer = setInterval(tick, 1000 / 60);
}

self.onmessage = (event) => {
  const data = event.data;
  if (data.type === "init") {
    initialized = true;
    startTimer();
    return;
  }
  if (data.type === "resize") {
    viewportWidth = Math.max(1, data.width);
    viewportHeight = Math.max(1, data.height);
    reducedMotion = Boolean(data.reducedMotion);
    initializeParticles();
    return;
  }
  if (data.type === "pointer") {
    pointer.active = Boolean(data.active);
    if (pointer.active) {
      pointer.x = data.x;
      pointer.y = data.y;
      pointer.vx = data.vx;
      pointer.vy = data.vy;
      pointer.burst = data.burst || 0;
      pointerSequence += 1;
    }
    return;
  }
  if (data.type === "visibility") {
    paused = Boolean(data.paused);
    lastTick = performance.now();
    accumulator = 0;
    return;
  }
  if (
    data.type === "recycle" &&
    data.generation === generation &&
    data.buffer.byteLength ===
      particleCount * 3 * Float32Array.BYTES_PER_ELEMENT
  ) {
    renderBuffers.push(data.buffer);
  }
};
