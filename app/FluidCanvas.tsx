"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { CpuFluidCanvas } from "./CpuFluidCanvas";

type GpuFluidCanvasProps = {
  onUnavailable: () => void;
  showControls: boolean;
};

type FluidCanvasProps = {
  showControls?: boolean;
};

type SimulationResources = {
  positions: any;
  previousPositions: any;
  velocities: any;
  lambdas: any;
  corrections: any;
  gridHeads: any;
  gridNext: any;
  smoothedVelocities: any;
  simulationUniform: any;
  renderUniform: any;
  blurUniform: any;
  densityTexture: any;
  horizontalBlurTexture: any;
  smoothDensityTexture: any;
  densityView: any;
  horizontalBlurView: any;
  smoothDensityView: any;
  computeBindGroup: any;
  densityBindGroup: any;
  horizontalBlurBindGroup: any;
  verticalBlurBindGroup: any;
  surfaceBindGroup: any;
  simulationParameters: Float32Array;
  renderParameters: Float32Array;
  blurParameters: Float32Array;
  particleCount: number;
  gridCellCount: number;
  workgroupCount: number;
  width: number;
  height: number;
  densityWidth: number;
  densityHeight: number;
  spacing: number;
  smoothingRadius: number;
  restDensity: number;
  gridWidth: number;
  gridHeight: number;
};

type PointerState = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  active: boolean;
  burst: number;
  previousX: number;
  previousY: number;
  previousTime: number;
};

const WORKGROUP_SIZE = 128;
const FIXED_TIME_STEP = 1 / 60;
const SOLVER_ITERATIONS = 3;

function calculateRestDensity(
  smoothingRadius: number,
  spacing: number,
) {
  const radiusSquared = smoothingRadius * smoothingRadius;
  const radius4 = radiusSquared * radiusSquared;
  const radius8 = radius4 * radius4;
  const coefficient = 4 / (Math.PI * radius8);
  const extent = Math.ceil(smoothingRadius / (spacing * 0.89));
  let density = 0;

  for (let row = -extent; row <= extent; row += 1) {
    for (let column = -extent; column <= extent; column += 1) {
      if (row === 0 && column === 0) continue;
      const rowOffset = Math.abs(row) % 2 === 1 ? 0.48 : 0;
      const x = (column + rowOffset) * spacing;
      const y = row * spacing * 0.89;
      const distanceSquared = x * x + y * y;
      if (distanceSquared <= 0 || distanceSquared >= radiusSquared) {
        continue;
      }
      const difference = radiusSquared - distanceSquared;
      density +=
        coefficient * difference * difference * difference;
    }
  }

  return Math.max(density, 0.000001);
}

function GpuFluidCanvas({
  onUnavailable,
  showControls,
}: GpuFluidCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const surfaceEnabledRef = useRef(true);
  const surfaceBlendRef = useRef(0.78);
  const visualDirtyRef = useRef(true);
  const [surfaceEnabled, setSurfaceEnabled] = useState(true);
  const [surfaceBlend, setSurfaceBlend] = useState(78);
  const [particleCount, setParticleCount] = useState(0);

  const toggleSurface = () => {
    setSurfaceEnabled((currentValue) => {
      const nextValue = !currentValue;
      surfaceEnabledRef.current = nextValue;
      visualDirtyRef.current = true;
      return nextValue;
    });
  };

  const updateSurfaceBlend = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const nextValue = Number(event.target.value);
    setSurfaceBlend(nextValue);
    surfaceBlendRef.current = nextValue / 100;
    visualDirtyRef.current = true;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;
    let animationFrame = 0;
    let resizeFrame = 0;
    let removeEventListeners = () => {};
    let releaseGpu = () => {};

    const start = async () => {
      const gpu = (navigator as any).gpu;
      const bufferUsage = (globalThis as any).GPUBufferUsage;
      const textureUsage = (globalThis as any).GPUTextureUsage;
      const shaderStage = (globalThis as any).GPUShaderStage;
      if (
        !gpu ||
        !bufferUsage ||
        !textureUsage ||
        !shaderStage ||
        !window.isSecureContext
      ) {
        onUnavailable();
        return;
      }

      const adapter = await gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter || disposed) {
        onUnavailable();
        return;
      }

      const device = await adapter.requestDevice();
      if (disposed) {
        device.destroy();
        return;
      }

      const context = canvas.getContext("webgpu") as any;
      if (!context) {
        device.destroy();
        onUnavailable();
        return;
      }

      const shaderResponses = await Promise.all([
        fetch("/gpu-pbf.wgsl"),
        fetch("/gpu-density.wgsl"),
        fetch("/gpu-blur.wgsl"),
        fetch("/gpu-surface.wgsl"),
      ]);
      if (shaderResponses.some((response) => !response.ok)) {
        throw new Error("WebGPU shader files could not be loaded.");
      }
      const [
        computeSource,
        densitySource,
        blurSource,
        surfaceSource,
      ] =
        await Promise.all(
          shaderResponses.map((response) => response.text()),
        );

      const computeModule = device.createShaderModule({
        label: "DOTAFS GPU PBF",
        code: computeSource,
      });
      const densityModule = device.createShaderModule({
        label: "DOTAFS density splat",
        code: densitySource,
      });
      const blurModule = device.createShaderModule({
        label: "DOTAFS density blur",
        code: blurSource,
      });
      const surfaceModule = device.createShaderModule({
        label: "DOTAFS water surface",
        code: surfaceSource,
      });

      const validateShader = async (module: any, label: string) => {
        if (!module.getCompilationInfo) return;
        const compilationInfo = await module.getCompilationInfo();
        const errors = compilationInfo.messages.filter(
          (message: any) => message.type === "error",
        );
        if (errors.length > 0) {
          const firstError = errors[0];
          throw new Error(
            `${label}: ${firstError.message} (${firstError.lineNum}:${firstError.linePos})`,
          );
        }
      };
      await Promise.all([
        validateShader(computeModule, "PBF compute"),
        validateShader(densityModule, "Density render"),
        validateShader(blurModule, "Density blur"),
        validateShader(surfaceModule, "Surface render"),
      ]);

      const computeBindGroupLayout = device.createBindGroupLayout({
        label: "PBF resources",
        entries: [
          ...Array.from({ length: 8 }, (_, binding) => ({
            binding,
            visibility: shaderStage.COMPUTE,
            buffer: { type: "storage" },
          })),
          {
            binding: 8,
            visibility: shaderStage.COMPUTE,
            buffer: { type: "uniform" },
          },
        ],
      });
      const densityBindGroupLayout = device.createBindGroupLayout({
        label: "Density resources",
        entries: [
          {
            binding: 0,
            visibility: shaderStage.VERTEX,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 1,
            visibility: shaderStage.VERTEX,
            buffer: { type: "uniform" },
          },
        ],
      });
      const surfaceBindGroupLayout = device.createBindGroupLayout({
        label: "Surface resources",
        entries: [
          {
            binding: 0,
            visibility: shaderStage.FRAGMENT,
            buffer: { type: "uniform" },
          },
          {
            binding: 1,
            visibility: shaderStage.FRAGMENT,
            texture: { sampleType: "float" },
          },
          {
            binding: 2,
            visibility: shaderStage.FRAGMENT,
            sampler: { type: "filtering" },
          },
        ],
      });
      const blurBindGroupLayout = device.createBindGroupLayout({
        label: "Blur resources",
        entries: [
          {
            binding: 0,
            visibility: shaderStage.FRAGMENT,
            texture: { sampleType: "float" },
          },
          {
            binding: 1,
            visibility: shaderStage.FRAGMENT,
            sampler: { type: "filtering" },
          },
          {
            binding: 2,
            visibility: shaderStage.FRAGMENT,
            buffer: { type: "uniform" },
          },
        ],
      });

      const computePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [computeBindGroupLayout],
      });
      const densityPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [densityBindGroupLayout],
      });
      const surfacePipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [surfaceBindGroupLayout],
      });
      const blurPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [blurBindGroupLayout],
      });

      const computeEntryPoints = [
        "integrate",
        "buildHash",
        "solveLambda",
        "solveCorrection",
        "applyCorrection",
        "reconstructVelocity",
        "solveViscosity",
        "applyViscosity",
      ] as const;
      const computePipelines = Object.fromEntries(
        computeEntryPoints.map((entryPoint) => [
          entryPoint,
          device.createComputePipeline({
            label: `PBF ${entryPoint}`,
            layout: computePipelineLayout,
            compute: {
              module: computeModule,
              entryPoint,
            },
          }),
        ]),
      ) as Record<(typeof computeEntryPoints)[number], any>;

      const canvasFormat = gpu.getPreferredCanvasFormat();
      const densityPipeline = device.createRenderPipeline({
        label: "Metaball density",
        layout: densityPipelineLayout,
        vertex: {
          module: densityModule,
          entryPoint: "densityVertex",
        },
        fragment: {
          module: densityModule,
          entryPoint: "densityFragment",
          targets: [
            {
              format: "rgba16float",
              blend: {
                color: {
                  operation: "add",
                  srcFactor: "one",
                  dstFactor: "one",
                },
                alpha: {
                  operation: "add",
                  srcFactor: "one",
                  dstFactor: "one",
                },
              },
            },
          ],
        },
        primitive: { topology: "triangle-list" },
      });
      const particlePipeline = device.createRenderPipeline({
        label: "Direct GPU particles",
        layout: densityPipelineLayout,
        vertex: {
          module: densityModule,
          entryPoint: "densityVertex",
        },
        fragment: {
          module: densityModule,
          entryPoint: "particleFragment",
          targets: [
            {
              format: canvasFormat,
              blend: {
                color: {
                  operation: "add",
                  srcFactor: "src-alpha",
                  dstFactor: "one-minus-src-alpha",
                },
                alpha: {
                  operation: "add",
                  srcFactor: "one",
                  dstFactor: "one-minus-src-alpha",
                },
              },
            },
          ],
        },
        primitive: { topology: "triangle-list" },
      });
      const createBlurPipeline = (
        label: string,
        entryPoint: "blurHorizontal" | "blurVertical",
      ) =>
        device.createRenderPipeline({
          label,
          layout: blurPipelineLayout,
          vertex: {
            module: blurModule,
            entryPoint: "fullscreenVertex",
          },
          fragment: {
            module: blurModule,
            entryPoint,
            targets: [{ format: "rgba16float" }],
          },
          primitive: { topology: "triangle-list" },
        });
      const horizontalBlurPipeline = createBlurPipeline(
        "Horizontal density blur",
        "blurHorizontal",
      );
      const verticalBlurPipeline = createBlurPipeline(
        "Vertical density blur",
        "blurVertical",
      );
      const surfacePipeline = device.createRenderPipeline({
        label: "Water surface",
        layout: surfacePipelineLayout,
        vertex: {
          module: surfaceModule,
          entryPoint: "fullscreenVertex",
        },
        fragment: {
          module: surfaceModule,
          entryPoint: "surfaceFragment",
          targets: [{ format: canvasFormat }],
        },
        primitive: { topology: "triangle-list" },
      });
      const densitySampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      });

      let resources: SimulationResources | undefined;
      const pointer: PointerState = {
        x: 0,
        y: 0,
        vx: 0,
        vy: 0,
        active: false,
        burst: 0,
        previousX: 0,
        previousY: 0,
        previousTime: performance.now(),
      };

      const createBuffer = (
        label: string,
        byteLength: number,
        usage: number,
        initialData?: ArrayBufferView,
      ) => {
        const alignedSize = Math.max(4, Math.ceil(byteLength / 4) * 4);
        const buffer = device.createBuffer({
          label,
          size: alignedSize,
          usage,
          mappedAtCreation: Boolean(initialData),
        });
        if (initialData) {
          const destination = new Uint8Array(buffer.getMappedRange());
          destination.set(
            new Uint8Array(
              initialData.buffer,
              initialData.byteOffset,
              initialData.byteLength,
            ),
          );
          buffer.unmap();
        }
        return buffer;
      };

      const destroyResources = (
        resourcesToDestroy: SimulationResources,
      ) => {
        resourcesToDestroy.positions.destroy();
        resourcesToDestroy.previousPositions.destroy();
        resourcesToDestroy.velocities.destroy();
        resourcesToDestroy.lambdas.destroy();
        resourcesToDestroy.corrections.destroy();
        resourcesToDestroy.gridHeads.destroy();
        resourcesToDestroy.gridNext.destroy();
        resourcesToDestroy.smoothedVelocities.destroy();
        resourcesToDestroy.simulationUniform.destroy();
        resourcesToDestroy.renderUniform.destroy();
        resourcesToDestroy.blurUniform.destroy();
        resourcesToDestroy.densityTexture.destroy();
        resourcesToDestroy.horizontalBlurTexture.destroy();
        resourcesToDestroy.smoothDensityTexture.destroy();
      };

      const createSimulationResources = (
        width: number,
        height: number,
      ): SimulationResources => {
        const targetCount =
          width < 700 ? 3200 : width < 1400 ? 6200 : 9000;
        const spacing = Math.max(
          10,
          Math.min(
            15,
            Math.sqrt((width * height * 0.5) / targetCount),
          ),
        );
        const padding = spacing * 1.1;
        const columns = Math.max(
          1,
          Math.floor((width - padding * 2) / spacing),
        );
        const maximumRows = Math.max(
          1,
          Math.floor((height * 0.62) / (spacing * 0.89)),
        );
        const rows = Math.min(
          maximumRows,
          Math.ceil(targetCount / columns),
        );
        const particleCount = Math.min(
          targetCount,
          columns * rows,
        );
        const positions = new Float32Array(particleCount * 2);

        let particleIndex = 0;
        for (
          let row = 0;
          row < rows && particleIndex < particleCount;
          row += 1
        ) {
          for (
            let column = 0;
            column < columns && particleIndex < particleCount;
            column += 1
          ) {
            const offset = particleIndex * 2;
            const rowOffset = row % 2 === 0 ? 0 : spacing * 0.48;
            positions[offset] =
              padding +
              column * spacing +
              rowOffset +
              (Math.random() - 0.5) * spacing * 0.018;
            positions[offset + 1] =
              height -
              padding -
              row * spacing * 0.89 +
              (Math.random() - 0.5) * spacing * 0.018;
            particleIndex += 1;
          }
        }

        const smoothingRadius = spacing * 2.2;
        const restDensity = calculateRestDensity(
          smoothingRadius,
          spacing,
        );
        const gridWidth = Math.max(
          1,
          Math.ceil(width / smoothingRadius),
        );
        const gridHeight = Math.max(
          1,
          Math.ceil(height / smoothingRadius),
        );
        const gridCellCount = gridWidth * gridHeight;
        const pixelRatio = Math.min(
          window.devicePixelRatio || 1,
          1.35,
        );
        canvas.width = Math.max(1, Math.round(width * pixelRatio));
        canvas.height = Math.max(1, Math.round(height * pixelRatio));
        const densityWidth = Math.max(
          1,
          Math.round(canvas.width * 0.4),
        );
        const densityHeight = Math.max(
          1,
          Math.round(canvas.height * 0.4),
        );

        context.configure({
          device,
          format: canvasFormat,
          alphaMode: "opaque",
        });

        const storageUsage =
          bufferUsage.STORAGE | bufferUsage.COPY_DST;
        const positionBuffer = createBuffer(
          "Particle positions",
          positions.byteLength,
          storageUsage,
          positions,
        );
        const previousPositions = createBuffer(
          "Previous positions",
          positions.byteLength,
          storageUsage,
          positions,
        );
        const velocities = createBuffer(
          "Particle velocities",
          positions.byteLength,
          storageUsage,
          new Float32Array(particleCount * 2),
        );
        const lambdas = createBuffer(
          "PBF lambdas",
          particleCount * Float32Array.BYTES_PER_ELEMENT,
          storageUsage,
        );
        const corrections = createBuffer(
          "PBF corrections",
          positions.byteLength,
          storageUsage,
        );
        const gridHeads = createBuffer(
          "Spatial hash heads",
          gridCellCount * Uint32Array.BYTES_PER_ELEMENT,
          storageUsage,
        );
        const gridNext = createBuffer(
          "Spatial hash links",
          particleCount * Uint32Array.BYTES_PER_ELEMENT,
          storageUsage,
        );
        const smoothedVelocities = createBuffer(
          "XSPH velocities",
          positions.byteLength,
          storageUsage,
        );
        const simulationUniform = createBuffer(
          "PBF parameters",
          20 * Float32Array.BYTES_PER_ELEMENT,
          bufferUsage.UNIFORM | bufferUsage.COPY_DST,
        );
        const renderUniform = createBuffer(
          "Water render parameters",
          12 * Float32Array.BYTES_PER_ELEMENT,
          bufferUsage.UNIFORM | bufferUsage.COPY_DST,
        );
        const blurUniform = createBuffer(
          "Density blur parameters",
          4 * Float32Array.BYTES_PER_ELEMENT,
          bufferUsage.UNIFORM | bufferUsage.COPY_DST,
        );
        const densityTexture = device.createTexture({
          label: "Metaball density texture",
          size: {
            width: densityWidth,
            height: densityHeight,
          },
          format: "rgba16float",
          usage:
            textureUsage.RENDER_ATTACHMENT |
            textureUsage.TEXTURE_BINDING,
        });
        const horizontalBlurTexture = device.createTexture({
          label: "Horizontal density blur texture",
          size: {
            width: densityWidth,
            height: densityHeight,
          },
          format: "rgba16float",
          usage:
            textureUsage.RENDER_ATTACHMENT |
            textureUsage.TEXTURE_BINDING,
        });
        const smoothDensityTexture = device.createTexture({
          label: "Smooth density texture",
          size: {
            width: densityWidth,
            height: densityHeight,
          },
          format: "rgba16float",
          usage:
            textureUsage.RENDER_ATTACHMENT |
            textureUsage.TEXTURE_BINDING,
        });
        const densityView = densityTexture.createView();
        const horizontalBlurView =
          horizontalBlurTexture.createView();
        const smoothDensityView =
          smoothDensityTexture.createView();

        const computeBindGroup = device.createBindGroup({
          layout: computeBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: previousPositions } },
            { binding: 2, resource: { buffer: velocities } },
            { binding: 3, resource: { buffer: lambdas } },
            { binding: 4, resource: { buffer: corrections } },
            { binding: 5, resource: { buffer: gridHeads } },
            { binding: 6, resource: { buffer: gridNext } },
            { binding: 7, resource: { buffer: smoothedVelocities } },
            { binding: 8, resource: { buffer: simulationUniform } },
          ],
        });
        const densityBindGroup = device.createBindGroup({
          layout: densityBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: positionBuffer } },
            { binding: 1, resource: { buffer: renderUniform } },
          ],
        });
        const horizontalBlurBindGroup = device.createBindGroup({
          layout: blurBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: densityView,
            },
            { binding: 1, resource: densitySampler },
            { binding: 2, resource: { buffer: blurUniform } },
          ],
        });
        const verticalBlurBindGroup = device.createBindGroup({
          layout: blurBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: horizontalBlurView,
            },
            { binding: 1, resource: densitySampler },
            { binding: 2, resource: { buffer: blurUniform } },
          ],
        });
        const surfaceBindGroup = device.createBindGroup({
          layout: surfaceBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: renderUniform } },
            {
              binding: 1,
              resource: smoothDensityView,
            },
            { binding: 2, resource: densitySampler },
          ],
        });
        return {
          positions: positionBuffer,
          previousPositions,
          velocities,
          lambdas,
          corrections,
          gridHeads,
          gridNext,
          smoothedVelocities,
          simulationUniform,
          renderUniform,
          blurUniform,
          densityTexture,
          horizontalBlurTexture,
          smoothDensityTexture,
          densityView,
          horizontalBlurView,
          smoothDensityView,
          computeBindGroup,
          densityBindGroup,
          horizontalBlurBindGroup,
          verticalBlurBindGroup,
          surfaceBindGroup,
          simulationParameters: new Float32Array(20),
          renderParameters: new Float32Array(12),
          blurParameters: new Float32Array(4),
          particleCount,
          gridCellCount,
          workgroupCount: Math.ceil(
            particleCount / WORKGROUP_SIZE,
          ),
          width,
          height,
          densityWidth,
          densityHeight,
          spacing,
          smoothingRadius,
          restDensity,
          gridWidth,
          gridHeight,
        };
      };

      const resize = () => {
        const bounds = canvas.getBoundingClientRect();
        const width = Math.max(1, bounds.width);
        const height = Math.max(1, bounds.height);
        const previousResources = resources;
        resources = createSimulationResources(width, height);
        visualDirtyRef.current = true;
        pointer.x = width * 0.5;
        pointer.y = height * 0.5;
        setParticleCount(resources.particleCount);

        if (previousResources) {
          device.queue
            .onSubmittedWorkDone()
            .then(() => destroyResources(previousResources))
            .catch(() => {});
        }
      };

      const queueResize = () => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(resize);
      };

      const updatePointer = (event: PointerEvent) => {
        const bounds = canvas.getBoundingClientRect();
        const x = event.clientX - bounds.left;
        const y = event.clientY - bounds.top;
        const now = performance.now();
        const elapsed = Math.max(
          8,
          Math.min(64, now - pointer.previousTime),
        );
        let velocityX =
          ((x - pointer.previousX) / elapsed) * 1000;
        let velocityY =
          ((y - pointer.previousY) / elapsed) * 1000;
        const speed = Math.hypot(velocityX, velocityY);
        if (speed > 5200) {
          const scale = 5200 / speed;
          velocityX *= scale;
          velocityY *= scale;
        }

        pointer.x = x;
        pointer.y = y;
        pointer.vx = velocityX;
        pointer.vy = velocityY;
        pointer.active = true;
        pointer.previousX = x;
        pointer.previousY = y;
        pointer.previousTime = now;
      };

      const pointerMove = (event: PointerEvent) => {
        updatePointer(event);
      };
      const pointerDown = (event: PointerEvent) => {
        canvas.setPointerCapture(event.pointerId);
        updatePointer(event);
        pointer.burst = 1450;
      };
      const pointerUp = (event: PointerEvent) => {
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
        pointer.burst = 0;
      };
      const pointerLeave = () => {
        pointer.active = false;
        pointer.vx = 0;
        pointer.vy = 0;
      };

      canvas.addEventListener("pointermove", pointerMove);
      canvas.addEventListener("pointerdown", pointerDown);
      canvas.addEventListener("pointerup", pointerUp);
      canvas.addEventListener("pointercancel", pointerUp);
      canvas.addEventListener("pointerleave", pointerLeave);
      window.addEventListener("resize", queueResize);
      removeEventListeners = () => {
        canvas.removeEventListener("pointermove", pointerMove);
        canvas.removeEventListener("pointerdown", pointerDown);
        canvas.removeEventListener("pointerup", pointerUp);
        canvas.removeEventListener("pointercancel", pointerUp);
        canvas.removeEventListener("pointerleave", pointerLeave);
        window.removeEventListener("resize", queueResize);
      };

      resize();

      const encodeParticlePass = (
        encoder: any,
        pipeline: any,
        activeResources: SimulationResources,
      ) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, activeResources.computeBindGroup);
        pass.dispatchWorkgroups(activeResources.workgroupCount);
        pass.end();
      };

      const encodeSimulationStep = (
        encoder: any,
        activeResources: SimulationResources,
      ) => {
        encodeParticlePass(
          encoder,
          computePipelines.integrate,
          activeResources,
        );

        encoder.clearBuffer(activeResources.gridHeads);
        const constraintPass = encoder.beginComputePass();
        constraintPass.setBindGroup(
          0,
          activeResources.computeBindGroup,
        );
        constraintPass.setPipeline(computePipelines.buildHash);
        constraintPass.dispatchWorkgroups(
          activeResources.workgroupCount,
        );
        for (
          let iteration = 0;
          iteration < SOLVER_ITERATIONS;
          iteration += 1
        ) {
          constraintPass.setPipeline(computePipelines.solveLambda);
          constraintPass.dispatchWorkgroups(
            activeResources.workgroupCount,
          );
          constraintPass.setPipeline(
            computePipelines.solveCorrection,
          );
          constraintPass.dispatchWorkgroups(
            activeResources.workgroupCount,
          );
          constraintPass.setPipeline(
            computePipelines.applyCorrection,
          );
          constraintPass.dispatchWorkgroups(
            activeResources.workgroupCount,
          );
        }
        constraintPass.end();

        encoder.clearBuffer(activeResources.gridHeads);
        const finalPass = encoder.beginComputePass();
        finalPass.setBindGroup(0, activeResources.computeBindGroup);
        finalPass.setPipeline(computePipelines.buildHash);
        finalPass.dispatchWorkgroups(activeResources.workgroupCount);
        finalPass.setPipeline(computePipelines.reconstructVelocity);
        finalPass.dispatchWorkgroups(activeResources.workgroupCount);
        finalPass.setPipeline(computePipelines.solveViscosity);
        finalPass.dispatchWorkgroups(activeResources.workgroupCount);
        finalPass.setPipeline(computePipelines.applyViscosity);
        finalPass.dispatchWorkgroups(activeResources.workgroupCount);
        finalPass.end();
      };

      let previousTime = performance.now();
      let accumulator = 0;

      const render = (now: number) => {
        if (disposed || !resources) return;
        const activeResources = resources;
        const elapsed = Math.min(
          0.05,
          Math.max(0, (now - previousTime) / 1000),
        );
        previousTime = now;
        accumulator += elapsed;

        let steps = 0;
        while (accumulator >= FIXED_TIME_STEP && steps < 2) {
          accumulator -= FIXED_TIME_STEP;
          steps += 1;
        }
        if (steps === 2 && accumulator >= FIXED_TIME_STEP) {
          accumulator = 0;
        }
        if (steps === 0 && !visualDirtyRef.current) {
          animationFrame = requestAnimationFrame(render);
          return;
        }

        const simulationParameters =
          activeResources.simulationParameters;
        simulationParameters[0] = activeResources.width;
        simulationParameters[1] = activeResources.height;
        simulationParameters[2] =
          activeResources.smoothingRadius;
        simulationParameters[3] = FIXED_TIME_STEP;
        simulationParameters[4] = activeResources.restDensity;
        simulationParameters[5] = 0.001;
        simulationParameters[6] =
          activeResources.particleCount;
        simulationParameters[7] = 980;
        simulationParameters[8] = activeResources.gridWidth;
        simulationParameters[9] = activeResources.gridHeight;
        simulationParameters[10] = 0.06;
        simulationParameters[11] = 2600;
        simulationParameters[12] = pointer.x;
        simulationParameters[13] = pointer.y;
        simulationParameters[14] = pointer.vx;
        simulationParameters[15] = pointer.vy;
        simulationParameters[16] = Math.max(
          76,
          activeResources.spacing * 7,
        );
        simulationParameters[17] = pointer.burst;
        simulationParameters[18] = pointer.active ? 1 : 0;
        simulationParameters[19] =
          activeResources.spacing * 0.62;

        const surfaceBlendValue = surfaceBlendRef.current;
        const surfaceMode = surfaceEnabledRef.current;
        const renderParameters = activeResources.renderParameters;
        renderParameters[0] = canvas.width;
        renderParameters[1] = canvas.height;
        renderParameters[2] = activeResources.densityWidth;
        renderParameters[3] = activeResources.densityHeight;
        renderParameters[4] = activeResources.width;
        renderParameters[5] = activeResources.height;
        renderParameters[6] = surfaceMode
          ? activeResources.spacing *
            (1.5 + surfaceBlendValue * 0.72)
          : activeResources.spacing * 0.48;
        renderParameters[7] = surfaceMode ? 1 : 0;
        renderParameters[8] =
          0.67 - surfaceBlendValue * 0.25;
        renderParameters[9] =
          0.055 + surfaceBlendValue * 0.045;
        renderParameters[10] = activeResources.particleCount;
        renderParameters[11] = 0;

        const blurParameters = activeResources.blurParameters;
        blurParameters[0] =
          0.72 + surfaceBlendValue * 1.72;
        blurParameters[1] = 0;
        blurParameters[2] = 0;
        blurParameters[3] = 0;

        if (steps > 0) {
          device.queue.writeBuffer(
            activeResources.simulationUniform,
            0,
            simulationParameters,
          );
        }
        device.queue.writeBuffer(
          activeResources.renderUniform,
          0,
          renderParameters,
        );
        device.queue.writeBuffer(
          activeResources.blurUniform,
          0,
          blurParameters,
        );

        const encoder = device.createCommandEncoder({
          label: "DOTAFS fluid frame",
        });
        for (let step = 0; step < steps; step += 1) {
          encodeSimulationStep(encoder, activeResources);
        }

        if (surfaceMode) {
          const densityPass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: activeResources.densityView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
          });
          densityPass.setPipeline(densityPipeline);
          densityPass.setBindGroup(
            0,
            activeResources.densityBindGroup,
          );
          densityPass.draw(
            6,
            activeResources.particleCount,
            0,
            0,
          );
          densityPass.end();

          const horizontalBlurPass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: activeResources.horizontalBlurView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
          });
          horizontalBlurPass.setPipeline(horizontalBlurPipeline);
          horizontalBlurPass.setBindGroup(
            0,
            activeResources.horizontalBlurBindGroup,
          );
          horizontalBlurPass.draw(3);
          horizontalBlurPass.end();

          const verticalBlurPass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: activeResources.smoothDensityView,
                clearValue: { r: 0, g: 0, b: 0, a: 0 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
          });
          verticalBlurPass.setPipeline(verticalBlurPipeline);
          verticalBlurPass.setBindGroup(
            0,
            activeResources.verticalBlurBindGroup,
          );
          verticalBlurPass.draw(3);
          verticalBlurPass.end();

          const surfacePass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
          });
          surfacePass.setPipeline(surfacePipeline);
          surfacePass.setBindGroup(
            0,
            activeResources.surfaceBindGroup,
          );
          surfacePass.draw(3);
          surfacePass.end();
        } else {
          const particlePass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: "clear",
                storeOp: "store",
              },
            ],
          });
          particlePass.setPipeline(particlePipeline);
          particlePass.setBindGroup(
            0,
            activeResources.densityBindGroup,
          );
          particlePass.draw(
            6,
            activeResources.particleCount,
            0,
            0,
          );
          particlePass.end();
        }

        device.queue.submit([encoder.finish()]);
        visualDirtyRef.current = false;

        if (steps > 0) {
          pointer.vx *= 0.68;
          pointer.vy *= 0.68;
          pointer.burst *= 0.5;
          if (Math.abs(pointer.vx) < 0.5) pointer.vx = 0;
          if (Math.abs(pointer.vy) < 0.5) pointer.vy = 0;
          if (pointer.burst < 1) pointer.burst = 0;
        }
        animationFrame = requestAnimationFrame(render);
      };

      animationFrame = requestAnimationFrame(render);
      device.lost.then(() => {
        if (!disposed) onUnavailable();
      });

      releaseGpu = () => {
        cancelAnimationFrame(animationFrame);
        cancelAnimationFrame(resizeFrame);
        removeEventListeners();
        if (resources) {
          destroyResources(resources);
          resources = undefined;
        }
        context.unconfigure();
        device.destroy();
      };
    };

    start().catch((error) => {
      console.warn("WebGPU PBF fallback:", error);
      if (!disposed) onUnavailable();
    });

    return () => {
      disposed = true;
      releaseGpu();
    };
  }, [onUnavailable]);

  return (
    <div className="fluidStage">
      <canvas
        ref={canvasRef}
        className="fluidCanvas"
        aria-label="使用 WebGPU PBF 计算、可通过鼠标推动的二维水体"
      />
      {showControls ? (
        <>
          <button
            className="surfaceToggle"
            type="button"
            aria-pressed={surfaceEnabled}
            onClick={toggleSurface}
          >
            <span>GPU SURFACE</span>
            <strong>
              {surfaceEnabled ? "ON" : "PARTICLES"}
            </strong>
          </button>
          <label
            className={`surfaceControl ${
              surfaceEnabled ? "" : "isDisabled"
            }`}
          >
            <span>
              <b>SURFACE BLEND</b>
              <output>{surfaceBlend}%</output>
            </span>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={surfaceBlend}
              disabled={!surfaceEnabled}
              onChange={updateSurfaceBlend}
              aria-label="连续水面的融合粗细"
            />
          </label>
          <div
            className={`fluidStatus ${
              particleCount > 0 ? "isReady" : ""
            }`}
          >
            <span>WEBGPU PBF</span>
            <strong>
              {particleCount > 0
                ? `${particleCount.toLocaleString()} PARTICLES`
                : "REQUESTING GPU"}
            </strong>
          </div>
        </>
      ) : null}
    </div>
  );
}

export function FluidCanvas({
  showControls = true,
}: FluidCanvasProps) {
  const [useCpuFallback, setUseCpuFallback] = useState(false);
  const enableCpuFallback = useCallback(
    () => setUseCpuFallback(true),
    [],
  );

  if (useCpuFallback) {
    return (
      <>
        <CpuFluidCanvas />
        <div className="fallbackBadge" role="status">
          CPU FALLBACK / WEBGPU UNAVAILABLE
        </div>
      </>
    );
  }

  return (
    <GpuFluidCanvas
      onUnavailable={enableCpuFallback}
      showControls={showControls}
    />
  );
}
