"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

type GpuCloudCanvasProps = {
  onUnavailable: () => void;
};

type CloudResources = {
  positions: any;
  previousPositions: any;
  velocities: any;
  states: any;
  lambdas: any;
  corrections: any;
  gridHeads: any;
  gridNext: any;
  waterActive: any;
  cloudActive: any;
  simulationUniform: any;
  renderUniform: any;
  blurUniform: any;
  densityTexture: any;
  horizontalBlurTexture: any;
  smoothDensityTexture: any;
  densityView: any;
  horizontalBlurView: any;
  smoothDensityView: any;
  integrateBindGroup: any;
  classifyBindGroup: any;
  waterBindGroup: any;
  cloudBindGroup: any;
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
  cloudRadius: number;
  cloudGridWidth: number;
  cloudGridHeight: number;
};

type CloudPointer = {
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
const CLOUD_SOLVER_ITERATIONS = 3;

function smoothStep(
  edge0: number,
  edge1: number,
  value: number,
) {
  const normalized = Math.max(
    0,
    Math.min(1, (value - edge0) / (edge1 - edge0)),
  );
  return normalized * normalized * (3 - 2 * normalized);
}

function hashValue(value: number) {
  const sine = Math.sin(value * 12.9898) * 43758.5453;
  return sine - Math.floor(sine);
}

function calculateRestDensity(
  smoothingRadius: number,
  spacing: number,
) {
  const radiusSquared = smoothingRadius * smoothingRadius;
  const radius4 = radiusSquared * radiusSquared;
  const radius8 = radius4 * radius4;
  const coefficient = 4 / (Math.PI * radius8);
  const extent = Math.ceil(
    smoothingRadius / (spacing * 0.89),
  );
  let density = 0;

  for (let row = -extent; row <= extent; row += 1) {
    for (
      let column = -extent;
      column <= extent;
      column += 1
    ) {
      if (row === 0 && column === 0) continue;
      const rowOffset =
        Math.abs(row) % 2 === 1 ? 0.48 : 0;
      const x = (column + rowOffset) * spacing;
      const y = row * spacing * 0.89;
      const distanceSquared = x * x + y * y;
      if (
        distanceSquared <= 0 ||
        distanceSquared >= radiusSquared
      ) {
        continue;
      }
      const difference = radiusSquared - distanceSquared;
      density +=
        coefficient *
        difference *
        difference *
        difference;
    }
  }

  return Math.max(density, 0.000001);
}

function GpuCloudCanvas({
  onUnavailable,
}: GpuCloudCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sourceEnabledRef = useRef(true);
  const surfaceEnabledRef = useRef(true);
  const cohesionRef = useRef(0.72);
  const [sourceEnabled, setSourceEnabled] = useState(true);
  const [surfaceEnabled, setSurfaceEnabled] = useState(true);
  const [cohesion, setCohesion] = useState(72);
  const [particleCount, setParticleCount] = useState(0);

  const toggleSource = () => {
    setSourceEnabled((currentValue) => {
      const nextValue = !currentValue;
      sourceEnabledRef.current = nextValue;
      return nextValue;
    });
  };

  const toggleSurface = () => {
    setSurfaceEnabled((currentValue) => {
      const nextValue = !currentValue;
      surfaceEnabledRef.current = nextValue;
      return nextValue;
    });
  };

  const updateCohesion = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const nextValue = Number(event.target.value);
    setCohesion(nextValue);
    cohesionRef.current = nextValue / 100;
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
      if (
        !adapter ||
        adapter.limits.maxStorageBuffersPerShaderStage < 8 ||
        disposed
      ) {
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
        fetch("/gpu-weather-integrate.wgsl"),
        fetch("/gpu-weather-classify.wgsl"),
        fetch("/gpu-weather-water.wgsl"),
        fetch("/gpu-weather-cloud.wgsl"),
        fetch("/gpu-cloud-density.wgsl"),
        fetch("/gpu-blur.wgsl"),
        fetch("/gpu-cloud-surface.wgsl"),
      ]);
      if (shaderResponses.some((response) => !response.ok)) {
        throw new Error("WebGPU cloud shaders could not be loaded.");
      }
      const [
        integrateSource,
        classifySource,
        waterSource,
        cloudSource,
        densitySource,
        blurSource,
        surfaceSource,
      ] = await Promise.all(
        shaderResponses.map((response) => response.text()),
      );

      const integrateModule = device.createShaderModule({
        label: "DOTAFS shared weather integration",
        code: integrateSource,
      });
      const classifyModule = device.createShaderModule({
        label: "DOTAFS logical pool classifier",
        code: classifySource,
      });
      const waterModule = device.createShaderModule({
        label: "DOTAFS water PBF pool",
        code: waterSource,
      });
      const cloudModule = device.createShaderModule({
        label: "DOTAFS cloud cohesion pool",
        code: cloudSource,
      });
      const densityModule = device.createShaderModule({
        label: "DOTAFS cloud density",
        code: densitySource,
      });
      const blurModule = device.createShaderModule({
        label: "DOTAFS cloud blur",
        code: blurSource,
      });
      const surfaceModule = device.createShaderModule({
        label: "DOTAFS cartoon cloud",
        code: surfaceSource,
      });

      const validateShader = async (
        module: any,
        label: string,
      ) => {
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
        validateShader(integrateModule, "Weather integration"),
        validateShader(classifyModule, "Pool classifier"),
        validateShader(waterModule, "Water PBF"),
        validateShader(cloudModule, "Cloud cohesion"),
        validateShader(densityModule, "Cloud density"),
        validateShader(blurModule, "Cloud blur"),
        validateShader(surfaceModule, "Cloud surface"),
      ]);

      const integrateBindGroupLayout =
        device.createBindGroupLayout({
          label: "Shared weather integration resources",
          entries: [
            ...Array.from({ length: 4 }, (_, binding) => ({
              binding,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "storage" },
            })),
            {
              binding: 4,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "uniform" },
            },
          ],
        });
      const classifyBindGroupLayout =
        device.createBindGroupLayout({
          label: "Logical weather pool resources",
          entries: [
            {
              binding: 0,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            },
            ...[1, 2].map((binding) => ({
              binding,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "storage" },
            })),
            {
              binding: 3,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "uniform" },
            },
          ],
        });
      const waterBindGroupLayout =
        device.createBindGroupLayout({
          label: "Water PBF active pool resources",
          entries: [
            {
              binding: 0,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "storage" },
            },
            {
              binding: 1,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            },
            ...[2, 3, 4, 5, 6].map((binding) => ({
              binding,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "storage" },
            })),
            {
              binding: 7,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            },
            {
              binding: 8,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "uniform" },
            },
          ],
        });
      const cloudBindGroupLayout =
        device.createBindGroupLayout({
          label: "Cloud cohesion active pool resources",
          entries: [
            {
              binding: 0,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            },
            {
              binding: 1,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "storage" },
            },
            {
              binding: 2,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            },
            ...[3, 4, 5].map((binding) => ({
              binding,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "storage" },
            })),
            {
              binding: 6,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "read-only-storage" },
            },
            {
              binding: 7,
              visibility: shaderStage.COMPUTE,
              buffer: { type: "uniform" },
            },
          ],
        });
      const densityBindGroupLayout =
        device.createBindGroupLayout({
          label: "Cloud density resources",
          entries: [
            {
              binding: 0,
              visibility: shaderStage.VERTEX,
              buffer: { type: "read-only-storage" },
            },
            {
              binding: 1,
              visibility: shaderStage.VERTEX,
              buffer: { type: "read-only-storage" },
            },
            {
              binding: 2,
              visibility: shaderStage.VERTEX,
              buffer: { type: "uniform" },
            },
          ],
        });
      const blurBindGroupLayout =
        device.createBindGroupLayout({
          label: "Cloud blur resources",
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
      const surfaceBindGroupLayout =
        device.createBindGroupLayout({
          label: "Cloud surface resources",
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

      const integratePipelineLayout =
        device.createPipelineLayout({
          bindGroupLayouts: [integrateBindGroupLayout],
        });
      const classifyPipelineLayout =
        device.createPipelineLayout({
          bindGroupLayouts: [classifyBindGroupLayout],
        });
      const waterPipelineLayout =
        device.createPipelineLayout({
          bindGroupLayouts: [waterBindGroupLayout],
        });
      const cloudPipelineLayout =
        device.createPipelineLayout({
          bindGroupLayouts: [cloudBindGroupLayout],
        });
      const densityPipelineLayout =
        device.createPipelineLayout({
          bindGroupLayouts: [densityBindGroupLayout],
        });
      const blurPipelineLayout =
        device.createPipelineLayout({
          bindGroupLayouts: [blurBindGroupLayout],
        });
      const surfacePipelineLayout =
        device.createPipelineLayout({
          bindGroupLayouts: [surfaceBindGroupLayout],
        });

      const integratePipeline = device.createComputePipeline({
        label: "Shared weather integrate",
        layout: integratePipelineLayout,
        compute: {
          module: integrateModule,
          entryPoint: "integrate",
        },
      });
      const classifyPipeline = device.createComputePipeline({
        label: "Classify shared particles",
        layout: classifyPipelineLayout,
        compute: {
          module: classifyModule,
          entryPoint: "classify",
        },
      });
      const finalizeDispatchPipeline =
        device.createComputePipeline({
          label: "Finalize logical pool dispatches",
          layout: classifyPipelineLayout,
          compute: {
            module: classifyModule,
            entryPoint: "finalize_dispatch",
          },
        });
      const waterEntryPoints = [
        "build_hash",
        "solve_lambda",
        "solve_correction",
        "apply_correction",
        "reconstruct_velocity",
        "solve_viscosity",
        "apply_viscosity",
      ] as const;
      const waterPipelines = Object.fromEntries(
        waterEntryPoints.map((entryPoint) => [
          entryPoint,
          device.createComputePipeline({
            label: `Water pool ${entryPoint}`,
            layout: waterPipelineLayout,
            compute: {
              module: waterModule,
              entryPoint,
            },
          }),
        ]),
      ) as Record<(typeof waterEntryPoints)[number], any>;
      const cloudEntryPoints = [
        "build_hash",
        "solve_cloud_velocity",
        "apply_cloud_velocity",
      ] as const;
      const cloudPipelines = Object.fromEntries(
        cloudEntryPoints.map((entryPoint) => [
          entryPoint,
          device.createComputePipeline({
            label: `Cloud pool ${entryPoint}`,
            layout: cloudPipelineLayout,
            compute: {
              module: cloudModule,
              entryPoint,
            },
          }),
        ]),
      ) as Record<(typeof cloudEntryPoints)[number], any>;

      const canvasFormat = gpu.getPreferredCanvasFormat();
      const densityPipeline = device.createRenderPipeline({
        label: "Cloud metaball density",
        layout: densityPipelineLayout,
        vertex: {
          module: densityModule,
          entryPoint: "cloud_density_vertex",
        },
        fragment: {
          module: densityModule,
          entryPoint: "cloud_density_fragment",
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
        label: "Unified weather particles",
        layout: densityPipelineLayout,
        vertex: {
          module: densityModule,
          entryPoint: "cloud_density_vertex",
        },
        fragment: {
          module: densityModule,
          entryPoint: "unified_particle_fragment",
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
        "Horizontal cloud blur",
        "blurHorizontal",
      );
      const verticalBlurPipeline = createBlurPipeline(
        "Vertical cloud blur",
        "blurVertical",
      );
      const surfacePipeline = device.createRenderPipeline({
        label: "Cartoon cloud surface",
        layout: surfacePipelineLayout,
        vertex: {
          module: surfaceModule,
          entryPoint: "cloud_fullscreen_vertex",
        },
        fragment: {
          module: surfaceModule,
          entryPoint: "cloud_surface_fragment",
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

      const createBuffer = (
        label: string,
        byteLength: number,
        usage: number,
        initialData?: ArrayBufferView,
      ) => {
        const alignedSize = Math.max(
          4,
          Math.ceil(byteLength / 4) * 4,
        );
        const buffer = device.createBuffer({
          label,
          size: alignedSize,
          usage,
          mappedAtCreation: Boolean(initialData),
        });
        if (initialData) {
          const destination = new Uint8Array(
            buffer.getMappedRange(),
          );
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
        resourcesToDestroy: CloudResources,
      ) => {
        resourcesToDestroy.positions.destroy();
        resourcesToDestroy.previousPositions.destroy();
        resourcesToDestroy.velocities.destroy();
        resourcesToDestroy.states.destroy();
        resourcesToDestroy.lambdas.destroy();
        resourcesToDestroy.corrections.destroy();
        resourcesToDestroy.gridHeads.destroy();
        resourcesToDestroy.gridNext.destroy();
        resourcesToDestroy.waterActive.destroy();
        resourcesToDestroy.cloudActive.destroy();
        resourcesToDestroy.simulationUniform.destroy();
        resourcesToDestroy.renderUniform.destroy();
        resourcesToDestroy.blurUniform.destroy();
        resourcesToDestroy.densityTexture.destroy();
        resourcesToDestroy.horizontalBlurTexture.destroy();
        resourcesToDestroy.smoothDensityTexture.destroy();
      };

      const createInitialParticles = (
        width: number,
        height: number,
        waterParticleCount: number,
        weatherParticleCount: number,
        spacing: number,
        columns: number,
      ) => {
        const particleCount =
          waterParticleCount + weatherParticleCount;
        const positions = new Float32Array(particleCount * 2);
        const velocities = new Float32Array(particleCount * 2);
        const states = new Float32Array(particleCount * 4);
        const padding = spacing * 1.1;

        for (
          let index = 0;
          index < waterParticleCount;
          index += 1
        ) {
          const seed = hashValue(index + 1.37);
          const row = Math.floor(index / columns);
          const column = index % columns;
          const rowOffset =
            row % 2 === 0 ? 0 : spacing * 0.48;
          const positionOffset = index * 2;
          const stateOffset = index * 4;
          positions[positionOffset] =
            padding +
            column * spacing +
            rowOffset +
            (hashValue(index * 3.7) - 0.5) *
              spacing *
              0.018;
          positions[positionOffset + 1] =
            height -
            padding -
            row * spacing * 0.89 +
            (hashValue(index * 5.1) - 0.5) *
              spacing *
              0.018;
          states[stateOffset] = -1;
          states[stateOffset + 1] = 0;
          states[stateOffset + 2] = seed;
          states[stateOffset + 3] = 1;
        }

        for (
          let weatherIndex = 0;
          weatherIndex < weatherParticleCount;
          weatherIndex += 1
        ) {
          const index = waterParticleCount + weatherIndex;
          const seed = hashValue(weatherIndex + 1.37);
          const secondarySeed = hashValue(index * 2.31 + 7.2);
          const condensationStart = 0.9 + seed * 0.38;
          const condensationEnd = 3.15 + seed * 0.62;
          const rainStart = 6.2 + seed * 2.35;
          const age = secondarySeed * (rainStart + 1.15);
          const cloudPhase = smoothStep(
            condensationStart,
            condensationEnd,
            age,
          );
          const phase = age > rainStart ? 2 : cloudPhase;
          const positionOffset = index * 2;
          const stateOffset = index * 4;

          if (phase > 1.5) {
            const rainProgress = Math.max(
              0,
              Math.min(1, (age - rainStart) / 1.15),
            );
            positions[positionOffset] =
              width *
              (0.5 +
                (seed - 0.5) * 0.34 +
                Math.sin(weatherIndex * 0.019) * 0.025);
            positions[positionOffset + 1] =
              height * (0.23 + rainProgress * 0.34);
            velocities[positionOffset] =
              Math.sin(seed * 22) * 5;
            velocities[positionOffset + 1] =
              48 + rainProgress * 74;
          } else if (age < condensationEnd) {
            const rise = smoothStep(
              0,
              condensationEnd,
              age,
            );
            const spread = 0.18 + rise * 0.19;
            positions[positionOffset] =
              width *
              (0.5 +
                (seed - 0.5) * spread +
                Math.sin(age * 1.4 + seed * 9) * 0.035);
            positions[positionOffset + 1] =
              height *
                (0.565 - rise * 0.33) +
              (secondarySeed - 0.5) * height * 0.035;
            velocities[positionOffset] =
              Math.sin(seed * 18) * 8;
            velocities[positionOffset + 1] =
              -36 * (1 - cloudPhase);
          } else {
            const lobe =
              Math.min(2, Math.floor(secondarySeed * 3));
            positions[positionOffset] =
              width *
              (0.24 +
                lobe * 0.26 +
                (seed - 0.5) * 0.12 +
                Math.sin(weatherIndex * 0.017) * 0.025);
            positions[positionOffset + 1] =
              height *
              (0.17 +
                seed * 0.13 +
                Math.sin(
                  seed * 14 + weatherIndex * 0.01,
                ) *
                  0.026);
            velocities[positionOffset] =
              Math.sin(seed * 22) * 6;
            velocities[positionOffset + 1] = 0;
          }

          states[stateOffset] = phase;
          states[stateOffset + 1] = age;
          states[stateOffset + 2] = seed;
          states[stateOffset + 3] = 1;
        }

        return { positions, velocities, states };
      };

      let resources: CloudResources | undefined;
      const pointer: CloudPointer = {
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

      const createSimulationResources = (
        width: number,
        height: number,
      ): CloudResources => {
        const waterTargetCount =
          width < 700 ? 3200 : width < 1400 ? 6200 : 9000;
        const spacing = Math.max(
          10,
          Math.min(
            15,
            Math.sqrt(
              (width * height * 0.5) / waterTargetCount,
            ),
          ),
        );
        const padding = spacing * 1.1;
        const columns = Math.max(
          1,
          Math.floor((width - padding * 2) / spacing),
        );
        const maximumRows = Math.max(
          1,
          Math.floor(
            (height * 0.62) / (spacing * 0.89),
          ),
        );
        const rows = Math.min(
          maximumRows,
          Math.ceil(waterTargetCount / columns),
        );
        const waterParticleCount = Math.min(
          waterTargetCount,
          columns * rows,
        );
        const weatherParticleCount = Math.max(
          420,
          Math.round(waterParticleCount * 0.16),
        );
        const particleCount =
          waterParticleCount + weatherParticleCount;
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
          Math.ceil(
            (height + spacing * 4) / smoothingRadius,
          ),
        );
        const cloudRadius = spacing * 4.6;
        const cloudGridWidth = Math.max(
          1,
          Math.ceil(width / cloudRadius),
        );
        const cloudGridHeight = Math.max(
          1,
          Math.ceil(
            (height + spacing * 4) / cloudRadius,
          ),
        );
        const gridCellCount = Math.max(
          gridWidth * gridHeight,
          cloudGridWidth * cloudGridHeight,
        );
        const pixelRatio = Math.min(
          window.devicePixelRatio || 1,
          1.35,
        );
        canvas.width = Math.max(
          1,
          Math.round(width * pixelRatio),
        );
        canvas.height = Math.max(
          1,
          Math.round(height * pixelRatio),
        );
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

        const initialParticles = createInitialParticles(
          width,
          height,
          waterParticleCount,
          weatherParticleCount,
          spacing,
          columns,
        );
        const storageUsage =
          bufferUsage.STORAGE | bufferUsage.COPY_DST;
        const positions = createBuffer(
          "Cloud positions",
          initialParticles.positions.byteLength,
          storageUsage,
          initialParticles.positions,
        );
        const previousPositions = createBuffer(
          "Cloud previous positions",
          initialParticles.positions.byteLength,
          storageUsage,
          initialParticles.positions,
        );
        const velocities = createBuffer(
          "Cloud velocities",
          initialParticles.velocities.byteLength,
          storageUsage,
          initialParticles.velocities,
        );
        const states = createBuffer(
          "Cloud phase states",
          initialParticles.states.byteLength,
          storageUsage,
          initialParticles.states,
        );
        const lambdas = createBuffer(
          "Cloud lambdas",
          particleCount * Float32Array.BYTES_PER_ELEMENT,
          storageUsage,
        );
        const corrections = createBuffer(
          "Cloud corrections",
          initialParticles.positions.byteLength,
          storageUsage,
        );
        const gridHeads = createBuffer(
          "Cloud spatial hash heads",
          gridCellCount * Uint32Array.BYTES_PER_ELEMENT,
          storageUsage,
        );
        const gridNext = createBuffer(
          "Cloud spatial hash links",
          particleCount * Uint32Array.BYTES_PER_ELEMENT,
          storageUsage,
        );
        const activeListUsage =
          storageUsage | bufferUsage.INDIRECT;
        const activeListByteLength =
          4 * Uint32Array.BYTES_PER_ELEMENT +
          particleCount * Uint32Array.BYTES_PER_ELEMENT;
        const waterActive = createBuffer(
          "Water logical active list",
          activeListByteLength,
          activeListUsage,
        );
        const cloudActive = createBuffer(
          "Cloud logical active list",
          activeListByteLength,
          activeListUsage,
        );
        const simulationUniform = createBuffer(
          "Cloud simulation parameters",
          32 * Float32Array.BYTES_PER_ELEMENT,
          bufferUsage.UNIFORM | bufferUsage.COPY_DST,
        );
        const renderUniform = createBuffer(
          "Cloud render parameters",
          12 * Float32Array.BYTES_PER_ELEMENT,
          bufferUsage.UNIFORM | bufferUsage.COPY_DST,
        );
        const blurUniform = createBuffer(
          "Cloud blur parameters",
          4 * Float32Array.BYTES_PER_ELEMENT,
          bufferUsage.UNIFORM | bufferUsage.COPY_DST,
        );

        const createDensityTexture = (label: string) =>
          device.createTexture({
            label,
            size: {
              width: densityWidth,
              height: densityHeight,
            },
            format: "rgba16float",
            usage:
              textureUsage.RENDER_ATTACHMENT |
              textureUsage.TEXTURE_BINDING,
          });
        const densityTexture = createDensityTexture(
          "Cloud density texture",
        );
        const horizontalBlurTexture = createDensityTexture(
          "Cloud horizontal blur",
        );
        const smoothDensityTexture = createDensityTexture(
          "Cloud smooth density",
        );
        const densityView = densityTexture.createView();
        const horizontalBlurView =
          horizontalBlurTexture.createView();
        const smoothDensityView =
          smoothDensityTexture.createView();

        const integrateBindGroup = device.createBindGroup({
          label: "Shared weather integration bind group",
          layout: integrateBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: positions } },
            {
              binding: 1,
              resource: { buffer: previousPositions },
            },
            { binding: 2, resource: { buffer: velocities } },
            { binding: 3, resource: { buffer: states } },
            {
              binding: 4,
              resource: { buffer: simulationUniform },
            },
          ],
        });
        const classifyBindGroup = device.createBindGroup({
          label: "Logical weather pool bind group",
          layout: classifyBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: states } },
            {
              binding: 1,
              resource: { buffer: waterActive },
            },
            {
              binding: 2,
              resource: { buffer: cloudActive },
            },
            {
              binding: 3,
              resource: { buffer: simulationUniform },
            },
          ],
        });
        const waterBindGroup = device.createBindGroup({
          label: "Water PBF logical pool bind group",
          layout: waterBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: positions } },
            {
              binding: 1,
              resource: { buffer: previousPositions },
            },
            { binding: 2, resource: { buffer: velocities } },
            { binding: 3, resource: { buffer: lambdas } },
            { binding: 4, resource: { buffer: corrections } },
            { binding: 5, resource: { buffer: gridHeads } },
            { binding: 6, resource: { buffer: gridNext } },
            { binding: 7, resource: { buffer: waterActive } },
            {
              binding: 8,
              resource: { buffer: simulationUniform },
            },
          ],
        });
        const cloudBindGroup = device.createBindGroup({
          label: "Cloud cohesion logical pool bind group",
          layout: cloudBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: positions } },
            { binding: 1, resource: { buffer: velocities } },
            { binding: 2, resource: { buffer: states } },
            { binding: 3, resource: { buffer: corrections } },
            { binding: 4, resource: { buffer: gridHeads } },
            { binding: 5, resource: { buffer: gridNext } },
            { binding: 6, resource: { buffer: cloudActive } },
            {
              binding: 7,
              resource: { buffer: simulationUniform },
            },
          ],
        });
        const densityBindGroup = device.createBindGroup({
          label: "Cloud density bind group",
          layout: densityBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: positions } },
            { binding: 1, resource: { buffer: states } },
            {
              binding: 2,
              resource: { buffer: renderUniform },
            },
          ],
        });
        const horizontalBlurBindGroup = device.createBindGroup({
          label: "Cloud horizontal blur bind group",
          layout: blurBindGroupLayout,
          entries: [
            { binding: 0, resource: densityView },
            { binding: 1, resource: densitySampler },
            {
              binding: 2,
              resource: { buffer: blurUniform },
            },
          ],
        });
        const verticalBlurBindGroup = device.createBindGroup({
          label: "Cloud vertical blur bind group",
          layout: blurBindGroupLayout,
          entries: [
            { binding: 0, resource: horizontalBlurView },
            { binding: 1, resource: densitySampler },
            {
              binding: 2,
              resource: { buffer: blurUniform },
            },
          ],
        });
        const surfaceBindGroup = device.createBindGroup({
          label: "Cloud surface bind group",
          layout: surfaceBindGroupLayout,
          entries: [
            {
              binding: 0,
              resource: { buffer: renderUniform },
            },
            { binding: 1, resource: smoothDensityView },
            { binding: 2, resource: densitySampler },
          ],
        });

        return {
          positions,
          previousPositions,
          velocities,
          states,
          lambdas,
          corrections,
          gridHeads,
          gridNext,
          waterActive,
          cloudActive,
          simulationUniform,
          renderUniform,
          blurUniform,
          densityTexture,
          horizontalBlurTexture,
          smoothDensityTexture,
          densityView,
          horizontalBlurView,
          smoothDensityView,
          integrateBindGroup,
          classifyBindGroup,
          waterBindGroup,
          cloudBindGroup,
          densityBindGroup,
          horizontalBlurBindGroup,
          verticalBlurBindGroup,
          surfaceBindGroup,
          simulationParameters: new Float32Array(32),
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
          cloudRadius,
          cloudGridWidth,
          cloudGridHeight,
        };
      };

      const resize = () => {
        const bounds = canvas.getBoundingClientRect();
        const width = Math.max(1, bounds.width);
        const height = Math.max(1, bounds.height);
        const previousResources = resources;
        resources = createSimulationResources(width, height);
        pointer.x = width * 0.5;
        pointer.y = height * 0.62;
        pointer.previousX = pointer.x;
        pointer.previousY = pointer.y;
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
        if (speed > 4800) {
          const scale = 4800 / speed;
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
        if (
          event.target instanceof Element &&
          event.target.closest(
            ".simulationModeSwitch, .surfaceToggle, .surfaceControl",
          )
        ) {
          return;
        }
        updatePointer(event);
      };
      const pointerDown = (event: PointerEvent) => {
        if (
          event.target instanceof Element &&
          event.target.closest(
            ".simulationModeSwitch, .surfaceToggle, .surfaceControl",
          )
        ) {
          return;
        }
        updatePointer(event);
        pointer.burst = 1250;
      };
      const pointerUp = () => {
        pointer.burst = 0;
      };
      const pointerLeave = () => {
        pointer.active = false;
        pointer.vx = 0;
        pointer.vy = 0;
      };

      window.addEventListener("pointermove", pointerMove);
      window.addEventListener("pointerdown", pointerDown);
      window.addEventListener("pointerup", pointerUp);
      window.addEventListener("pointercancel", pointerUp);
      window.addEventListener("blur", pointerLeave);
      window.addEventListener("resize", queueResize);
      removeEventListeners = () => {
        window.removeEventListener("pointermove", pointerMove);
        window.removeEventListener("pointerdown", pointerDown);
        window.removeEventListener("pointerup", pointerUp);
        window.removeEventListener("pointercancel", pointerUp);
        window.removeEventListener("blur", pointerLeave);
        window.removeEventListener("resize", queueResize);
      };

      resize();

      const encodeSimulationStep = (
        encoder: any,
        activeResources: CloudResources,
      ) => {
        const integratePass = encoder.beginComputePass({
          label: "Integrate shared weather particles",
        });
        integratePass.setPipeline(integratePipeline);
        integratePass.setBindGroup(
          0,
          activeResources.integrateBindGroup,
        );
        integratePass.dispatchWorkgroups(
          activeResources.workgroupCount,
        );
        integratePass.end();

        encoder.clearBuffer(activeResources.waterActive, 0, 16);
        encoder.clearBuffer(activeResources.cloudActive, 0, 16);
        const classifyPass = encoder.beginComputePass({
          label: "Build logical water and cloud pools",
        });
        classifyPass.setBindGroup(
          0,
          activeResources.classifyBindGroup,
        );
        classifyPass.setPipeline(classifyPipeline);
        classifyPass.dispatchWorkgroups(
          activeResources.workgroupCount,
        );
        classifyPass.setPipeline(finalizeDispatchPipeline);
        classifyPass.dispatchWorkgroups(1);
        classifyPass.end();

        encoder.clearBuffer(activeResources.gridHeads);
        const waterPass = encoder.beginComputePass({
          label: "Water active pool PBF",
        });
        waterPass.setBindGroup(
          0,
          activeResources.waterBindGroup,
        );
        waterPass.setPipeline(waterPipelines.build_hash);
        waterPass.dispatchWorkgroupsIndirect(
          activeResources.waterActive,
          4,
        );
        for (
          let iteration = 0;
          iteration < CLOUD_SOLVER_ITERATIONS;
          iteration += 1
        ) {
          waterPass.setPipeline(
            waterPipelines.solve_lambda,
          );
          waterPass.dispatchWorkgroupsIndirect(
            activeResources.waterActive,
            4,
          );
          waterPass.setPipeline(
            waterPipelines.solve_correction,
          );
          waterPass.dispatchWorkgroupsIndirect(
            activeResources.waterActive,
            4,
          );
          waterPass.setPipeline(
            waterPipelines.apply_correction,
          );
          waterPass.dispatchWorkgroupsIndirect(
            activeResources.waterActive,
            4,
          );
        }
        waterPass.setPipeline(
          waterPipelines.reconstruct_velocity,
        );
        waterPass.dispatchWorkgroupsIndirect(
          activeResources.waterActive,
          4,
        );
        waterPass.setPipeline(
          waterPipelines.solve_viscosity,
        );
        waterPass.dispatchWorkgroupsIndirect(
          activeResources.waterActive,
          4,
        );
        waterPass.setPipeline(
          waterPipelines.apply_viscosity,
        );
        waterPass.dispatchWorkgroupsIndirect(
          activeResources.waterActive,
          4,
        );
        waterPass.end();

        encoder.clearBuffer(activeResources.gridHeads);
        const cloudPass = encoder.beginComputePass({
          label: "Cloud active pool cohesion",
        });
        cloudPass.setBindGroup(
          0,
          activeResources.cloudBindGroup,
        );
        cloudPass.setPipeline(cloudPipelines.build_hash);
        cloudPass.dispatchWorkgroupsIndirect(
          activeResources.cloudActive,
          4,
        );
        cloudPass.setPipeline(
          cloudPipelines.solve_cloud_velocity,
        );
        cloudPass.dispatchWorkgroupsIndirect(
          activeResources.cloudActive,
          4,
        );
        cloudPass.setPipeline(
          cloudPipelines.apply_cloud_velocity,
        );
        cloudPass.dispatchWorkgroupsIndirect(
          activeResources.cloudActive,
          4,
        );
        cloudPass.end();
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
        if (steps === 0) {
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
        simulationParameters[4] =
          activeResources.restDensity;
        simulationParameters[5] = 0.001;
        simulationParameters[6] =
          activeResources.particleCount;
        simulationParameters[7] = now * 0.001;
        simulationParameters[8] = activeResources.gridWidth;
        simulationParameters[9] = activeResources.gridHeight;
        simulationParameters[10] = 0;
        simulationParameters[11] = 1900;
        simulationParameters[12] = pointer.x;
        simulationParameters[13] = pointer.y;
        simulationParameters[14] = pointer.vx;
        simulationParameters[15] = pointer.vy;
        simulationParameters[16] = Math.max(
          88,
          activeResources.spacing * 8,
        );
        simulationParameters[17] = pointer.burst;
        simulationParameters[18] = pointer.active ? 1 : 0;
        simulationParameters[19] =
          activeResources.spacing * 0.56;
        simulationParameters[20] = 76;
        simulationParameters[21] = 980;
        simulationParameters[22] =
          sourceEnabledRef.current ? 1 : 0;
        simulationParameters[23] = 1.75;
        simulationParameters[24] =
          activeResources.cloudRadius;
        simulationParameters[25] =
          activeResources.cloudGridWidth;
        simulationParameters[26] =
          activeResources.cloudGridHeight;
        simulationParameters[27] =
          16 + cohesionRef.current * 40;
        simulationParameters[28] = 0.68;
        simulationParameters[29] =
          0.015 + cohesionRef.current * 0.07;
        simulationParameters[30] =
          activeResources.height * 0.575;
        simulationParameters[31] = 64;

        const renderParameters =
          activeResources.renderParameters;
        const surfaceMode = surfaceEnabledRef.current;
        renderParameters[0] = canvas.width;
        renderParameters[1] = canvas.height;
        renderParameters[2] = activeResources.densityWidth;
        renderParameters[3] = activeResources.densityHeight;
        renderParameters[4] = activeResources.width;
        renderParameters[5] = activeResources.height;
        renderParameters[6] =
          activeResources.spacing *
          (surfaceMode ? 2.02 : 0.46);
        renderParameters[7] =
          activeResources.spacing *
          (surfaceMode ? 3.05 : 0.72);
        renderParameters[8] = 0.24;
        renderParameters[9] = 0.075;
        renderParameters[10] =
          activeResources.particleCount;
        renderParameters[11] =
          activeResources.spacing *
          (surfaceMode ? 1.28 : 0.44);

        const blurParameters = activeResources.blurParameters;
        blurParameters[0] = 1.18;
        blurParameters[1] = 0;
        blurParameters[2] = 0;
        blurParameters[3] = 0;

        device.queue.writeBuffer(
          activeResources.simulationUniform,
          0,
          simulationParameters,
        );
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
          label: "DOTAFS particle cloud frame",
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
          horizontalBlurPass.setPipeline(
            horizontalBlurPipeline,
          );
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
              view: context
                .getCurrentTexture()
                .createView(),
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
                view: context
                  .getCurrentTexture()
                  .createView(),
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
        pointer.vx *= 0.67;
        pointer.vy *= 0.67;
        pointer.burst *= 0.46;
        if (Math.abs(pointer.vx) < 0.5) pointer.vx = 0;
        if (Math.abs(pointer.vy) < 0.5) pointer.vy = 0;
        if (pointer.burst < 1) pointer.burst = 0;
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
      console.warn("WebGPU particle cloud unavailable:", error);
      if (!disposed) onUnavailable();
    });

    return () => {
      disposed = true;
      releaseGpu();
    };
  }, [onUnavailable]);

  return (
    <div className="fluidStage smokeStage">
      <canvas
        ref={canvasRef}
        className="fluidCanvas smokeCanvas"
        aria-label="同一批 GPU PBF 粒子在水、蒸汽、云和雨状态间循环"
      />
      <div className="weatherControls">
        <button
          className="surfaceToggle"
          type="button"
          aria-pressed={sourceEnabled}
          onClick={toggleSource}
        >
          <span>WEATHER CYCLE</span>
          <strong>{sourceEnabled ? "ON" : "OFF"}</strong>
        </button>
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
        <label className="surfaceControl">
          <span>
            <b>CLOUD COHESION</b>
            <output>{cohesion}%</output>
          </span>
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={cohesion}
            onChange={updateCohesion}
            aria-label="云粒子凝聚强度"
          />
        </label>
      </div>
      <div
        className={`fluidStatus ${
          particleCount > 0 ? "isReady" : ""
        }`}
      >
        <span>GPU DUAL-POOL PBF</span>
        <strong>
          {particleCount > 0
              ? `${particleCount.toLocaleString()} SHARED IDS`
            : "REQUESTING GPU"}
        </strong>
      </div>
    </div>
  );
}

export function SmokeCanvas() {
  const [unavailable, setUnavailable] = useState(false);
  const showUnavailable = useCallback(
    () => setUnavailable(true),
    [],
  );

  if (unavailable) {
    return (
      <div className="smokeUnavailable">
        <div className="fallbackBadge" role="status">
          GPU CLOUD / WEBGPU UNAVAILABLE
        </div>
      </div>
    );
  }

  return <GpuCloudCanvas onUnavailable={showUnavailable} />;
}
