"use client";

import { useEffect, useRef, useState } from "react";

type WorkerFrame = {
  type: "frame";
  buffer: ArrayBuffer;
  count: number;
  radius: number;
  generation: number;
};

type WorkerReady = {
  type: "ready";
  count: number;
  generation: number;
};

export function CpuFluidCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const surfaceEnabledRef = useRef(false);
  const [particleCount, setParticleCount] = useState(0);
  const [supported, setSupported] = useState(true);
  const [surfaceEnabled, setSurfaceEnabled] = useState(false);

  const toggleSurface = () => {
    setSurfaceEnabled((currentValue) => {
      const nextValue = !currentValue;
      surfaceEnabledRef.current = nextValue;
      return nextValue;
    });
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof Worker === "undefined") {
      setSupported(false);
      return;
    }

    const context = canvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    });
    if (!context) {
      setSupported(false);
      return;
    }

    const worker = new Worker("/pbf-worker.js");
    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;

    let canvasWidth = 1;
    let canvasHeight = 1;
    let pixelRatio = 1;
    let resizeFrame = 0;
    let pointerFrame = 0;
    let pendingPointer:
      | {
          x: number;
          y: number;
          vx: number;
          vy: number;
          active: boolean;
          burst: number;
        }
      | undefined;
    let previousPointer:
      | {
          x: number;
          y: number;
          time: number;
        }
      | undefined;
    let densityField = new Float32Array(0);
    let fieldColumns = 0;
    let fieldRows = 0;
    let fieldCellSize = 11;
    let surfaceFrame = 0;
    let cachedSurfacePath: Path2D | undefined;
    let cachedContourPath: Path2D | undefined;
    let previousSurfaceMode = false;

    type Point = readonly [number, number];

    const addPolygon = (path: Path2D, points: Point[]) => {
      if (points.length < 3) return;
      path.moveTo(points[0][0], points[0][1]);
      for (let index = 1; index < points.length; index += 1) {
        path.lineTo(points[index][0], points[index][1]);
      }
      path.closePath();
    };

    const addSegment = (
      path: Path2D,
      from: Point,
      to: Point,
    ) => {
      path.moveTo(from[0], from[1]);
      path.lineTo(to[0], to[1]);
    };

    const interpolate = (
      from: Point,
      to: Point,
      fromValue: number,
      toValue: number,
      isoLevel: number,
    ): Point => {
      const difference = toValue - fromValue;
      const amount =
        Math.abs(difference) < 0.00001
          ? 0.5
          : Math.max(
              0,
              Math.min(1, (isoLevel - fromValue) / difference),
            );
      return [
        from[0] + (to[0] - from[0]) * amount,
        from[1] + (to[1] - from[1]) * amount,
      ];
    };

    const buildDensityField = (
      positions: Float32Array,
      frame: WorkerFrame,
    ) => {
      fieldCellSize = Math.max(
        10,
        Math.min(15, Math.round(frame.radius * 2.35)),
      );
      const nextColumns = Math.ceil(canvasWidth / fieldCellSize);
      const nextRows = Math.ceil(canvasHeight / fieldCellSize);
      const requiredLength = (nextColumns + 1) * (nextRows + 1);
      if (
        nextColumns !== fieldColumns ||
        nextRows !== fieldRows ||
        densityField.length !== requiredLength
      ) {
        fieldColumns = nextColumns;
        fieldRows = nextRows;
        densityField = new Float32Array(requiredLength);
      } else {
        densityField.fill(0);
      }

      const stride = fieldColumns + 1;
      const influenceRadius = frame.radius * 3.55;
      const influenceSquared = influenceRadius * influenceRadius;

      for (let index = 0; index < frame.count; index += 1) {
        const offset = index * 3;
        const particleX = positions[offset];
        const particleY = positions[offset + 1];
        const minimumColumn = Math.max(
          0,
          Math.floor((particleX - influenceRadius) / fieldCellSize),
        );
        const maximumColumn = Math.min(
          fieldColumns,
          Math.ceil((particleX + influenceRadius) / fieldCellSize),
        );
        const minimumRow = Math.max(
          0,
          Math.floor((particleY - influenceRadius) / fieldCellSize),
        );
        const maximumRow = Math.min(
          fieldRows,
          Math.ceil((particleY + influenceRadius) / fieldCellSize),
        );

        for (let row = minimumRow; row <= maximumRow; row += 1) {
          const differenceY = row * fieldCellSize - particleY;
          const differenceYSquared = differenceY * differenceY;
          const rowOffset = row * stride;
          for (
            let column = minimumColumn;
            column <= maximumColumn;
            column += 1
          ) {
            const differenceX =
              column * fieldCellSize - particleX;
            const distanceSquared =
              differenceX * differenceX + differenceYSquared;
            if (distanceSquared >= influenceSquared) continue;
            const falloff = 1 - distanceSquared / influenceSquared;
            densityField[rowOffset + column] += falloff * falloff;
          }
        }
      }
    };

    const buildSurfacePaths = () => {
      const surfacePath = new Path2D();
      const contourPath = new Path2D();
      const isoLevel = 0.42;
      const stride = fieldColumns + 1;

      for (let row = 0; row < fieldRows; row += 1) {
        const topRow = row * stride;
        const bottomRow = (row + 1) * stride;
        const y0 = row * fieldCellSize;
        const y1 = (row + 1) * fieldCellSize;
        let solidRunStart = -1;

        for (
          let column = 0;
          column < fieldColumns;
          column += 1
        ) {
          const x0 = column * fieldCellSize;
          const x1 = (column + 1) * fieldCellSize;
          const topLeftValue = densityField[topRow + column];
          const topRightValue =
            densityField[topRow + column + 1];
          const bottomRightValue =
            densityField[bottomRow + column + 1];
          const bottomLeftValue =
            densityField[bottomRow + column];

          let cellCase = 0;
          if (topLeftValue >= isoLevel) cellCase |= 1;
          if (topRightValue >= isoLevel) cellCase |= 2;
          if (bottomRightValue >= isoLevel) cellCase |= 4;
          if (bottomLeftValue >= isoLevel) cellCase |= 8;

          if (cellCase === 15) {
            if (solidRunStart < 0) solidRunStart = x0;
            continue;
          }
          if (solidRunStart >= 0) {
            surfacePath.rect(
              solidRunStart,
              y0,
              x0 - solidRunStart,
              fieldCellSize,
            );
            solidRunStart = -1;
          }
          if (cellCase === 0) continue;

          const topLeft: Point = [x0, y0];
          const topRight: Point = [x1, y0];
          const bottomRight: Point = [x1, y1];
          const bottomLeft: Point = [x0, y1];
          const top = interpolate(
            topLeft,
            topRight,
            topLeftValue,
            topRightValue,
            isoLevel,
          );
          const right = interpolate(
            topRight,
            bottomRight,
            topRightValue,
            bottomRightValue,
            isoLevel,
          );
          const bottom = interpolate(
            bottomRight,
            bottomLeft,
            bottomRightValue,
            bottomLeftValue,
            isoLevel,
          );
          const left = interpolate(
            bottomLeft,
            topLeft,
            bottomLeftValue,
            topLeftValue,
            isoLevel,
          );

          switch (cellCase) {
            case 1:
              addPolygon(surfacePath, [topLeft, top, left]);
              addSegment(contourPath, left, top);
              break;
            case 2:
              addPolygon(surfacePath, [topRight, right, top]);
              addSegment(contourPath, top, right);
              break;
            case 3:
              addPolygon(surfacePath, [
                topLeft,
                topRight,
                right,
                left,
              ]);
              addSegment(contourPath, left, right);
              break;
            case 4:
              addPolygon(surfacePath, [
                bottomRight,
                bottom,
                right,
              ]);
              addSegment(contourPath, right, bottom);
              break;
            case 5:
              addPolygon(surfacePath, [topLeft, top, left]);
              addPolygon(surfacePath, [
                bottomRight,
                bottom,
                right,
              ]);
              addSegment(contourPath, left, top);
              addSegment(contourPath, right, bottom);
              break;
            case 6:
              addPolygon(surfacePath, [
                topRight,
                bottomRight,
                bottom,
                top,
              ]);
              addSegment(contourPath, top, bottom);
              break;
            case 7:
              addPolygon(surfacePath, [
                topLeft,
                topRight,
                bottomRight,
                bottom,
                left,
              ]);
              addSegment(contourPath, left, bottom);
              break;
            case 8:
              addPolygon(surfacePath, [bottomLeft, left, bottom]);
              addSegment(contourPath, bottom, left);
              break;
            case 9:
              addPolygon(surfacePath, [
                topLeft,
                top,
                bottom,
                bottomLeft,
              ]);
              addSegment(contourPath, top, bottom);
              break;
            case 10:
              addPolygon(surfacePath, [topRight, right, top]);
              addPolygon(surfacePath, [bottomLeft, left, bottom]);
              addSegment(contourPath, top, right);
              addSegment(contourPath, bottom, left);
              break;
            case 11:
              addPolygon(surfacePath, [
                topLeft,
                topRight,
                right,
                bottom,
                bottomLeft,
              ]);
              addSegment(contourPath, right, bottom);
              break;
            case 12:
              addPolygon(surfacePath, [
                left,
                right,
                bottomRight,
                bottomLeft,
              ]);
              addSegment(contourPath, left, right);
              break;
            case 13:
              addPolygon(surfacePath, [
                topLeft,
                top,
                right,
                bottomRight,
                bottomLeft,
              ]);
              addSegment(contourPath, top, right);
              break;
            case 14:
              addPolygon(surfacePath, [
                top,
                topRight,
                bottomRight,
                bottomLeft,
                left,
              ]);
              addSegment(contourPath, left, top);
              break;
          }
        }

        if (solidRunStart >= 0) {
          surfacePath.rect(
            solidRunStart,
            y0,
            fieldColumns * fieldCellSize - solidRunStart,
            fieldCellSize,
          );
        }
      }

      return { surfacePath, contourPath };
    };

    const drawFrame = (frame: WorkerFrame) => {
      const positions = new Float32Array(frame.buffer);
      const surfaceMode = surfaceEnabledRef.current;

      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.globalCompositeOperation = "source-over";
      context.fillStyle = "rgba(3, 7, 10, 0.58)";
      context.fillRect(0, 0, canvasWidth, canvasHeight);

      if (surfaceMode) {
        if (!previousSurfaceMode) {
          cachedSurfacePath = undefined;
          cachedContourPath = undefined;
          surfaceFrame = 0;
        }
        surfaceFrame += 1;
        if (
          surfaceFrame % 2 === 1 ||
          !cachedSurfacePath ||
          !cachedContourPath
        ) {
          buildDensityField(positions, frame);
          const nextPaths = buildSurfacePaths();
          cachedSurfacePath = nextPaths.surfacePath;
          cachedContourPath = nextPaths.contourPath;
        }

        const waterGradient = context.createLinearGradient(
          0,
          0,
          0,
          canvasHeight,
        );
        waterGradient.addColorStop(0, "rgba(53, 193, 246, 0.72)");
        waterGradient.addColorStop(0.55, "rgba(7, 115, 194, 0.88)");
        waterGradient.addColorStop(1, "rgba(3, 48, 112, 0.96)");
        context.fillStyle = waterGradient;
        context.fill(cachedSurfacePath);

        context.lineWidth = Math.max(0.8, frame.radius * 0.19);
        context.lineCap = "round";
        context.lineJoin = "round";
        context.strokeStyle = "rgba(169, 237, 255, 0.7)";
        context.stroke(cachedContourPath);

        context.globalCompositeOperation = "screen";
        context.beginPath();
        for (let index = 0; index < frame.count; index += 1) {
          const offset = index * 3;
          const speed = positions[offset + 2];
          if (speed < 13) continue;
          const radius =
            frame.radius * Math.min(0.62, 0.25 + speed * 0.004);
          context.moveTo(
            positions[offset] + radius,
            positions[offset + 1],
          );
          context.arc(
            positions[offset],
            positions[offset + 1],
            radius,
            0,
            Math.PI * 2,
          );
        }
        context.fillStyle = "rgba(195, 244, 255, 0.62)";
        context.fill();
      } else {
        context.globalCompositeOperation = "lighter";
        context.beginPath();
        for (let index = 0; index < frame.count; index += 1) {
          const offset = index * 3;
          const speed = Math.min(
            positions[offset + 2] * 0.012,
            0.7,
          );
          const radius = frame.radius * (1 + speed * 0.5);
          context.moveTo(
            positions[offset] + radius,
            positions[offset + 1],
          );
          context.arc(
            positions[offset],
            positions[offset + 1],
            radius,
            0,
            Math.PI * 2,
          );
        }
        context.fillStyle = "rgba(15, 117, 205, 0.3)";
        context.fill();

        context.beginPath();
        for (let index = 0; index < frame.count; index += 1) {
          const offset = index * 3;
          const radius = frame.radius * 0.42;
          context.moveTo(
            positions[offset] + radius,
            positions[offset + 1],
          );
          context.arc(
            positions[offset],
            positions[offset + 1],
            radius,
            0,
            Math.PI * 2,
          );
        }
        context.fillStyle = "rgba(132, 220, 255, 0.52)";
        context.fill();
      }
      previousSurfaceMode = surfaceMode;
      context.globalCompositeOperation = "source-over";

      worker.postMessage(
        {
          type: "recycle",
          buffer: frame.buffer,
          generation: frame.generation,
        },
        [frame.buffer],
      );
    };

    worker.onmessage = (
      event: MessageEvent<WorkerFrame | WorkerReady>,
    ) => {
      if (event.data.type === "frame") {
        drawFrame(event.data);
        return;
      }
      if (event.data.type === "ready") {
        setParticleCount(event.data.count);
      }
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      canvasWidth = Math.max(1, rect.width);
      canvasHeight = Math.max(1, rect.height);
      pixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
      canvas.width = Math.round(canvasWidth * pixelRatio);
      canvas.height = Math.round(canvasHeight * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.fillStyle = "#03070a";
      context.fillRect(0, 0, canvasWidth, canvasHeight);
      worker.postMessage({
        type: "resize",
        width: canvasWidth,
        height: canvasHeight,
        reducedMotion,
      });
    };

    const queueResize = () => {
      cancelAnimationFrame(resizeFrame);
      resizeFrame = requestAnimationFrame(resize);
    };

    const flushPointer = () => {
      pointerFrame = 0;
      if (!pendingPointer) return;
      worker.postMessage({ type: "pointer", ...pendingPointer });
      pendingPointer = undefined;
    };

    const queuePointer = (
      event: PointerEvent,
      burst = 0,
    ) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const time = performance.now();
      const elapsed = Math.max(
        8,
        Math.min(64, time - (previousPointer?.time ?? time - 16)),
      );
      let vx = previousPointer
        ? ((x - previousPointer.x) / elapsed) * 1000
        : 0;
      let vy = previousPointer
        ? ((y - previousPointer.y) / elapsed) * 1000
        : 0;
      const speed = Math.hypot(vx, vy);
      if (speed > 4200) {
        const scale = 4200 / speed;
        vx *= scale;
        vy *= scale;
      }
      previousPointer = { x, y, time };
      pendingPointer = {
        x,
        y,
        vx,
        vy,
        active: true,
        burst,
      };
      if (burst > 0) {
        cancelAnimationFrame(pointerFrame);
        flushPointer();
      } else if (!pointerFrame) {
        pointerFrame = requestAnimationFrame(flushPointer);
      }
    };

    const pointerMove = (event: PointerEvent) => queuePointer(event);
    const pointerDown = (event: PointerEvent) => {
      canvas.setPointerCapture(event.pointerId);
      queuePointer(event, 26);
    };
    const pointerUp = (event: PointerEvent) => {
      if (canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
      previousPointer = undefined;
      worker.postMessage({ type: "pointer", active: false });
    };
    const pointerLeave = () => {
      previousPointer = undefined;
      worker.postMessage({ type: "pointer", active: false });
    };
    const visibilityChange = () => {
      worker.postMessage({
        type: "visibility",
        paused: document.hidden,
      });
    };

    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);
    canvas.addEventListener("pointerleave", pointerLeave);
    window.addEventListener("resize", queueResize);
    document.addEventListener("visibilitychange", visibilityChange);

    worker.postMessage({ type: "init" });
    resize();

    return () => {
      cancelAnimationFrame(resizeFrame);
      cancelAnimationFrame(pointerFrame);
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("pointercancel", pointerUp);
      canvas.removeEventListener("pointerleave", pointerLeave);
      window.removeEventListener("resize", queueResize);
      document.removeEventListener("visibilitychange", visibilityChange);
      worker.terminate();
    };
  }, []);

  return (
    <div className="fluidStage">
      <canvas
        ref={canvasRef}
        className="fluidCanvas"
        aria-label="可通过鼠标推动的二维 PBF 水粒子模拟"
      />
      <button
        className="surfaceToggle"
        type="button"
        aria-pressed={surfaceEnabled}
        onClick={toggleSurface}
      >
        <span>SURFACE</span>
        <strong>{surfaceEnabled ? "ON" : "OFF"}</strong>
      </button>
      <div className={`fluidStatus ${particleCount > 0 ? "isReady" : ""}`}>
        <span>{supported ? "PBF SOLVER" : "CANVAS UNAVAILABLE"}</span>
        <strong>
          {particleCount > 0
            ? `${particleCount.toLocaleString()} PARTICLES`
            : "INITIALIZING"}
        </strong>
      </div>
    </div>
  );
}
