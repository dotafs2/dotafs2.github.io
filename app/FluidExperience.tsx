"use client";

import { useState } from "react";
import { FluidCanvas } from "./FluidCanvas";
import { SmokeCanvas } from "./SmokeCanvas";

type FluidMode = "water" | "smoke";

const interfaceCopy = {
  water: {
    topRight: ["WEBGPU PBF / 2D", "GPU SPATIAL HASH"],
    bottomLeft: ["MOVE FAST", "BREAK THE SURFACE"],
    bottomRight: [
      "GPU METABALL SURFACE",
      "KURO PBF / WGSL PORT",
    ],
  },
  smoke: {
    topRight: [
      "LOGICAL DUAL POOL / 2D",
      "SHARED PARTICLE IDENTITY",
    ],
    bottomLeft: ["STIR THE WATER", "BUILD A RAIN CLOUD"],
    bottomRight: [
      "EVAPORATE / CONDENSE / RAIN",
      "RAIN RETURNS TO WATER",
    ],
  },
} as const;

export function FluidExperience() {
  const [mode, setMode] = useState<FluidMode>("water");
  const copy = interfaceCopy[mode];

  return (
    <>
      {mode === "water" ? (
        <FluidCanvas />
      ) : (
        <SmokeCanvas />
      )}

      <div
        className="simulationModeSwitch"
        role="group"
        aria-label="选择二维流体模拟"
      >
        <button
          type="button"
          aria-pressed={mode === "water"}
          onClick={() => setMode("water")}
        >
          WATER
        </button>
        <button
          type="button"
          aria-pressed={mode === "smoke"}
          onClick={() => setMode("smoke")}
        >
          SMOKE
        </button>
      </div>

      <div className="interfaceLayer" aria-hidden="true">
        <div className="corner cornerTopLeft">
          <strong>DOTAFS</strong>
          <span>GPU FLUID LAB</span>
        </div>
        <div className="corner cornerTopRight">
          <span>{copy.topRight[0]}</span>
          <span>{copy.topRight[1]}</span>
        </div>
        <div className="corner cornerBottomLeft">
          <span>{copy.bottomLeft[0]}</span>
          <strong>{copy.bottomLeft[1]}</strong>
        </div>
        <div className="corner cornerBottomRight">
          <span>{copy.bottomRight[0]}</span>
          <span>{copy.bottomRight[1]}</span>
        </div>
      </div>
    </>
  );
}
