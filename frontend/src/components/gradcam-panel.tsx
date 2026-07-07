"use client";

import { Scan } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";

interface GradCamPanelProps {
  imageB64: string | null;
}

export function GradCamPanel({ imageB64 }: GradCamPanelProps) {
  const reduce = useReducedMotion();

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="panel overflow-hidden"
    >
      <div className="border-b border-border px-5 py-4">
        <div className="flex items-center gap-2">
          <Scan size={18} weight="duotone" className="text-accent" />
          <h3 className="text-sm font-medium">Grad-CAM overlay</h3>
        </div>
        <p className="mt-1 text-xs text-muted">
          Regions that drove the detection prediction
        </p>
      </div>
      <div className="relative aspect-square w-full bg-surface-elevated md:aspect-[4/3]">
        {imageB64 ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/png;base64,${imageB64}`}
            alt="Grad-CAM heatmap overlay on liver scan"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full min-h-64 items-center justify-center p-8 text-center text-sm text-muted">
            Grad-CAM visualization unavailable for this sample.
          </div>
        )}
      </div>
    </motion.div>
  );
}
