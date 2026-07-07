"use client";

import { Warning, CheckCircle } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { cn } from "@/lib/utils";

interface UncertaintyBannerProps {
  variance: number;
  threshold: number;
  needsReview: boolean;
}

export function UncertaintyBanner({
  variance,
  threshold,
  needsReview,
}: UncertaintyBannerProps) {
  const reduce = useReducedMotion();

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "panel flex items-start gap-4 p-5 md:p-6",
        needsReview
          ? "border-warning/30 bg-warning/5"
          : "border-success/30 bg-success/5",
      )}
    >
      <div
        className={cn(
          "rounded-xl p-3",
          needsReview ? "bg-warning/15 text-warning" : "bg-success/15 text-success",
        )}
      >
        {needsReview ? (
          <Warning size={24} weight="duotone" />
        ) : (
          <CheckCircle size={24} weight="duotone" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <h3 className="text-base font-medium">
          {needsReview ? "Flagged for human review" : "Confident prediction"}
        </h3>
        <p className="mt-1 text-sm leading-relaxed text-muted">
          MC-Dropout variance is{" "}
          <span className="font-mono text-foreground">{variance.toFixed(4)}</span>
          {" "}(threshold {threshold.toFixed(2)}).
          {needsReview
            ? " Route this case to a clinician before acting on the output."
            : " Model uncertainty is within acceptable bounds for this sample."}
        </p>
      </div>
    </motion.div>
  );
}
