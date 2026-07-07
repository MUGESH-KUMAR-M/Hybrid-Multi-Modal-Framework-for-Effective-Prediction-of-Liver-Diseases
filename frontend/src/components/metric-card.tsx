"use client";

import { ChartLineUp, Brain, Pulse, ShieldWarning } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string;
  sublabel?: string;
  tone?: "default" | "success" | "warning" | "danger";
  icon: "chart" | "brain" | "pulse" | "shield";
  delay?: number;
}

const icons = {
  chart: ChartLineUp,
  brain: Brain,
  pulse: Pulse,
  shield: ShieldWarning,
};

const toneStyles = {
  default: "text-foreground",
  success: "text-success",
  warning: "text-warning",
  danger: "text-danger",
};

export function MetricCard({
  label,
  value,
  sublabel,
  tone = "default",
  icon,
  delay = 0,
}: MetricCardProps) {
  const reduce = useReducedMotion();
  const Icon = icons[icon];

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.16, 1, 0.3, 1] }}
      className="panel metric-glow p-5 md:p-6"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm text-muted">{label}</p>
          <p
            className={cn(
              "mt-2 font-mono text-3xl font-medium tracking-tight md:text-4xl",
              toneStyles[tone],
            )}
          >
            {value}
          </p>
          {sublabel && (
            <p className="mt-1 text-sm text-muted">{sublabel}</p>
          )}
        </div>
        <div className="rounded-xl bg-accent-muted p-3 text-accent">
          <Icon size={22} weight="duotone" />
        </div>
      </div>
    </motion.div>
  );
}
