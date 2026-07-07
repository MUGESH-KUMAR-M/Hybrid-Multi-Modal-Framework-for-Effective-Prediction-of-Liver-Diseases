"use client";

import { ArrowCounterClockwise } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "motion/react";

interface CounterfactualListProps {
  items: string[];
}

export function CounterfactualList({ items }: CounterfactualListProps) {
  const reduce = useReducedMotion();

  if (!items.length) {
    return (
      <div className="panel flex h-40 items-center justify-center p-6 text-sm text-muted">
        No counterfactual explanations generated.
      </div>
    );
  }

  return (
    <div className="grid gap-4">
      {items.map((text, i) => (
        <motion.div
          key={i}
          initial={reduce ? false : { opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.4, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
          className="panel flex gap-4 p-5"
        >
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent-muted text-accent">
            <ArrowCounterClockwise size={20} weight="duotone" />
          </div>
          <div>
            <p className="text-xs font-mono uppercase tracking-wider text-accent">
              Scenario {i + 1}
            </p>
            <p className="mt-2 text-sm leading-relaxed text-foreground/90">
              {text}
            </p>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
