"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { motion, useReducedMotion } from "motion/react";
import type { ShapItem } from "@/lib/types";

interface ShapChartProps {
  data: ShapItem[];
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: ShapItem }>;
}) {
  if (!active || !payload?.length) return null;
  const item = payload[0].payload;
  return (
    <div className="rounded-lg border border-border bg-surface-elevated px-3 py-2 text-sm shadow-xl">
      <p className="font-medium">{item.feature.replace(/_/g, " ")}</p>
      <p className="mt-1 font-mono text-accent">
        {item.value >= 0 ? "+" : ""}
        {item.value.toFixed(4)}
      </p>
    </div>
  );
}

export function ShapChart({ data }: ShapChartProps) {
  const reduce = useReducedMotion();
  const chartData = [...data].reverse().map((d) => ({
    ...d,
    label: d.feature.replace(/_/g, " "),
  }));

  if (!chartData.length) {
    return (
      <div className="panel flex h-72 items-center justify-center p-6 text-sm text-muted">
        SHAP values unavailable for this sample.
      </div>
    );
  }

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="panel p-4 md:p-6"
    >
      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            type="number"
            tick={{ fill: "#8b95a5", fontSize: 11 }}
            axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={120}
            tick={{ fill: "#8b95a5", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={14}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.value >= 0 ? "#14b8a6" : "#f87171"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
