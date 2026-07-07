"use client";

import { UserCircle, CaretRight } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import type { PatientMeta } from "@/lib/types";

interface PatientSidebarProps {
  patients: PatientMeta[];
  selectedId: number;
  features: Record<string, number>;
  onSelect: (id: number) => void;
  loading?: boolean;
}

export function PatientSidebar({
  patients,
  selectedId,
  features,
  onSelect,
  loading,
}: PatientSidebarProps) {
  return (
    <aside className="panel flex h-full flex-col overflow-hidden">
      <div className="border-b border-border p-5">
        <h2 className="text-sm font-medium">Patient cohort</h2>
        <p className="mt-1 text-xs text-muted">
          {patients.length} patients
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        <ul className="space-y-1">
          {patients.map((p) => {
            const active = p.id === selectedId;
            return (
              <li key={p.id}>
                <button
                  type="button"
                  disabled={loading}
                  onClick={() => onSelect(p.id)}
                  className={cn(
                    "group flex w-full items-center gap-3 rounded-xl px-3 py-3 text-left transition-colors active:scale-[0.98]",
                    active
                      ? "bg-accent-muted text-accent"
                      : "text-muted hover:bg-white/[0.03] hover:text-foreground",
                    loading && "opacity-60",
                  )}
                >
                  <UserCircle
                    size={28}
                    weight={active ? "duotone" : "regular"}
                    className="shrink-0"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium">Patient {p.id + 1}</p>
                    <p className="truncate text-xs opacity-80">{p.stage}</p>
                  </div>
                  <CaretRight
                    size={14}
                    className={cn(
                      "shrink-0 opacity-0 transition-opacity",
                      active && "opacity-100",
                    )}
                  />
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="border-t border-border p-5">
        <h3 className="text-xs font-mono uppercase tracking-wider text-muted">
          Clinical features
        </h3>
        <dl className="mt-3 max-h-48 space-y-2 overflow-y-auto text-xs">
          {Object.entries(features).map(([name, val]) => (
            <div key={name} className="flex justify-between gap-2">
              <dt className="truncate text-muted">{name.replace(/_/g, " ")}</dt>
              <dd className="shrink-0 font-mono text-foreground">
                {val.toFixed(3)}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </aside>
  );
}
