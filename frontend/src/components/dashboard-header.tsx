"use client";

import { Flask, Heartbeat } from "@phosphor-icons/react";

export function DashboardHeader() {
  return (
    <header className="border-b border-border bg-surface/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[1600px] items-center justify-between px-4 md:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-accent-muted text-accent">
            <Heartbeat size={20} weight="duotone" />
          </div>
          <div>
            <p className="text-sm font-semibold tracking-tight">CMCHT-XAI</p>
            <p className="hidden text-xs text-muted sm:block">
              Liver disease intelligence
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-border bg-surface-elevated px-3 py-1.5 text-xs text-muted">
          <Flask size={14} weight="duotone" className="text-accent" />
          Research preview
        </div>
      </div>
    </header>
  );
}
