"use client";
import { useEffect } from "react";

import type { AgentSummary } from "../hooks/useAgents";

type AgentPickerProps = {
  value: string;
  onChange: (value: string) => void;
  agents: AgentSummary[];
  loading: boolean;
  allowAutoRoute?: boolean;
};

export function AgentPicker({
  value,
  onChange,
  agents,
  loading,
  allowAutoRoute = true,
}: AgentPickerProps) {
  useEffect(() => {
    if (!value) {
      return;
    }
    const found = agents.some((agent) => agent.slug === value);
    if (!found) {
      onChange("");
    }
  }, [agents, onChange, value]);

  const disabled = loading || (!allowAutoRoute && agents.length === 0);

  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      disabled={disabled}
      title="Select the agent that should receive the request"
    >
      {allowAutoRoute && <option value="">auto-route</option>}
      {agents.map((agent) => (
        <option key={agent.slug} value={agent.slug}>
          {agent.title || agent.slug}
        </option>
      ))}
    </select>
  );
}
