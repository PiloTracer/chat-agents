"use client";
import { useMemo } from "react";

import type { PrincipalSummary } from "../hooks/useDemoPrincipals";

interface PrincipalPickerProps {
  principals: PrincipalSummary[];
  value: string;
  onChange: (principal: PrincipalSummary) => void;
  loading: boolean;
  disabled?: boolean;
}

export function PrincipalPicker({
  principals,
  value,
  onChange,
  loading,
  disabled = false,
}: PrincipalPickerProps) {
  const effectiveDisabled = disabled || loading || principals.length === 0;

  const options = useMemo(() => principals, [principals]);

  return (
    <div style={{ display: "grid", gap: 4 }}>
      <label>
        Acting user:
        <select
          value={value}
          onChange={(event) => {
            const selected = options.find(
              (principal) => principal.username === event.target.value,
            );
            if (selected) {
              onChange(selected);
            }
          }}
          disabled={effectiveDisabled}
          style={{ marginLeft: 8 }}
        >
          {options.length === 0 && <option value="">No demo users</option>}
          {options.map((principal) => (
            <option key={principal.username} value={principal.username}>
              {principal.displayName} ({principal.role})
            </option>
          ))}
        </select>
      </label>
      {loading && <span style={{ fontSize: 12 }}>Loading users...</span>}
    </div>
  );
}
