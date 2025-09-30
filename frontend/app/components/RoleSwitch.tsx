"use client";
import { useId } from "react";

type RoleSwitchProps = {
  value: string;
  onChange: (role: string) => void;
  roles?: string[];
  disabled?: boolean;
};

export default function RoleSwitch({
  value,
  onChange,
  roles = ["owner", "admin"],
  disabled = false,
}: RoleSwitchProps) {
  const selectId = useId();

  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
      <label htmlFor={selectId}>Role:</label>
      <select
        id={selectId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        {roles.map((roleOption) => (
          <option key={roleOption} value={roleOption}>
            {roleOption}
          </option>
        ))}
      </select>
    </div>
  );
}
