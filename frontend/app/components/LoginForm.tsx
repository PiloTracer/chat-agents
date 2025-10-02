"use client";
import { FormEvent, useState } from "react";

type LoginFormProps = {
  onSubmit: (username: string, password: string) => Promise<boolean>;
  loading: boolean;
  error: string | null;
  title?: string;
};

export function LoginForm({ onSubmit, loading, error, title }: LoginFormProps) {
  const [username, setUsername] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!username || !password) {
      setLocalError("Provide both username and password.");
      return;
    }
    setLocalError(null);
    setSubmitting(true);
    try {
      await onSubmit(username, password);
    } finally {
      setSubmitting(false);
    }
  };

  const disabled = loading || submitting;

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        display: "grid",
        gap: 16,
        padding: 24,
        border: "1px solid #d1d5db",
        borderRadius: 12,
        maxWidth: 360,
        margin: "0 auto",
      }}
    >
      <div style={{ display: "grid", gap: 4 }}>
        <h2 style={{ margin: 0, textAlign: "center" }}>{title ?? "Sign in"}</h2>
        <p style={{ margin: 0, textAlign: "center", color: "#6b7280" }}>
          Enter your credentials to access the workspace.
        </p>
      </div>
      <label style={{ display: "grid", gap: 4 }}>
        <span>Username</span>
        <input
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          autoComplete="username"
          disabled={disabled}
        />
      </label>
      <label style={{ display: "grid", gap: 4 }}>
        <span>Password</span>
        <input
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          autoComplete="current-password"
          disabled={disabled}
        />
      </label>
      {(localError || error) && (
        <p style={{ margin: 0, color: "#b91c1c" }}>{localError ?? error}</p>
      )}
      <button type="submit" disabled={disabled}>
        {disabled ? "Signing in..." : "Sign in"}
      </button>
    </form>
  );
}

