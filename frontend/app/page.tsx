"use client";
import { useEffect, useMemo, useState } from "react";

import Chat from "./components/Chat";
import { PrincipalPicker } from "./components/PrincipalPicker";
import RoleSwitch from "./components/RoleSwitch";
import { useDemoPrincipals } from "./hooks/useDemoPrincipals";

export default function Home() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
  const { principals, loading, error } = useDemoPrincipals(apiBase);

  const [user, setUser] = useState<string>("");
  const [role, setRole] = useState<string>("owner");

  const selectedPrincipal = useMemo(
    () => principals.find((principal) => principal.username === user),
    [principals, user],
  );

  useEffect(() => {
    if (principals.length === 0) {
      return;
    }
    if (!selectedPrincipal) {
      const first = principals[0];
      setUser(first.username);
      setRole(first.role);
    }
  }, [principals, selectedPrincipal]);

  return (
    <main
      style={{ maxWidth: 960, margin: "40px auto", display: "grid", gap: 24 }}
    >
      <h1>Multi-Agent RAG Chat</h1>
      <section
        style={{
          display: "grid",
          gap: 12,
          padding: 16,
          border: "1px solid #d1d5db",
          borderRadius: 12,
        }}
      >
        <h2 style={{ margin: 0 }}>Demo Controls</h2>
        <PrincipalPicker
          principals={principals}
          value={user}
          onChange={(principal) => {
            setUser(principal.username);
            setRole(principal.role);
          }}
          loading={loading}
        />
        <RoleSwitch value={role} onChange={setRole} disabled={!user} />
        {selectedPrincipal && (
          <p style={{ fontSize: 13, color: "#374151", margin: 0 }}>
            Accessible agents:{" "}
            {selectedPrincipal.agentSlugs.join(", ") || "(none)"}
          </p>
        )}
        {error && <p style={{ color: "#b91c1c", margin: 0 }}>{error}</p>}
      </section>
      <Chat apiBase={apiBase} role={role} user={user} />
      <a href="/upload" style={{ justifySelf: "start" }}>
        Go to Upload
      </a>
    </main>
  );
}
