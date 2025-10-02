"use client";
import Chat from "./components/Chat";
import { LoginForm } from "./components/LoginForm";
import { useAuth } from "./hooks/useAuth";

export default function Home() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:18000";
  const { ready, user, token, agents, login, loginPending, loginError, logout, refresh } =
    useAuth(apiBase);

  if (!ready) {
    return (
      <main style={{ maxWidth: 480, margin: "80px auto", textAlign: "center" }}>
        <p>Loading...</p>
      </main>
    );
  }

  if (!user || !token) {
    return (
      <main style={{ maxWidth: 540, margin: "80px auto" }}>
        <h1 style={{ textAlign: "center" }}>Multi-Agent RAG Chat</h1>
        <LoginForm onSubmit={login} loading={loginPending} error={loginError} />
        <section style={{ marginTop: 24, fontSize: 13, color: "#6b7280" }}>
          <p>
            Default demo users are provisioned automatically. Update the credentials
            by setting the <code>AUTH_USERS</code> environment variable.
          </p>
        </section>
      </main>
    );
  }

  return (
    <main style={{ maxWidth: 960, margin: "40px auto", display: "grid", gap: 24 }}>
      <header
        style={{
          display: "grid",
          gap: 8,
          padding: 16,
          border: "1px solid #d1d5db",
          borderRadius: 12,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1 style={{ margin: 0 }}>Multi-Agent RAG Chat</h1>
          <button type="button" onClick={logout}>
            Sign out
          </button>
        </div>
        <p style={{ margin: 0, color: "#374151" }}>
          Signed in as <strong>{user.username}</strong> ({user.role}).
        </p>
        <p style={{ margin: 0, color: "#4b5563", fontSize: 14 }}>
          Accessible agents: {agents.length ? agents.join(", ") : "(none)"}
        </p>
        <div>
          <button type="button" onClick={refresh} style={{ fontSize: 13 }}>
            Refresh access
          </button>
        </div>
      </header>
      <Chat apiBase={apiBase} token={token} />
      <a href="/upload" style={{ justifySelf: "start" }}>
        Go to Upload
      </a>
    </main>
  );
}
