"use client";
import { LoginForm } from "../components/LoginForm";
import Uploader from "../components/Uploader";
import { useAuth } from "../hooks/useAuth";

export default function UploadPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:18000";
  const {
    ready,
    user,
    token,
    agents,
    login,
    loginPending,
    loginError,
    logout,
    refresh,
  } = useAuth(apiBase);

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
        <h1 style={{ textAlign: "center" }}>Upload Documents</h1>
        <LoginForm
          onSubmit={login}
          loading={loginPending}
          error={loginError}
          title="Sign in to upload"
        />
        <section style={{ marginTop: 24, fontSize: 13, color: "#6b7280" }}>
          <p>
            Use the same credentials that grant chat access. Owners can upload to
            any agent; admins are limited to their assigned ones.
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
          <h1 style={{ margin: 0 }}>Upload Documents</h1>
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
      <Uploader apiBase={apiBase} token={token} />
      <a href="/" style={{ justifySelf: "start" }}>
        &lt;- Back
      </a>
    </main>
  );
}
