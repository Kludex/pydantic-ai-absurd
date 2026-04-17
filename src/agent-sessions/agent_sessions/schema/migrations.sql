CREATE SCHEMA IF NOT EXISTS agent_sessions;

CREATE TABLE IF NOT EXISTS agent_sessions.sessions (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}',
    -- Lease column for single-active-brain enforcement when concurrency='queue'.
    -- Holds the Absurd task_id of the running brain, or NULL when no brain is
    -- active. A row-level CAS on this column replaces the advisory-lock pattern
    -- so contended brains never pin a pool connection.
    running_task_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_sessions.session_events (
    session_id UUID NOT NULL REFERENCES agent_sessions.sessions(id) ON DELETE CASCADE,
    sequence BIGINT NOT NULL,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'public',
    payload_version INT NOT NULL DEFAULT 1,
    payload JSONB NOT NULL,
    causation_id BIGINT,
    supersedes BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, sequence)
);

CREATE INDEX IF NOT EXISTS session_events_kind_idx
    ON agent_sessions.session_events (session_id, kind);
CREATE INDEX IF NOT EXISTS session_events_actor_idx
    ON agent_sessions.session_events (session_id, actor);

CREATE TABLE IF NOT EXISTS agent_sessions.session_snapshots (
    session_id UUID NOT NULL REFERENCES agent_sessions.sessions(id) ON DELETE CASCADE,
    up_to_sequence BIGINT NOT NULL,
    summary_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, up_to_sequence)
);
