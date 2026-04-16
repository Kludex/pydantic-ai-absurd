CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS session_events (
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
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

CREATE INDEX IF NOT EXISTS session_events_kind_idx ON session_events (session_id, kind);
CREATE INDEX IF NOT EXISTS session_events_actor_idx ON session_events (session_id, actor);

CREATE TABLE IF NOT EXISTS session_snapshots (
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    up_to_sequence BIGINT NOT NULL,
    summary_payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, up_to_sequence)
);

CREATE TABLE IF NOT EXISTS wake_dedup (
    dedup_key TEXT PRIMARY KEY,
    session_id UUID NOT NULL,
    brain_name TEXT NOT NULL,
    task_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
