-- Canonical install for the agent_sessions schema at the current library
-- version. Runs on a fresh database with no `agent_sessions` schema.
--
-- For existing databases, `apply_migrations()` detects the installed version
-- via `agent_sessions.get_schema_version()` and applies the deltas under
-- `schema/migrations/<from>-<to>.sql` in order. This file always represents
-- the current version; the migrations directory lets existing deployments
-- roll forward without reinstalling.

CREATE SCHEMA IF NOT EXISTS agent_sessions;

CREATE TABLE IF NOT EXISTS agent_sessions.sessions (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}',
    -- Lease columns for single-active-brain enforcement when concurrency='queue'.
    -- Both set together when a brain takes the lease; both cleared on release.
    -- `running_task_id` is the Absurd task_id of the active brain;
    -- `running_brain_name` names which brain is running (needed to target a
    -- specific brain in `concurrency='supersede'`). A row-level CAS on these
    -- replaces the advisory-lock pattern so contended brains never pin a pool
    -- connection.
    running_task_id TEXT,
    running_brain_name TEXT,
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

-- Version marker. Each release (or each breaking migration) updates this
-- function to return a new version string. `apply_migrations()` reads it to
-- decide which deltas to apply.
CREATE OR REPLACE FUNCTION agent_sessions.get_schema_version()
    RETURNS TEXT
    LANGUAGE SQL IMMUTABLE
    AS $$ SELECT '0.0.1' $$;
