-- 0.0.1 -> 0.0.2: add `running_brain_name` to support `concurrency="supersede"`.
--
-- `supersede` needs to find the currently-leased brain for a given
-- (session_id, brain_name) tuple so it can cancel it before spawning a
-- replacement. The lease column alone (`running_task_id`) isn't enough -
-- it tells us *who* holds the lease, not *which* brain. Recording
-- `running_brain_name` alongside it is cheap and keeps the lease
-- information self-contained on the session row.

ALTER TABLE agent_sessions.sessions
    ADD COLUMN IF NOT EXISTS running_brain_name TEXT;

CREATE OR REPLACE FUNCTION agent_sessions.get_schema_version()
    RETURNS TEXT
    LANGUAGE SQL IMMUTABLE
    AS $$ SELECT '0.0.2' $$;
