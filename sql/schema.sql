CREATE TABLE IF NOT EXISTS requests (
    id BIGSERIAL PRIMARY KEY,
    app_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT NOT NULL,
    operation TEXT NOT NULL,
    cost NUMERIC(20, 6) NOT NULL,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    total_tokens BIGINT,
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_requests_app_created ON requests (app_id, created_at);
