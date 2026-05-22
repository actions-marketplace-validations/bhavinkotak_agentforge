-- Add api_error to the failure_cluster enum to represent infrastructure failures
-- (rate limits, HTTP 5xx, timeouts) that are not agent quality issues.
ALTER TYPE failure_cluster ADD VALUE IF NOT EXISTS 'api_error';
