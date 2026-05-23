-- Track the iterative self-improvement loop state on each eval run.
--
-- opt_status:       'running' | 'converged' | 'no_improvement' | 'max_iterations' | 'failed'
-- opt_rounds:       number of optimization rounds completed (0 = not started)
-- opt_best_score:   highest aggregate score achieved across all rounds
-- opt_best_agent_id: UUID of the agent version saved in the best round (NULL if none saved)

ALTER TABLE eval_runs
  ADD COLUMN IF NOT EXISTS opt_status       TEXT,
  ADD COLUMN IF NOT EXISTS opt_rounds       INT NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS opt_best_score   DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS opt_best_agent_id UUID;
