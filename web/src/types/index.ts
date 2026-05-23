// ── Agents ─────────────────────────────────────────────────────────────────
export interface AgentResponse {
  id: string
  name: string
  version: string
  sha: string
  parent_sha?: string
  changelog?: string
  format: string
  promoted: boolean
  is_champion: boolean
  created_at: string
}

// ── Eval Runs ───────────────────────────────────────────────────────────────
export type RunStatus = 'pending' | 'running' | 'complete' | 'error'

export interface RunResponse {
  id: string
  agent_id: string
  status: RunStatus
  aggregate_score?: number
  pass_rate?: number
  scenario_count: number
  completed_count: number
  error_count: number
  created_at: string
}

export interface DimensionScores {
  task_completion: number
  tool_selection: number
  argument_correctness: number
  schema_compliance: number
  instruction_adherence: number
  path_efficiency: number
}

export interface FailureCluster {
  cluster: string
  count: number
  percentage: number
}

/** Shape returned by GET /runs/:id/scorecard (full EvalRun JSON) */
export interface EvalRunDetail {
  id: string
  agent_id: string
  status: RunStatus
  scenario_count: number
  completed_count: number
  error_count: number
  aggregate_score?: number
  pass_rate?: number
  scores?: DimensionScores
  failure_clusters?: FailureCluster[]
  seed: number
  concurrency: number
  error_message?: string
  started_at?: string
  completed_at?: string
  created_at: string
  updated_at: string
}

// ── Promote ─────────────────────────────────────────────────────────────────
export interface GateResult {
  gate: string
  status: 'pass' | 'fail' | 'waived'
  message: string
}

export interface PromoteResponse {
  run_id: string
  agent_id: string
  approved: boolean
  changelog: string
  gates: GateResult[]
}

// ── Diff ────────────────────────────────────────────────────────────────────
export interface AgentSummary {
  id: string
  name: string
  version: string
  sha: string
  is_champion: boolean
}

export interface ToolChanges {
  added: string[]
  removed: string[]
  modified: string[]
}

export interface ConstraintChanges {
  added: string[]
  removed: string[]
}

export interface DiffResponse {
  v1: AgentSummary
  v2: AgentSummary
  system_prompt_diff?: string
  tool_changes: ToolChanges
  constraint_changes: ConstraintChanges
}

// ── Shadow Runs ─────────────────────────────────────────────────────────────
export type ShadowStatus = 'pending' | 'running' | 'complete' | 'error'

export interface DimensionComparison {
  dimension: string
  champion_score: number
  candidate_score: number
  outcome: 'Win' | 'Loss' | 'Tie'
  delta: number
}

export interface ShadowComparison {
  run_id: string
  champion_agent_id: string
  candidate_agent_id: string
  traffic_fraction: number
  total_requests: number
  champion_aggregate_score: number
  candidate_aggregate_score: number
  aggregate_delta: number
  per_dimension: DimensionComparison[]
  candidate_wins: number
  compared_at: string
}

export interface ShadowRunResponse {
  id: string
  champion_agent_id: string
  candidate_agent_id: string
  traffic_percent: number
  status: ShadowStatus
  created_at: string
  comparison?: ShadowComparison
}

// ── Fine-tune Export ────────────────────────────────────────────────────────
export type ExportStatus = 'pending' | 'running' | 'complete' | 'error'
export type ExportFormat = 'openai' | 'anthropic' | 'huggingface'

export interface FineTuneExportResponse {
  id: string
  run_id: string
  format: ExportFormat
  status: ExportStatus
  row_count?: number
  file_path?: string
  created_at: string
  completed_at?: string
}

// ── Benchmarks ──────────────────────────────────────────────────────────────
export type BenchmarkSuite = 'gaia' | 'agentbench' | 'webarena'
export type BenchmarkStatus = 'pending' | 'running' | 'complete' | 'error'

export interface BenchmarkRunResponse {
  id: string
  agent_id: string
  suite: BenchmarkSuite
  status: BenchmarkStatus
  total_tasks: number
  correct: number
  accuracy: number
  percentile_rank?: number
  started_at: string
  completed_at?: string
}

// ── Generic API error ────────────────────────────────────────────────────────
export interface ApiErrorBody {
  error: { code: string; message: string }
}
