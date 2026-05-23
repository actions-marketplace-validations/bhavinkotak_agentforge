import { apiFetch } from './client'
import type { EvalRunDetail, PromoteResponse, RunResponse } from '@/types'

export interface StartRunRequest {
  agent_id: string
  scenario_count?: number
  concurrency?: number
  seed?: number
  /** Enable iterative self-improvement loop (default: true on the server). */
  auto_optimize?: boolean
  /** Target score 0.0–1.0 (default: 0.95). */
  threshold?: number
  /** Maximum optimization rounds (default: 5). */
  max_opt_iterations?: number
}

export const startRun = (req: StartRunRequest) =>
  apiFetch<RunResponse>('/runs', { method: 'POST', body: JSON.stringify(req) })

export const fetchRun = (id: string) => apiFetch<RunResponse>(`/runs/${id}`)

export const fetchScorecard = (id: string) =>
  apiFetch<EvalRunDetail>(`/runs/${id}/scorecard`)

export const promoteRun = (runId: string) =>
  apiFetch<PromoteResponse>(`/promote/${runId}`, { method: 'POST' })

export const fetchRunsForAgent = (agentId: string, limit = 20) =>
  apiFetch<RunResponse[]>(`/agents/${agentId}/runs?limit=${limit}`)
