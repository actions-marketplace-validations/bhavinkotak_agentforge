import { apiFetch } from './client'
import type { AgentResponse } from '@/types'

export interface CreateAgentRequest {
  content: string
}

export const fetchAgents = (limit = 200, offset = 0) =>
  apiFetch<AgentResponse[]>(`/agents?limit=${limit}&offset=${offset}`)

export const fetchAgentVersionsByName = (name: string) =>
  apiFetch<AgentResponse[]>(`/agents?name=${encodeURIComponent(name)}`)

export const fetchAgent = (id: string) =>
  apiFetch<AgentResponse>(`/agents/${id}`)

export const createAgent = (req: CreateAgentRequest) =>
  apiFetch<AgentResponse>('/agents', {
    method: 'POST',
    body: JSON.stringify(req),
  })
