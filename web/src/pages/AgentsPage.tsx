import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Plus, Search, Layers } from 'lucide-react'
import { fetchAgents } from '@/api/agents'
import { RunStatusBadge } from '@/components/RunStatusBadge'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card } from '@/components/ui/Card'
import { fmtDate, truncate } from '@/lib/utils'
import type { AgentResponse } from '@/types'

/** Pick the latest version per agent name (highest created_at). */
function groupByName(agents: AgentResponse[]): { latest: AgentResponse; count: number }[] {
  const groups = new Map<string, { latest: AgentResponse; count: number }>()
  for (const agent of agents) {
    const existing = groups.get(agent.name)
    if (!existing) {
      groups.set(agent.name, { latest: agent, count: 1 })
    } else {
      existing.count++
      if (new Date(agent.created_at) > new Date(existing.latest.created_at)) {
        existing.latest = agent
      }
    }
  }
  // Sort by latest created_at descending
  return Array.from(groups.values()).sort(
    (a, b) => new Date(b.latest.created_at).getTime() - new Date(a.latest.created_at).getTime(),
  )
}

export function AgentsPage() {
  const [search, setSearch] = useState('')

  const { data, isLoading } = useQuery({
    queryKey: ['agents'],
    queryFn: () => fetchAgents(200, 0),
  })

  const groups = groupByName(
    (data ?? []).filter(
      (a) =>
        !search ||
        a.name.toLowerCase().includes(search.toLowerCase()) ||
        a.version.includes(search),
    ),
  )

  return (
    <div className="mx-auto max-w-5xl space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Agents</h1>
          <p className="mt-0.5 text-sm text-gray-500">
            Latest version of each agent. Click to view version history.
          </p>
        </div>
        <Link to="/agents/new">
          <Button>
            <Plus className="h-4 w-4" />
            New Agent
          </Button>
        </Link>
      </div>

      <div className="max-w-xs">
        <Input
          placeholder="Filter by name or version…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-8"
        />
        <Search className="-mt-7 ml-2.5 h-4 w-4 text-gray-400 relative pointer-events-none" />
      </div>

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50">
                <Th>Name</Th>
                <Th>Latest Version</Th>
                <Th>Format</Th>
                <Th>SHA</Th>
                <Th>Status</Th>
                <Th>Updated</Th>
                <Th>Versions</Th>
                <Th />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {isLoading && (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-gray-500">
                    Loading…
                  </td>
                </tr>
              )}
              {!isLoading && groups.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-gray-500">
                    No agents found.
                  </td>
                </tr>
              )}
              {groups.map(({ latest: agent, count }) => (
                <tr key={agent.id} className="hover:bg-gray-50 transition-colors">
                  <td className="px-4 py-3 font-medium text-gray-900">{agent.name}</td>
                  <td className="px-4 py-3 text-gray-600">v{agent.version}</td>
                  <td className="px-4 py-3 text-gray-500">{agent.format}</td>
                  <td className="px-4 py-3 font-mono text-gray-400">
                    {truncate(agent.sha, 10)}
                  </td>
                  <td className="px-4 py-3">
                    {agent.is_champion && <RunStatusBadge status="champion" />}
                  </td>
                  <td className="px-4 py-3 text-gray-400">{fmtDate(agent.created_at)}</td>
                  <td className="px-4 py-3">
                    <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
                      <Layers className="h-3 w-3" />
                      {count}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <Link
                      to={`/agents/${agent.id}`}
                      className="text-indigo-600 hover:underline"
                    >
                      View →
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function Th({ children }: { children?: React.ReactNode }) {
  return (
    <th className="px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">
      {children}
    </th>
  )
}
