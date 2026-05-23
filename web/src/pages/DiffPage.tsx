import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchDiff } from '@/api/diff'
import { fetchAgents } from '@/api/agents'
import { DiffViewer } from '@/components/DiffViewer'
import { Button } from '@/components/ui/Button'
import { Input } from '@/components/ui/Input'
import { Card, CardContent } from '@/components/ui/Card'
import type { AgentResponse } from '@/types'
import { fmtDate, truncate } from '@/lib/utils'

/** Group a flat agents list by name → sorted versions (newest first). */
function groupByName(agents: AgentResponse[]): Map<string, AgentResponse[]> {
  const map = new Map<string, AgentResponse[]>()
  for (const a of agents) {
    const list = map.get(a.name) ?? []
    list.push(a)
    map.set(a.name, list)
  }
  // Sort each group newest-first
  map.forEach(list => list.sort((a, b) => b.created_at.localeCompare(a.created_at)))
  return map
}

interface AgentPickerProps {
  label: string
  agents: AgentResponse[]
  value: string
  onChange: (id: string) => void
}

function AgentPicker({ label, agents, value, onChange }: AgentPickerProps) {
  const groups = useMemo(() => groupByName(agents), [agents])
  const names = useMemo(() => Array.from(groups.keys()).sort(), [groups])
  const [name, setName] = useState<string>(() => {
    const found = agents.find(a => a.id === value)
    return found?.name ?? ''
  })

  // When name changes, auto-select the newest version
  function handleNameChange(n: string) {
    setName(n)
    const versions = groups.get(n) ?? []
    if (versions.length > 0) onChange(versions[0].id)
    else onChange('')
  }

  const versions = groups.get(name) ?? []

  return (
    <div className="flex-1 space-y-2">
      <p className="text-xs font-medium text-gray-600">{label}</p>
      <select
        className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        value={name}
        onChange={e => handleNameChange(e.target.value)}
      >
        <option value="">— pick an agent —</option>
        {names.map(n => (
          <option key={n} value={n}>{n}</option>
        ))}
      </select>
      {versions.length > 0 && (
        <select
          className="w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={value}
          onChange={e => onChange(e.target.value)}
        >
          {versions.map(v => (
            <option key={v.id} value={v.id}>
              {v.version} · {truncate(v.sha, 10)} · {fmtDate(v.created_at)}{v.is_champion ? ' ★' : ''}{v.parent_sha ? ' 🤖' : ''}
            </option>
          ))}
        </select>
      )}
      {/* UUID fallback input */}
      <Input
        placeholder="…or paste UUID directly"
        value={value}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  )
}

export function DiffPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [v1, setV1] = useState(searchParams.get('v1') ?? '')
  const [v2, setV2] = useState(searchParams.get('v2') ?? '')
  const [submitted, setSubmitted] = useState(!!searchParams.get('v1') && !!searchParams.get('v2'))

  const agentsQ = useQuery({
    queryKey: ['agents-for-diff'],
    queryFn: () => fetchAgents(200, 0),
    staleTime: 60_000,
  })

  const diffQ = useQuery({
    queryKey: ['diff', v1, v2],
    queryFn: () => fetchDiff(v1, v2),
    enabled: submitted && !!v1 && !!v2,
    retry: false,
  })

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSearchParams({ v1, v2 })
    setSubmitted(true)
  }

  useEffect(() => {
    const p1 = searchParams.get('v1')
    const p2 = searchParams.get('v2')
    if (p1) setV1(p1)
    if (p2) setV2(p2)
    if (p1 && p2) setSubmitted(true)
  }, [searchParams])

  const agents = agentsQ.data ?? []

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Version Diff</h1>
        <p className="mt-0.5 text-sm text-gray-500">
          Compare two agent versions — pick from the dropdowns or paste UUIDs directly.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        <Card>
          <CardContent className="flex flex-col gap-4 sm:flex-row sm:items-start">
            {agents.length > 0 ? (
              <>
                <AgentPicker label="Version A" agents={agents} value={v1} onChange={setV1} />
                <AgentPicker label="Version B" agents={agents} value={v2} onChange={setV2} />
              </>
            ) : (
              <>
                <div className="flex-1">
                  <Input
                    label="Agent v1 (UUID)"
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                    value={v1}
                    onChange={e => setV1(e.target.value)}
                    required
                  />
                </div>
                <div className="flex-1">
                  <Input
                    label="Agent v2 (UUID)"
                    placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
                    value={v2}
                    onChange={e => setV2(e.target.value)}
                    required
                  />
                </div>
              </>
            )}
            <div className="flex items-end pt-6">
              <Button type="submit" disabled={!v1 || !v2}>Compare →</Button>
            </div>
          </CardContent>
        </Card>
      </form>

      {diffQ.isLoading && (
        <p className="text-sm text-gray-500">Loading diff…</p>
      )}

      {diffQ.error && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {diffQ.error instanceof Error ? diffQ.error.message : 'Failed to load diff'}
        </div>
      )}

      {diffQ.data && <DiffViewer diff={diffQ.data} />}
    </div>
  )
}
