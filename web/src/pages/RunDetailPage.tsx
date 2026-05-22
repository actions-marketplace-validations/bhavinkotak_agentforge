import { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Shield, FileDown, GitCompare } from 'lucide-react'
import { fetchRun, fetchScorecard, promoteRun } from '@/api/runs'
import { startExport } from '@/api/finetune'
import { ApiError } from '@/api/client'
import { ScorecardDisplay } from '@/components/ScorecardDisplay'
import { GateResultsList } from '@/components/GateResultsList'
import { PollingCard } from '@/components/PollingCard'
import { Button } from '@/components/ui/Button'
import { Select } from '@/components/ui/Input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { useRecentItems } from '@/hooks/useRecentItems'
import { fmtDate } from '@/lib/utils'
import type { PromoteResponse } from '@/types'

export function RunDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { push } = useRecentItems()
  const [promote, setPromote] = useState<PromoteResponse | null>(null)
  const [promoteErr, setPromoteErr] = useState<string | null>(null)
  const [showExport, setShowExport] = useState(false)
  const [exportFormat, setExportFormat] = useState('openai')

  const runQ = useQuery({
    queryKey: ['run', id],
    queryFn: () => fetchRun(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const s = query.state.data?.status
      if (s === 'complete' || s === 'error') return false
      return 3000
    },
  })

  const isComplete = runQ.data?.status === 'complete'
  const isError = runQ.data?.status === 'error'

  const scorecardQ = useQuery({
    queryKey: ['scorecard', id],
    queryFn: () => fetchScorecard(id!),
    enabled: isComplete,
  })

  const promoteMut = useMutation({ mutationFn: promoteRun })
  const exportMut = useMutation({ mutationFn: startExport })

  async function handlePromote() {
    setPromoteErr(null)
    try {
      const res = await promoteMut.mutateAsync(id!)
      setPromote(res)
    } catch (e) {
      setPromoteErr(e instanceof ApiError ? e.message : 'Promotion failed')
    }
  }

  async function handleExport(e: React.FormEvent) {
    e.preventDefault()
    try {
      const ex = await exportMut.mutateAsync({
        run_id: id!,
        format: exportFormat as 'openai' | 'anthropic' | 'huggingface',
      })
      push({ id: ex.id, type: 'export', label: `Export ${exportFormat} · run ${id!.slice(0, 8)}`, timestamp: Date.now() })
      navigate(`/exports/finetune/${ex.id}`)
    } catch (e) {
      // error surfaced via exportMut.error
      console.error(e)
    }
  }

  const run = runQ.data

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Eval Run</h1>
        <p className="mt-0.5 font-mono text-xs text-gray-400">{id}</p>
      </div>

      {/* Status + meta */}
      <PollingCard
        status={run?.status ?? 'pending'}
        label="Running evaluation… this may take a few minutes"
      >
        {run && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-sm">
            <div>
              <p className="text-xs text-gray-500">Agent</p>
              <Link
                to={`/agents/${run.agent_id}`}
                className="mt-0.5 block font-mono text-sm font-medium text-blue-600 hover:underline"
              >
                {run.agent_id.slice(0, 8)}…
              </Link>
            </div>
            <Meta label="Created" value={fmtDate(run.created_at)} />
            <Meta label="Status" value={run.status} />
          </div>
        )}
      </PollingCard>

      {/* Error */}
      {isError && scorecardQ.data?.error_message && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {scorecardQ.data.error_message}
        </div>
      )}

      {/* Scorecard */}
      {isComplete && scorecardQ.data && (
        <>
          <ScorecardDisplay run={scorecardQ.data} />

          {/* Action row */}
          {!promote && (
            <div className="flex flex-wrap gap-2">
              <Button onClick={handlePromote} loading={promoteMut.isPending}>
                <Shield className="h-4 w-4" />
                Promote Agent
              </Button>
              <Button
                variant="outline"
                onClick={() => navigate(`/diff?v1=${run?.agent_id}`)}
              >
                <GitCompare className="h-4 w-4" />
                Diff Agent
              </Button>
              <Button
                variant="outline"
                onClick={() => setShowExport(!showExport)}
              >
                <FileDown className="h-4 w-4" />
                Export Traces
              </Button>
            </div>
          )}

          {promoteErr && (
            <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              {promoteErr}
            </div>
          )}

          {/* Export form */}
          {showExport && (
            <Card>
              <CardHeader><CardTitle>Export Fine-tune Dataset</CardTitle></CardHeader>
              <CardContent>
                <form onSubmit={handleExport} className="flex items-end gap-3">
                  <div className="flex-1">
                    <Select
                      label="Format"
                      value={exportFormat}
                      onChange={e => setExportFormat(e.target.value)}
                      options={[
                        { value: 'openai', label: 'OpenAI (JSONL)' },
                        { value: 'anthropic', label: 'Anthropic (JSONL)' },
                        { value: 'huggingface', label: 'HuggingFace (JSONL)' },
                      ]}
                    />
                  </div>
                  <Button type="submit" loading={exportMut.isPending}>
                    Export →
                  </Button>
                </form>
                {exportMut.error && (
                  <p className="mt-2 text-xs text-red-600">
                    {exportMut.error instanceof ApiError
                      ? exportMut.error.message
                      : 'Export failed'}
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {/* Promotion result */}
          {promote && (
            <GateResultsList
              gates={promote.gates}
              approved={promote.approved}
              changelog={promote.changelog}
            />
          )}
        </>
      )}
    </div>
  )
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-gray-500">{label}</p>
      <p className="mt-0.5 text-sm font-medium text-gray-900">{value}</p>
    </div>
  )
}
