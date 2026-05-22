import { cn, fmtPct, fmtScore } from '@/lib/utils'
import type { DimensionScores, EvalRunDetail, FailureCluster } from '@/types'

const DIMS: { key: keyof DimensionScores; label: string }[] = [
  { key: 'task_completion', label: 'Task Completion' },
  { key: 'tool_selection', label: 'Tool Selection' },
  { key: 'argument_correctness', label: 'Argument Correctness' },
  { key: 'schema_compliance', label: 'Schema Compliance' },
  { key: 'instruction_adherence', label: 'Instruction Adherence' },
  { key: 'path_efficiency', label: 'Path Efficiency' },
]

function DimBar({ label, score }: { label: string; score: number }) {
  const pct = Math.round(score * 100)
  const color =
    pct >= 80 ? 'bg-green-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500'
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="text-gray-600">{label}</span>
        <span className="font-medium tabular-nums text-gray-900">{pct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-gray-100">
        <div
          className={cn('h-2 rounded-full transition-all', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

const CLUSTER_LABELS: Record<string, string> = {
  wrong_tool: 'Wrong Tool Called',
  hallucinated_argument: 'Hallucinated Argument',
  looping: 'Agent Looping',
  premature_stop: 'Premature Stop',
  schema_violation: 'Schema Violation',
  constraint_breach: 'Constraint Breach',
  api_error: 'API / Infrastructure Error',
  unknown: 'Unclassified',
}

function ClusterRow({ c }: { c: FailureCluster }) {
  const label = CLUSTER_LABELS[c.cluster] ?? c.cluster
  return (
    <div className="flex items-center justify-between py-1 text-sm">
      <span className="text-gray-700">{label}</span>
      <span className="font-medium text-gray-900">
        {c.count} ({(c.percentage * 100).toFixed(0)}%)
      </span>
    </div>
  )
}

interface Props {
  run: EvalRunDetail
}

export function ScorecardDisplay({ run }: Props) {
  const agg = run.aggregate_score
  const pr = run.pass_rate
  const allErrored = run.error_count > 0 && run.error_count >= run.scenario_count
  const someErrored = run.error_count > 0 && !allErrored

  return (
    <div className="space-y-6">
      {/* All traces errored — explain why scores are 0 */}
      {allErrored && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <p className="font-medium">
            All {run.error_count} trace{run.error_count !== 1 ? 's' : ''} errored — scores are unavailable.
          </p>
          <p className="mt-1 text-amber-700">
            The LLM API returned an error for every trace. Check that your API key is valid
            and that you have sufficient quota for the configured provider.
          </p>
          {run.error_message && (
            <p className="mt-1.5 font-mono text-xs text-amber-700 break-all">
              Sample error: {run.error_message}
            </p>
          )}
        </div>
      )}

      {/* Some traces errored */}
      {someErrored && (
        <div className="rounded-md border border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-800">
          {run.error_count} of {run.scenario_count} traces errored — scores reflect only the{' '}
          {run.completed_count} successful traces. Check LLM API availability and quota.
        </div>
      )}

      {/* Headline metrics */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Metric label="Aggregate Score" value={fmtScore(agg)} highlight={!!agg && agg >= 0.85} />
        <Metric label="Pass Rate" value={fmtPct(pr)} />
        <Metric
          label="Scenarios"
          value={`${run.completed_count} / ${run.scenario_count}`}
        />
        <Metric label="Errors" value={String(run.error_count)} warn={run.error_count > 0} />
      </div>

      {/* Dimension bars — suppress when all traces errored (all-zero bars are misleading) */}
      {run.scores && !allErrored && (
        <div className="rounded-lg border border-gray-200 bg-white p-5">
          <h3 className="mb-4 text-sm font-semibold text-gray-900">
            Dimension Scores
          </h3>
          <div className="space-y-3">
            {DIMS.map(({ key, label }) => (
              <DimBar key={key} label={label} score={run.scores![key]} />
            ))}
          </div>
        </div>
      )}

      {/* Failure clusters — exclude no_failure entries which are not failures */}
      {run.failure_clusters && run.failure_clusters.filter(c => c.cluster !== 'no_failure').length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-5">
          <h3 className="mb-3 text-sm font-semibold text-gray-900">
            Failure Clusters
          </h3>
          <div className="divide-y divide-gray-100">
            {run.failure_clusters
              .filter((c) => c.cluster !== 'no_failure')
              .map((c) => (
                <ClusterRow key={c.cluster} c={c} />
              ))}
          </div>
        </div>
      )}
    </div>
  )
}

function Metric({
  label,
  value,
  highlight,
  warn,
}: {
  label: string
  value: string
  highlight?: boolean
  warn?: boolean
}) {
  return (
    <div
      className={cn(
        'rounded-lg border p-4',
        highlight
          ? 'border-green-200 bg-green-50'
          : warn
          ? 'border-amber-200 bg-amber-50'
          : 'border-gray-200 bg-white',
      )}
    >
      <p className="text-xs text-gray-500">{label}</p>
      <p
        className={cn(
          'mt-1 text-2xl font-semibold tabular-nums',
          highlight ? 'text-green-700' : warn ? 'text-amber-700' : 'text-gray-900',
        )}
      >
        {value}
      </p>
    </div>
  )
}
