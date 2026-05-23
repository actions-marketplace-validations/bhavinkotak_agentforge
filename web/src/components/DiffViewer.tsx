import { cn } from '@/lib/utils'
import type { DiffResponse } from '@/types'

function DiffLine({ line }: { line: string }) {
  if (line.startsWith('+'))
    return (
      <div className="bg-green-50 px-3 text-green-800">
        <code className="text-xs">{line}</code>
      </div>
    )
  if (line.startsWith('-'))
    return (
      <div className="bg-red-50 px-3 text-red-800">
        <code className="text-xs">{line}</code>
      </div>
    )
  return (
    <div className="px-3 text-gray-600">
      <code className="text-xs">{line}</code>
    </div>
  )
}

function ChangeList({
  items,
  kind,
}: {
  items: string[]
  kind: 'added' | 'removed' | 'modified'
}) {
  if (items.length === 0) return null
  const color =
    kind === 'added'
      ? 'text-green-700 bg-green-50'
      : kind === 'removed'
        ? 'text-red-700 bg-red-50'
        : 'text-yellow-700 bg-yellow-50'
  const prefix = kind === 'added' ? '+' : kind === 'removed' ? '−' : '~'
  return (
    <div className="space-y-1">
      {items.map((item) => (
        <div key={item} className={cn('rounded px-2 py-0.5 text-xs font-mono', color)}>
          {prefix} {item}
        </div>
      ))}
    </div>
  )
}

interface Props {
  diff: DiffResponse
}

export function DiffViewer({ diff }: Props) {
  const lines = diff.system_prompt_diff?.split('\n') ?? []
  const hasChanges =
    lines.length > 0 ||
    diff.tool_changes.added.length > 0 ||
    diff.tool_changes.removed.length > 0 ||
    diff.tool_changes.modified.length > 0 ||
    diff.constraint_changes.added.length > 0 ||
    diff.constraint_changes.removed.length > 0

  return (
    <div className="space-y-6">
      {/* Agent summaries */}
      <div className="grid grid-cols-2 gap-4">
        <AgentCard label="v1" agent={diff.v1} />
        <AgentCard label="v2" agent={diff.v2} />
      </div>

      {!hasChanges && (
        <p className="rounded-md border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-500">
          No content differences found between these two versions.
        </p>
      )}

      {/* System prompt diff */}
      {lines.length > 0 && (
        <Section title="System Prompt">
          <div className="overflow-x-auto rounded-md border border-gray-200 font-mono">
            <div className="divide-y divide-gray-100 py-2">
              {lines.map((line, i) => (
                <DiffLine key={i} line={line} />
              ))}
            </div>
          </div>
        </Section>
      )}

      {/* Tool changes */}
      {(diff.tool_changes.added.length > 0 ||
        diff.tool_changes.removed.length > 0 ||
        diff.tool_changes.modified.length > 0) && (
        <Section title="Tool Changes">
          <div className="space-y-2">
            <ChangeList items={diff.tool_changes.added} kind="added" />
            <ChangeList items={diff.tool_changes.removed} kind="removed" />
            <ChangeList items={diff.tool_changes.modified} kind="modified" />
          </div>
        </Section>
      )}

      {/* Constraint changes */}
      {(diff.constraint_changes.added.length > 0 ||
        diff.constraint_changes.removed.length > 0) && (
        <Section title="Constraint Changes">
          <div className="space-y-2">
            <ChangeList items={diff.constraint_changes.added} kind="added" />
            <ChangeList items={diff.constraint_changes.removed} kind="removed" />
          </div>
        </Section>
      )}
    </div>
  )
}

function AgentCard({
  label,
  agent,
}: {
  label: string
  agent: DiffResponse['v1']
}) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-gray-400">
          {label}
        </span>
        {agent.is_champion && (
          <span className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-700 ring-1 ring-inset ring-indigo-200">
            champion
          </span>
        )}
      </div>
      <p className="mt-2 text-sm font-medium text-gray-900">
        {agent.name} v{agent.version}
      </p>
      <p className="mt-0.5 font-mono text-xs text-gray-400">
        {agent.sha.slice(0, 12)}
      </p>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-3 text-sm font-semibold text-gray-900">{title}</h3>
      {children}
    </div>
  )
}
