# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `v1.x`  | ✅ Yes    |
| `< v1`  | ❌ No     |

## Reporting a Vulnerability

**Please do not file a public GitHub issue for security vulnerabilities.**

Report vulnerabilities privately via one of the following channels:

- **GitHub Security Advisories** (preferred): [Report a vulnerability](https://github.com/bhavinkotak/agentforge/security/advisories/new)
- **GitHub DM**: Reach the maintainer at [@bhavinkotak](https://github.com/bhavinkotak) via GitHub

Include as much detail as possible:
- Description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept if available)
- Affected versions
- Any suggested mitigations

You can expect an acknowledgement within **48 hours** and a resolution or status update within **14 days**.

## Scope

AgentForge handles LLM API keys and executes evaluation runs against external AI providers. The following areas are considered in-scope:

| Area | Risk |
|------|------|
| API key handling / leakage | Critical |
| Prompt injection via agent files | High |
| SQL injection via the REST API or CLI | High |
| Path traversal in agent file loading | High |
| Denial-of-service via unbounded scenario counts | Medium |
| Insecure default configuration | Medium |

## Security Best Practices for Users

- **Store API keys as secrets** — never hardcode keys in agent files or workflow YAML. Use `${{ secrets.OPENAI_API_KEY }}` in GitHub Actions.
- **Pin the action to a specific version** — use `bhavinkotak/agentforge@v1.2.3` rather than `@v1` or `@main` to prevent supply-chain attacks.
- **Restrict `DATABASE_URL` permissions** — the AgentForge DB user only needs `SELECT`, `INSERT`, `UPDATE` on its own tables.
- **Review agent files before running** — the `system_prompt` field is passed verbatim to the LLM provider. Malicious agent files could include prompt injection attempts.
- **Set scenario limits** — use `AGENTFORGE_MAX_SCENARIOS` to cap runaway LLM costs.

## Disclosure Policy

We follow **coordinated disclosure**: once a fix is available, we will publish a GitHub Security Advisory and reference it in the release notes. Credit will be given to the reporter unless they prefer to remain anonymous.
