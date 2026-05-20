use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process;
use uuid::Uuid;

use agentforge_db::{
    agent_repo::AgentRepo, create_pool, eval_repo::EvalRepo, finetune_repo::FineTuneRepo,
    scenario_repo::ScenarioRepo, shadow_repo::ShadowRepo, trace_repo::TraceRepo,
};
use agentforge_gatekeeper::{GateStatus, Gatekeeper, GatekeeperConfig};
use agentforge_parser::{parse_agent_file, to_agent_version, validate_agent_file};
use agentforge_runner::{AgentRunner, AnthropicClient, NvidiaClient, OpenAiClient, RunnerConfig};
use agentforge_scenarios::{generate_scenarios, ScenarioGeneratorConfig};
use agentforge_scorer::{score_run, ScorerConfig};

#[derive(Parser)]
#[command(
    name = "agentforge",
    about = "AI agent optimization and evaluation platform",
    version,
    author
)]
struct Cli {
    /// Increase verbosity (-v debug, -vv trace)
    #[arg(short = 'v', long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run evaluation on an agent file
    Run {
        /// Path to the agent YAML/JSON file
        #[arg(short, long)]
        agent: PathBuf,

        /// Number of scenarios to generate
        #[arg(short, long, default_value = "100")]
        scenarios: u32,

        /// Concurrency for parallel scenario execution
        #[arg(short, long, default_value = "10")]
        concurrency: u32,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// LLM provider for the agent under test: openai | anthropic | nvidia
        #[arg(long, default_value = "openai")]
        provider: String,

        /// LLM provider for the judge (must differ from agent provider): openai | anthropic | nvidia
        #[arg(long, default_value = "anthropic")]
        judge_provider: String,

        /// Minimum aggregate score to gate promotion (0.0–1.0)
        #[arg(long, default_value = "0.85")]
        threshold: f64,

        /// Write full scorecard JSON to FILE (used by the GitHub Action)
        #[arg(long)]
        output_json: Option<PathBuf>,

        /// Validate agent file and preview scenario count without executing any LLM calls
        #[arg(long, default_value = "false")]
        dry_run: bool,

        /// Override agent file format detection: native_yaml | openai_json | anthropic_json |
        /// langchain_yaml | crewai_yaml | copilot_agent_md
        #[arg(long)]
        agent_format: Option<String>,

        /// Override weight for task-completion dimension (default: 0.35)
        #[arg(long)]
        weight_task: Option<f64>,

        /// Override weight for tool-selection dimension (default: 0.20)
        #[arg(long)]
        weight_tool: Option<f64>,

        /// Override weight for argument-correctness dimension (default: 0.20)
        #[arg(long)]
        weight_args: Option<f64>,

        /// Override weight for schema-compliance dimension (default: 0.15)
        #[arg(long)]
        weight_schema: Option<f64>,

        /// Override weight for instruction-adherence dimension (default: 0.07)
        #[arg(long)]
        weight_instr: Option<f64>,

        /// Override weight for path-efficiency dimension (default: 0.03)
        #[arg(long)]
        weight_path: Option<f64>,

        /// Enable red-team adversarial probes in addition to standard scenarios
        #[arg(long, default_value = "false")]
        red_team: bool,

        /// After the eval run, analyze cost and suggest cheaper model alternatives
        #[arg(long, default_value = "false")]
        cost_optimize: bool,
    },

    /// Compare two agent versions
    Diff {
        /// First agent version ID (UUID)
        v1: Uuid,
        /// Second agent version ID (UUID)
        v2: Uuid,
    },

    /// Promote a run's agent version to champion
    Promote {
        /// Run ID to promote
        run_id: Uuid,
    },

    /// Show scores for a run
    Scores {
        /// Run ID to display scores for
        #[arg(long)]
        run: Uuid,
    },

    /// Start a shadow (online eval) run comparing champion vs. candidate
    Shadow {
        /// Champion agent version ID (UUID)
        #[arg(long)]
        champion: Uuid,
        /// Candidate agent version ID (UUID)
        #[arg(long)]
        candidate: Uuid,
        /// Percentage of traffic to route to candidate (1–100)
        #[arg(long, default_value = "10")]
        traffic_percent: u8,
    },

    /// Export labeled traces as a fine-tuning dataset
    Export {
        /// Eval run ID to export traces from
        #[arg(long)]
        run: Uuid,
        /// Output format: openai | anthropic | huggingface
        #[arg(long, default_value = "openai")]
        format: String,
        /// Output file path (default: stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Run an agent against a public benchmark suite
    Benchmark {
        /// Path to the agent YAML/JSON file
        #[arg(short, long)]
        agent: PathBuf,
        /// Benchmark suite: gaia | agentbench | webarena
        #[arg(long)]
        suite: String,
        /// Path to the benchmark JSONL task file
        #[arg(long)]
        tasks: PathBuf,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Load .env
    let _ = dotenvy::dotenv();

    // Init tracing based on verbosity
    let level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .init();

    let exit_code = match run_command(cli.command).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e:#}");
            2
        }
    };

    process::exit(exit_code);
}

/// Returns 0 for pass, 1 for gate failure, 2 for error.
async fn run_command(command: Commands) -> Result<i32> {
    match command {
        Commands::Run {
            agent,
            scenarios,
            concurrency,
            seed,
            provider,
            judge_provider,
            threshold,
            output_json,
            dry_run,
            agent_format,
            weight_task,
            weight_tool,
            weight_args,
            weight_schema,
            weight_instr,
            weight_path,
            red_team,
            cost_optimize,
        } => {
            cmd_run(
                agent,
                scenarios,
                concurrency,
                seed,
                provider,
                judge_provider,
                threshold,
                output_json,
                dry_run,
                agent_format,
                weight_task,
                weight_tool,
                weight_args,
                weight_schema,
                weight_instr,
                weight_path,
                red_team,
                cost_optimize,
            )
            .await
        }
        Commands::Diff { v1, v2 } => cmd_diff(v1, v2).await,
        Commands::Promote { run_id } => cmd_promote(run_id).await,
        Commands::Scores { run } => cmd_scores(run).await,
        Commands::Shadow {
            champion,
            candidate,
            traffic_percent,
        } => cmd_shadow(champion, candidate, traffic_percent).await,
        Commands::Export {
            run,
            format,
            output,
        } => cmd_export(run, format, output).await,
        Commands::Benchmark {
            agent,
            suite,
            tasks,
        } => cmd_benchmark(agent, suite, tasks).await,
    }
}

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    agent_path: PathBuf,
    scenario_count: u32,
    concurrency: u32,
    seed: u64,
    provider: String,
    judge_provider: String,
    threshold: f64,
    output_json: Option<PathBuf>,
    dry_run: bool,
    agent_format: Option<String>,
    weight_task: Option<f64>,
    weight_tool: Option<f64>,
    weight_args: Option<f64>,
    weight_schema: Option<f64>,
    weight_instr: Option<f64>,
    weight_path: Option<f64>,
    red_team: bool,
    cost_optimize: bool,
) -> Result<i32> {
    let content = std::fs::read_to_string(&agent_path)
        .with_context(|| format!("Failed to read agent file: {}", agent_path.display()))?;

    // Parse (optionally with an explicit format override)
    let parsed = if let Some(fmt_str) = &agent_format {
        let fmt: agentforge_parser::AgentFileFormat = fmt_str
            .parse()
            .with_context(|| format!("Unknown agent format: {fmt_str}"))?;
        agentforge_parser::parse_agent_file_with_format(&content, fmt)
            .with_context(|| "Failed to parse agent file")?
    } else {
        parse_agent_file(&content).with_context(|| "Failed to parse agent file")?
    };

    // Validate
    let validation = validate_agent_file(&parsed.agent);
    for err in &validation.errors {
        eprintln!("[ERROR] {}", err.message);
    }
    for warn in &validation.warnings {
        eprintln!("[WARN]  {}", warn.message);
    }
    if !validation.errors.is_empty() {
        eprintln!(
            "Validation failed with {} error(s)",
            validation.errors.len()
        );
        return Ok(2);
    }

    let agent_file = parsed.agent.clone();
    let format = parsed.format.clone();
    let sha = parsed.sha.clone();
    println!(
        "Agent: {} v{} (format: {}, sha: {})",
        agent_file.name,
        agent_file.version,
        format,
        &sha[..12]
    );

    // --dry-run: validate only, no LLM calls
    if dry_run {
        println!(
            "\n--dry-run: validation passed. {} scenarios would be generated.",
            scenario_count
        );
        println!("  Provider: {provider} | Judge: {judge_provider} | Threshold: {threshold:.2}");
        return Ok(0);
    }

    // Build eval weights from CLI flags (fall back to env-var defaults)
    let weights = agentforge_core::EvalWeights {
        task_completion: weight_task.unwrap_or_else(|| {
            std::env::var("AGENTFORGE_WEIGHT_TASK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.35)
        }),
        tool_selection: weight_tool.unwrap_or_else(|| {
            std::env::var("AGENTFORGE_WEIGHT_TOOL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.20)
        }),
        argument_correctness: weight_args.unwrap_or_else(|| {
            std::env::var("AGENTFORGE_WEIGHT_ARGS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.20)
        }),
        schema_compliance: weight_schema.unwrap_or_else(|| {
            std::env::var("AGENTFORGE_WEIGHT_SCHEMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.15)
        }),
        instruction_adherence: weight_instr.unwrap_or_else(|| {
            std::env::var("AGENTFORGE_WEIGHT_INSTR")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.07)
        }),
        path_efficiency: weight_path.unwrap_or_else(|| {
            std::env::var("AGENTFORGE_WEIGHT_PATH")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.03)
        }),
    };
    if !weights.validate() {
        eprintln!(
            "[WARN] Scoring weights do not sum to 1.0 ({:.3}). Results may be unexpected.",
            weights.task_completion
                + weights.tool_selection
                + weights.argument_correctness
                + weights.schema_compliance
                + weights.instruction_adherence
                + weights.path_efficiency
        );
    }

    // DB (optional — skip if no DATABASE_URL)
    let db_opt = if let Ok(url) = std::env::var("DATABASE_URL") {
        let pool = create_pool(&url).await.ok();
        if let Some(ref pool) = pool {
            let _ = agentforge_db::run_migrations(pool).await;
        }
        pool
    } else {
        None
    };

    // Store agent version in DB if available
    let agent_id = if let Some(ref db) = db_opt {
        let repo = AgentRepo::new(db.clone());
        let agent_version = to_agent_version(parsed.clone());
        match repo.find_by_sha(&sha).await? {
            Some(existing) => existing.id,
            None => repo.insert(&agent_version).await?.id,
        }
    } else {
        Uuid::new_v4()
    };

    // Generate scenarios
    println!("Generating {} scenarios...", scenario_count);
    let scorer_config = build_scorer_config(&judge_provider, weights);
    let mut scenarios = generate_scenarios(
        &agent_file,
        &ScenarioGeneratorConfig {
            total_count: scenario_count,
            agent_id,
            llm_base_url: Some(scorer_config.judge_base_url.clone()),
            llm_api_key: if scorer_config.judge_api_key.is_empty() {
                None
            } else {
                Some(scorer_config.judge_api_key.clone())
            },
            llm_model: Some(scorer_config.judge_model.clone()),
            ..Default::default()
        },
    )
    .await
    .with_context(|| "Scenario generation failed")?;

    // v2: Append red-team scenarios if --red-team flag is set
    if red_team {
        use agentforge_redteam::{RedTeamConfig, RedTeamGenerator};
        let rt_gen = RedTeamGenerator::new(RedTeamConfig {
            count: (scenario_count / 5).max(10) as usize,
            seed,
        });
        let rt_scenarios = rt_gen.generate(&agent_file);
        println!(
            "Red-team: appending {} adversarial probes.",
            rt_scenarios.len()
        );
        scenarios.extend(rt_scenarios);
    }

    println!("Generated {} scenarios total.", scenarios.len());

    // Build LLM client
    let llm_client = build_llm_client(&provider)?;

    // Run agent
    println!(
        "Running agent across {} scenarios (concurrency: {})...",
        scenarios.len(),
        concurrency
    );
    let runner = AgentRunner::new(
        llm_client,
        RunnerConfig {
            concurrency: concurrency as usize,
            ..Default::default()
        },
    );
    let run_result = runner
        .run(
            &agent_file,
            scenarios.clone(),
            Some(std::sync::Arc::new(move |done: u32, total: u32| {
                if done.is_multiple_of(10) || done == total {
                    print!("\r  Progress: {}/{} scenarios", done, total);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
            })),
        )
        .await;
    let mut traces = run_result.traces;
    println!();

    // Score
    let run_id = Uuid::new_v4();
    let scorecard = score_run(&mut traces, &scenarios, &agent_file, run_id, &scorer_config)
        .await
        .with_context(|| "Scoring failed")?;

    // Print results
    print_scorecard(&scorecard);

    // Write scorecard JSON if --output-json was specified (used by the GitHub Action)
    if let Some(ref json_path) = output_json {
        let pass_threshold_used = agent_file
            .eval_hints
            .as_ref()
            .and_then(|h| h.pass_threshold)
            .unwrap_or(threshold);
        let promoted = scorecard.aggregate_score >= pass_threshold_used;
        let json = serde_json::json!({
            "run_id": run_id,
            "agent_name": scorecard.agent_name,
            "agent_version": scorecard.agent_version,
            "pass_rate": scorecard.pass_rate,
            "aggregate_score": scorecard.aggregate_score,
            "promoted": promoted,
            "dimension_scores": {
                "task_completion": scorecard.dimension_scores.task_completion,
                "tool_selection": scorecard.dimension_scores.tool_selection,
                "argument_correctness": scorecard.dimension_scores.argument_correctness,
                "schema_compliance": scorecard.dimension_scores.schema_compliance,
                "instruction_adherence": scorecard.dimension_scores.instruction_adherence,
                "path_efficiency": scorecard.dimension_scores.path_efficiency,
            },
        });
        std::fs::write(json_path, serde_json::to_string_pretty(&json)?)
            .with_context(|| format!("Failed to write scorecard to {}", json_path.display()))?;
        println!("Scorecard written to {}", json_path.display());
    }

    // v2: Cost optimizer analysis
    if cost_optimize {
        use agentforge_optimizer::CostOptimizer;
        let recommendations = CostOptimizer::analyze(&agent_file, &traces, 0.02);
        if recommendations.is_empty() {
            println!("\nCost Optimizer: No cheaper model alternatives found.");
        } else {
            println!("\nCost Optimizer Recommendations:");
            for rec in &recommendations {
                println!(
                    "  {} → {} | Est. savings: ${:.4} | Current score: {:.3}",
                    rec.current_model,
                    rec.recommended_model,
                    rec.estimated_savings_usd,
                    rec.current_aggregate_score
                );
            }
        }
    }

    // Persist to DB if available
    if let Some(ref db) = db_opt {
        let eval_repo = EvalRepo::new(db.clone());
        let scenario_repo = ScenarioRepo::new(db.clone());
        let trace_repo = TraceRepo::new(db.clone());

        let new_run = agentforge_core::EvalRun {
            id: Uuid::new_v4(),
            agent_id,
            scenario_set_id: None,
            status: agentforge_core::EvalRunStatus::Pending,
            scenario_count: scenarios.len() as u32,
            completed_count: 0,
            error_count: 0,
            aggregate_score: None,
            pass_rate: None,
            scores: None,
            failure_clusters: None,
            seed: seed as u32,
            concurrency,
            error_message: None,
            started_at: None,
            completed_at: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        let eval_run = eval_repo.insert(&new_run).await?;
        scenario_repo.insert_batch(&scenarios).await?;
        for trace in &traces {
            let _ = trace_repo.insert(trace).await;
        }
        eval_repo
            .save_scores(
                eval_run.id,
                &scorecard.dimension_scores,
                scorecard.aggregate_score,
                scorecard.pass_rate,
                &scorecard.failure_clusters,
            )
            .await?;
        eval_repo
            .update_status(eval_run.id, &agentforge_core::EvalRunStatus::Complete)
            .await?;
        println!("\nRun ID: {}", eval_run.id);
    }

    // Exit code: 0 = pass threshold met, 1 = failed
    let pass_threshold = agent_file
        .eval_hints
        .as_ref()
        .and_then(|h| h.pass_threshold)
        .unwrap_or(threshold);

    if scorecard.aggregate_score >= pass_threshold {
        Ok(0)
    } else {
        eprintln!(
            "\nFailed: aggregate score {:.3} < threshold {:.3}",
            scorecard.aggregate_score, pass_threshold
        );
        Ok(1)
    }
}

async fn cmd_diff(v1: Uuid, v2: Uuid) -> Result<i32> {
    let db = require_db().await?;
    let repo = AgentRepo::new(db);

    let ver1 = repo
        .find_by_id(v1)
        .await
        .map_err(|_| anyhow::anyhow!("Version {v1} not found"))?;
    let ver2 = repo
        .find_by_id(v2)
        .await
        .map_err(|_| anyhow::anyhow!("Version {v2} not found"))?;

    println!(
        "Diff: {} v{} → {} v{}",
        ver1.name, ver1.version, ver2.name, ver2.version
    );
    println!("SHA: {} → {}", &ver1.sha[..12], &ver2.sha[..12]);

    let prompt1 = ver1.file_content.system_prompt.as_str();
    let prompt2 = ver2.file_content.system_prompt.as_str();

    if prompt1 != prompt2 {
        println!(
            "\nSystem prompt changed ({} → {} chars)",
            prompt1.len(),
            prompt2.len()
        );
    } else {
        println!("\nSystem prompt: unchanged");
    }

    Ok(0)
}

async fn cmd_promote(run_id: Uuid) -> Result<i32> {
    let db = require_db().await?;
    let eval_repo = EvalRepo::new(db.clone());
    let agent_repo = AgentRepo::new(db.clone());
    let trace_repo = TraceRepo::new(db.clone());

    let run = eval_repo
        .find_by_id(run_id)
        .await
        .map_err(|_| anyhow::anyhow!("Run {run_id} not found"))?;

    if run.status != agentforge_core::EvalRunStatus::Complete {
        anyhow::bail!("Run {run_id} is not complete (status: {:?})", run.status);
    }

    let challenger_scorecard = run
        .to_scorecard()
        .ok_or_else(|| anyhow::anyhow!("Run {run_id} has no scores"))?;

    let challenger_traces = trace_repo.list_by_run(run_id).await?;

    // Find current champion by agent name — fetch agent first to get name
    let challenger_agent = agent_repo
        .find_by_id(run.agent_id)
        .await
        .map_err(|_| anyhow::anyhow!("Agent not found"))?;
    let champion_versions = agent_repo.list_by_name(&challenger_agent.name).await?;
    let champion = champion_versions.iter().find(|v| v.is_champion);
    let champion_scorecard = if let Some(champ) = champion {
        let runs = eval_repo.list_by_agent(champ.id, 1).await?;
        runs.into_iter().next().and_then(|r| r.to_scorecard())
    } else {
        None
    };

    let champion_passing = if let Some(champ) = champion {
        let runs = eval_repo.list_by_agent(champ.id, 1).await?;
        if let Some(champ_run) = runs.into_iter().next() {
            trace_repo.list_passing_scenario_ids(champ_run.id).await?
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    let seed_scores = vec![challenger_scorecard.aggregate_score; 3];

    let gk = Gatekeeper::new(GatekeeperConfig::default());
    let decision = gk.evaluate(
        run_id,
        run.agent_id,
        champion_scorecard.as_ref(),
        &challenger_scorecard,
        &champion_passing,
        &challenger_traces,
        &seed_scores,
    )?;

    println!(
        "\nPromotion Decision: {}",
        if decision.approved {
            "APPROVED ✅"
        } else {
            "DENIED ❌"
        }
    );
    println!();
    for gate in &decision.gates {
        let sym = match gate.status {
            GateStatus::Pass => "✅",
            GateStatus::Fail => "❌",
            GateStatus::Waived => "⏭",
        };
        println!("{} {} — {}", sym, gate.gate, gate.message);
    }
    println!("\n{}", decision.changelog);

    if decision.approved {
        agent_repo
            .set_champion(challenger_agent.id, &challenger_agent.name)
            .await?;
        agent_repo
            .update_changelog(challenger_agent.id, &decision.changelog)
            .await?;
        println!("\nAgent promoted to champion.");
        Ok(0)
    } else {
        Ok(1)
    }
}

async fn cmd_scores(run_id: Uuid) -> Result<i32> {
    let db = require_db().await?;
    let eval_repo = EvalRepo::new(db);
    let run = eval_repo
        .find_by_id(run_id)
        .await
        .map_err(|_| anyhow::anyhow!("Run {run_id} not found"))?;

    let scorecard = run
        .to_scorecard()
        .ok_or_else(|| anyhow::anyhow!("Run {run_id} has no scores yet"))?;

    print_scorecard(&scorecard);
    Ok(0)
}

fn print_scorecard(sc: &agentforge_core::Scorecard) {
    println!("\n╔══════════════════════════════════════╗");
    println!("║          AgentForge Scorecard         ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  Agent:      {} v{}", sc.agent_name, sc.agent_version);
    println!("║  Run ID:     {}", sc.run_id);
    println!("╠══════════════════════════════════════╣");
    println!("║  Aggregate:  {:.3}", sc.aggregate_score);
    println!(
        "║  Pass Rate:  {:.1}% ({}/{} scenarios)",
        sc.pass_rate * 100.0,
        sc.passed,
        sc.total_scenarios
    );
    println!("╠══════════════════════════════════════╣");
    println!("║  Dimension Scores:                   ║");
    println!(
        "║    Task Completion:    {:.3}",
        sc.dimension_scores.task_completion
    );
    println!(
        "║    Tool Selection:     {:.3}",
        sc.dimension_scores.tool_selection
    );
    println!(
        "║    Arg Correctness:    {:.3}",
        sc.dimension_scores.argument_correctness
    );
    println!(
        "║    Schema Compliance:  {:.3}",
        sc.dimension_scores.schema_compliance
    );
    println!(
        "║    Instr. Adherence:   {:.3}",
        sc.dimension_scores.instruction_adherence
    );
    println!(
        "║    Path Efficiency:    {:.3}",
        sc.dimension_scores.path_efficiency
    );
    if !sc.failure_clusters.is_empty() {
        println!("╠══════════════════════════════════════╣");
        println!("║  Failure Clusters:");
        for cluster in &sc.failure_clusters {
            println!(
                "║    {:?}: {} ({:.0}%)",
                cluster.cluster,
                cluster.count,
                cluster.percentage * 100.0
            );
        }
    }
    println!("╠══════════════════════════════════════╣");
    println!(
        "║  Duration: {}s  Tokens: {}in/{}out",
        sc.duration_seconds, sc.total_input_tokens, sc.total_output_tokens
    );
    println!("╚══════════════════════════════════════╝");
}

fn build_scorer_config(
    judge_provider: &str,
    weights: agentforge_core::EvalWeights,
) -> ScorerConfig {
    let (default_base_url, default_api_key, default_model) = match judge_provider {
        "anthropic" => (
            "https://api.anthropic.com/v1",
            std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            "claude-3-5-sonnet-20241022".to_string(),
        ),
        "nvidia" => (
            "https://integrate.api.nvidia.com/v1",
            std::env::var("NVIDIA_API_KEY").unwrap_or_default(),
            "meta/llama-3.1-70b-instruct".to_string(),
        ),
        _ => (
            "https://api.openai.com/v1",
            std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            "gpt-4o".to_string(),
        ),
    };
    ScorerConfig {
        judge_model: std::env::var("AGENTFORGE_JUDGE_MODEL").unwrap_or(default_model),
        judge_base_url: std::env::var("AGENTFORGE_JUDGE_BASE_URL")
            .unwrap_or_else(|_| default_base_url.to_string()),
        judge_api_key: default_api_key,
        weights,
        ..Default::default()
    }
}

fn build_llm_client(provider: &str) -> Result<std::sync::Arc<dyn agentforge_runner::LlmClient>> {
    match provider {
        "anthropic" => Ok(std::sync::Arc::new(
            AnthropicClient::from_env()
                .ok_or_else(|| anyhow::anyhow!("ANTHROPIC_API_KEY must be set"))?,
        )
            as std::sync::Arc<dyn agentforge_runner::LlmClient>),
        "nvidia" => Ok(std::sync::Arc::new(
            NvidiaClient::from_env()
                .ok_or_else(|| anyhow::anyhow!("NVIDIA_API_KEY must be set"))?,
        )
            as std::sync::Arc<dyn agentforge_runner::LlmClient>),
        _ => Ok(std::sync::Arc::new(
            OpenAiClient::from_env()
                .ok_or_else(|| anyhow::anyhow!("OPENAI_API_KEY must be set"))?,
        )
            as std::sync::Arc<dyn agentforge_runner::LlmClient>),
    }
}

async fn require_db() -> Result<agentforge_db::PgPool> {
    let url =
        std::env::var("DATABASE_URL").context("DATABASE_URL must be set for database commands")?;
    create_pool(&url)
        .await
        .context("Failed to connect to database")
}

/// v2: Shadow run — compare champion vs. candidate on live traffic.
async fn cmd_shadow(champion_id: Uuid, candidate_id: Uuid, traffic_percent: u8) -> Result<i32> {
    let db = require_db().await?;
    let shadow_repo = ShadowRepo::new(db.clone());
    let agent_repo = AgentRepo::new(db);

    let champion = agent_repo
        .find_by_id(champion_id)
        .await
        .map_err(|_| anyhow::anyhow!("Champion agent {champion_id} not found"))?;
    let candidate = agent_repo
        .find_by_id(candidate_id)
        .await
        .map_err(|_| anyhow::anyhow!("Candidate agent {candidate_id} not found"))?;

    println!(
        "Shadow run: {} v{} (champion) vs {} v{} (candidate) — {}% traffic",
        champion.name, champion.version, candidate.name, candidate.version, traffic_percent
    );

    let run = agentforge_core::ShadowRun {
        id: Uuid::new_v4(),
        champion_agent_id: champion_id,
        candidate_agent_id: candidate_id,
        traffic_percent: traffic_percent.clamp(1, 100),
        status: agentforge_core::ShadowRunStatus::Pending,
        comparison: None,
        error_message: None,
        created_at: chrono::Utc::now(),
        started_at: None,
        completed_at: None,
    };

    let saved = shadow_repo.insert(&run).await?;
    println!("Shadow run created: {} (status: pending)", saved.id);
    println!("Use `GET /shadow-runs/{}` to check progress.", saved.id);
    Ok(0)
}

/// v2: Export traces as fine-tuning dataset.
async fn cmd_export(run_id: Uuid, format_str: String, output: Option<PathBuf>) -> Result<i32> {
    use agentforge_core::ExportFormat;
    use agentforge_finetune::FineTuneExporter;

    let db = require_db().await?;
    let trace_repo = TraceRepo::new(db.clone());
    let finetune_repo = FineTuneRepo::new(db);

    let format: ExportFormat = format_str.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    println!("Loading traces for run {}...", run_id);
    let traces = trace_repo
        .list_by_run(run_id)
        .await
        .with_context(|| "Failed to load traces")?;

    let records = FineTuneExporter::export(&traces, &format)
        .map_err(|e| anyhow::anyhow!("Export failed: {e}"))?;

    let jsonl = FineTuneExporter::to_jsonl(&records)
        .map_err(|e| anyhow::anyhow!("Serialization failed: {e}"))?;

    match output {
        Some(path) => {
            std::fs::write(&path, &jsonl)
                .with_context(|| format!("Failed to write to {}", path.display()))?;
            println!(
                "Exported {} records ({} format) → {}",
                records.len(),
                format,
                path.display()
            );

            // Update DB record if it exists
            let export_record = agentforge_core::FineTuneExport {
                id: Uuid::new_v4(),
                run_id,
                format: format.clone(),
                status: agentforge_core::ExportStatus::Complete,
                row_count: Some(records.len() as u32),
                file_path: Some(path.to_string_lossy().to_string()),
                error_message: None,
                created_at: chrono::Utc::now(),
                completed_at: Some(chrono::Utc::now()),
            };
            let _ = finetune_repo.insert(&export_record).await;
        }
        None => {
            println!("{jsonl}");
        }
    }

    Ok(0)
}

/// v2: Benchmark — run agent against a benchmark suite.
async fn cmd_benchmark(agent_path: PathBuf, suite_str: String, tasks_path: PathBuf) -> Result<i32> {
    use agentforge_benchmarks::{
        agentbench, gaia, webarena, BenchmarkRunner, BenchmarkRunnerConfig,
    };
    use agentforge_core::BenchmarkSuite;

    let content = std::fs::read_to_string(&agent_path)
        .with_context(|| format!("Failed to read agent file: {}", agent_path.display()))?;
    let parsed = parse_agent_file(&content).with_context(|| "Failed to parse agent file")?;
    let agent_file = parsed.agent.clone();

    let tasks_content = std::fs::read_to_string(&tasks_path)
        .with_context(|| format!("Failed to read tasks file: {}", tasks_path.display()))?;

    let suite = match suite_str.to_lowercase().as_str() {
        "gaia" => BenchmarkSuite::Gaia,
        "agentbench" => BenchmarkSuite::AgentBench,
        "webarena" => BenchmarkSuite::WebArena,
        other => anyhow::bail!("Unknown benchmark suite: {other}"),
    };

    let tasks = match &suite {
        BenchmarkSuite::Gaia => gaia::load_from_jsonl(&tasks_content),
        BenchmarkSuite::AgentBench => agentbench::load_from_jsonl(&tasks_content),
        BenchmarkSuite::WebArena => webarena::load_from_jsonl(&tasks_content),
    };

    println!(
        "Benchmark: {} suite — {} tasks loaded from {}",
        suite,
        tasks.len(),
        tasks_path.display()
    );

    let llm = build_llm_client(
        &std::env::var("AGENTFORGE_JUDGE_PROVIDER").unwrap_or_else(|_| "openai".to_string()),
    )?;
    let agent_id = Uuid::new_v5(&Uuid::NAMESPACE_DNS, agent_file.name.as_bytes());

    let runner = BenchmarkRunner::new(
        llm,
        BenchmarkRunnerConfig {
            suite: suite.clone(),
            agent_id,
            runner_config: RunnerConfig::default(),
        },
    );

    println!("Running agent on {} tasks...", tasks.len());
    let run = runner
        .run(&agent_file, tasks)
        .await
        .with_context(|| "Benchmark run failed")?;

    println!("\n╔══════════════════════════════════════╗");
    println!("║        Benchmark Results              ║");
    println!("╠══════════════════════════════════════╣");
    println!("║  Suite:    {}", run.suite);
    println!("║  Tasks:    {}/{} correct", run.correct, run.total_tasks);
    println!("║  Accuracy: {:.1}%", run.accuracy * 100.0);
    if let Some(pct) = run.percentile_rank {
        println!("║  Percentile: {:.0}th vs. published baselines", pct);
    }
    println!("╚══════════════════════════════════════╝");

    Ok(0)
}
