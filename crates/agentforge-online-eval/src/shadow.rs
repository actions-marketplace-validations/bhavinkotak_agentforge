use agentforge_core::{AgentFile, Result, Scenario, Trace};
use agentforge_runner::{AgentRunner, LlmClient, RunnerConfig};
use agentforge_scorer::ScorerConfig;
use rand::RngExt;
use std::sync::Arc;
use uuid::Uuid;

/// Configuration for a shadow run.
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Fraction of traffic to send to the candidate (0.0–1.0). Default: 0.1.
    pub traffic_fraction: f64,
    /// Random seed for reproducible traffic splitting.
    pub seed: u64,
    pub runner: RunnerConfig,
    pub scorer: ScorerConfig,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            traffic_fraction: 0.1,
            seed: 42,
            runner: RunnerConfig::default(),
            scorer: ScorerConfig::default(),
        }
    }
}

/// Output of a shadow run: traces from both champion and candidate.
pub struct ShadowRunOutput {
    pub run_id: Uuid,
    pub champion_traces: Vec<Trace>,
    pub candidate_traces: Vec<Trace>,
    /// Scenarios sent to both agents (the overlapping subset).
    pub shared_scenarios: Vec<Scenario>,
}

/// Runs champion and candidate in parallel on the same traffic sample.
pub struct ShadowRunner {
    champion_llm: Arc<dyn LlmClient>,
    candidate_llm: Arc<dyn LlmClient>,
    config: ShadowConfig,
}

impl ShadowRunner {
    pub fn new(
        champion_llm: Arc<dyn LlmClient>,
        candidate_llm: Arc<dyn LlmClient>,
        config: ShadowConfig,
    ) -> Self {
        Self {
            champion_llm,
            candidate_llm,
            config,
        }
    }

    /// Run both agents on the provided scenarios.
    ///
    /// The `traffic_fraction` from config determines what share of `scenarios`
    /// is routed to the candidate. Both champion and candidate always see the
    /// same sampled subset so results are directly comparable.
    pub async fn run(
        &self,
        champion: &AgentFile,
        candidate: &AgentFile,
        scenarios: Vec<Scenario>,
    ) -> Result<ShadowRunOutput> {
        let run_id = Uuid::new_v4();

        // Sample the subset that goes to both agents.
        let shared = self.sample_scenarios(&scenarios);

        tracing::info!(
            run_id = %run_id,
            total = scenarios.len(),
            sampled = shared.len(),
            traffic_fraction = self.config.traffic_fraction,
            "Starting shadow run"
        );

        let champion_runner = AgentRunner::new(
            self.champion_llm.clone(),
            RunnerConfig {
                run_id,
                ..self.config.runner.clone()
            },
        );

        let candidate_runner = AgentRunner::new(
            self.candidate_llm.clone(),
            RunnerConfig {
                run_id,
                ..self.config.runner.clone()
            },
        );

        let (champion_result, candidate_result) = tokio::join!(
            champion_runner.run(champion, shared.clone(), None),
            candidate_runner.run(candidate, shared.clone(), None),
        );

        Ok(ShadowRunOutput {
            run_id,
            champion_traces: champion_result.traces,
            candidate_traces: candidate_result.traces,
            shared_scenarios: shared,
        })
    }

    /// Deterministically sample `traffic_fraction` of the scenarios using the seed.
    fn sample_scenarios(&self, scenarios: &[Scenario]) -> Vec<Scenario> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        scenarios
            .iter()
            .filter(|_| rng.random::<f64>() < self.config.traffic_fraction)
            .cloned()
            .collect()
    }
}
