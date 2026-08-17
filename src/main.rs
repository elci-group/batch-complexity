use std::io::{self, Read, Write};
use std::process::ExitCode;

use batch_complexity::{Message, score};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "batch-complexity")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Reads `{"messages":[{"role":...,"content":...}]}` on stdin, writes a
    /// complexity score to stdout as JSON. Non-zero exit + stderr on any
    /// malformed input — callers should treat that as "unknown" and fall
    /// back to the safe default model, the same fail-open pattern goblin's
    /// preflight already uses.
    Score,
}

#[derive(Deserialize)]
struct ScoreRequest {
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct ScoreResponse {
    status: &'static str,
    result: batch_complexity::ComplexityScore,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::Score => run_score(),
    }
}

fn run_score() -> ExitCode {
    let mut input = String::new();
    if let Err(error) = io::stdin().read_to_string(&mut input) {
        eprintln!("batch-complexity: failed to read stdin: {error}");
        return ExitCode::FAILURE;
    }
    let request: ScoreRequest = match serde_json::from_str(&input) {
        Ok(request) => request,
        Err(error) => {
            eprintln!("batch-complexity: invalid input JSON: {error}");
            return ExitCode::FAILURE;
        }
    };
    if request.messages.is_empty() {
        eprintln!("batch-complexity: at least one message is required");
        return ExitCode::FAILURE;
    }
    let result = score(&request.messages);
    let response = ScoreResponse {
        status: "success",
        result,
    };
    match serde_json::to_string(&response) {
        Ok(json) => {
            if let Err(error) = io::stdout().write_all(json.as_bytes()) {
                eprintln!("batch-complexity: failed to write stdout: {error}");
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("batch-complexity: failed to serialize result: {error}");
            ExitCode::FAILURE
        }
    }
}
