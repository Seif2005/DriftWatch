# AWS AI/ML Engineering Portfolio Blueprint
### Two production-grade projects aligned to MLA-C01, built for an Agentic AI engineer moving deeper into AWS

---

## 0. MLA-C01 Analysis (grounding for everything below)

The exam guide (docs.aws.amazon.com, AWS Certification) defines four domains:

| Domain | Weight | Focus |
|---|---|---|
| 1. Data Preparation for ML | 28% | Ingest, transform, validate, prepare data |
| 2. ML Model Development | 26% | Select approach, train, tune, evaluate, version |
| 3. Deployment & Orchestration of ML Workflows | 22% | Infra, endpoints, autoscaling, CI/CD orchestration |
| 4. ML Solution Monitoring, Maintenance & Security | 24% | Monitoring, drift, security, cost, retraining |

Important recent development: the exam refresh added explicit **generative AI, agentic AI, foundation models, and Amazon Bedrock** content into these same four domains — it did not add a fifth domain. This is exactly why one Bedrock-centric project and one classic-ML/SageMaker project together cover the certification more completely than two similar projects would.

**One correction to your original scope, based on current AWS docs**: Amazon Bedrock Agents ("Agents Classic") stopped accepting new customers on **July 30, 2026** and is in maintenance mode. AWS is directing new builds to **Amazon Bedrock AgentCore**. Project 1 below is designed around AgentCore, not Agents Classic — this is more current, more resume-relevant, and arguably a better engineering story since AgentCore is more primitives-based (you assemble the runtime, gateway, memory, and identity pieces yourself rather than getting one wizard-configured "Agent" resource), which fits your stated goal of understanding what's under the hood.

**Where your Agentic AI background helps**: you likely already understand tool-calling loops, planning/orchestration, and prompt design. The gaps MLA-C01 + industry practice will care about, that a pure agent builder often lacks, are: AWS IAM least-privilege design, model evaluation methodology (not just "it seems to work"), observability/cost telemetry, guardrail/safety engineering, and classical ML lifecycle (data → training → registry → monitored deployment → retraining). The two projects are scoped to fill exactly those gaps.

---

## 1. Project 1 — "ClaimGuard": An Agentic Insurance-Claims Triage & Investigation System on Bedrock AgentCore

### 1.1 Elevator pitch
A multi-agent system that ingests an insurance claim (structured form + free-text description + optional photos), retrieves relevant policy and historical-claims context, reasons about fraud risk and coverage validity, calls real tools (policy lookup, fraud-score API, weather-history API for storm claims, document OCR), and produces a structured adjudication recommendation with citations — with guardrails, human-in-the-loop escalation, full observability, and a measured hallucination/accuracy rate.

### 1.2 Problem statement
Insurance claims triage is manual, slow, and inconsistent. An agent that can gather context, apply policy rules, flag anomalies, and produce an auditable recommendation (never an autonomous denial) demonstrates a realistic enterprise generative-AI use case: high stakes, need for grounding, need for guardrails, need for human oversight.

### 1.3 Why it's commercially relevant
Insurance/claims/fraud triage is one of the most cited real-world Bedrock case studies (grounding-heavy, compliance-heavy, human-in-the-loop required) — it forces you to build the parts AWS console demos skip: guardrails tuned to a domain, structured tool outputs, and evaluation against ground truth, not just a demo chat.

### 1.4 Architecture (textual — see diagram below)
```
Client (web form / API) 
   -> API Gateway (REST) -> Lambda (request validation, auth)
   -> AgentCore Runtime (orchestrator agent, session-scoped)
        -> AgentCore Gateway -> Action Groups / Tools:
             - PolicyLookupTool   (Lambda -> DynamoDB)
             - ClaimsHistoryTool  (Lambda -> DynamoDB / OpenSearch)
             - DocumentOCRTool    (Lambda -> Amazon Textract)
             - WeatherHistoryTool (Lambda -> external API, cached in S3)
             - FraudScoreTool     (Lambda -> your own lightweight scoring
                                    model — this is where Project 2's
                                    skills reappear, deliberately)
        -> Bedrock Knowledge Base (policy documents, underwriting guidelines)
             -> OpenSearch Serverless (vector store)
        -> Bedrock Guardrails (denied topics, PII redaction, grounding
             check on every generation)
        -> Bedrock model (Claude on Bedrock — see 1.11)
   -> Step Functions (for the multi-step / human-in-the-loop escalation
        path — this is the orchestration layer you build yourself)
   -> DynamoDB (claim + decision + audit trail)
   -> CloudWatch + X-Ray (observability, tracing, custom metrics)
   -> SNS (escalation to human adjuster)
```

### 1.5 Architecture diagram
(rendered separately below as an inline diagram)

### 1.6 Request/data flow
1. Claim submitted via API Gateway → validated & stored (DynamoDB, status=`received`).
2. Orchestrator agent invoked with claim ID. It plans: "gather policy, gather history, check for red flags, retrieve relevant guideline passages, produce a recommendation."
3. Agent calls tools through the AgentCore Gateway (each tool is a real Lambda with a JSON schema, not a mock). Tool results are appended to context.
4. Agent queries the Knowledge Base for relevant underwriting/policy language; RetrieveAndGenerate returns grounded citations.
5. Guardrail evaluates both the retrieved content usage and the draft response (denied topics: medical diagnosis, legal advice; PII redaction; contextual grounding check to catch hallucinated policy clauses).
6. If fraud score or confidence is below threshold → Step Functions routes to a human-review state (SNS notification), else the recommendation is finalized.
7. Full trace (prompts, tool calls, guardrail interventions, latency, token cost) is logged to CloudWatch and queryable.

### 1.7 AWS services used and why
- **Bedrock (model access)** — Core. The reasoning engine.
- **Bedrock AgentCore (Runtime, Gateway, Identity, Memory)** — Core. Current, supported way to build production agents; teaches session/identity isolation and tool-gateway design instead of a black-box "Agent" resource.
- **Bedrock Knowledge Bases + OpenSearch Serverless** — Core. Managed RAG; you still own chunking strategy and evaluation.
- **Bedrock Guardrails** — Core. Direct MLA-C01 "responsible AI" and domain-4 security content.
- **Lambda** — Core. Every tool; this is where you write real code.
- **DynamoDB** — Core. Claim state, audit trail, policy data.
- **Step Functions** — Recommended. Orchestrates the human-in-the-loop branch; teaches you a second orchestration pattern distinct from the agent's own reasoning loop.
- **Textract** — Recommended. Real OCR on uploaded documents, not a toy.
- **API Gateway** — Core. Front door.
- **CloudWatch + X-Ray** — Core. Observability is an explicit exam domain (4) and a real production requirement.
- **IAM** — Core. Every Lambda/tool gets its own least-privilege role.
- **SNS** — Optional. Human escalation notification.
- **Secrets Manager** — Recommended if using any external API key (weather).

Do **not** add: SageMaker here (that's Project 2's job — keep this project GenAI-pure), Bedrock Data Automation (adds breadth without new skill given Textract already covers OCR), Bedrock Flows (a visual-builder abstraction that would hide the orchestration logic you're trying to demonstrate you can write yourself).

### 1.8 What you implement yourself
- Tool Lambdas and their JSON schemas (real code).
- Orchestrator prompt/planning logic and Step Functions state machine.
- Guardrail policy configuration and testing harness (adversarial prompt set).
- Evaluation harness (golden dataset of claims with expected outcomes + citation accuracy scoring).
- Cost/latency instrumentation (custom CloudWatch metrics per agent turn: tokens in/out, tool latency, guardrail latency).
- A lightweight fraud-scoring **model of your own** (simple gradient-boosted classifier, trained on a synthetic/public claims-fraud dataset) exposed as a tool — this is the deliberate bridge to classical ML/Project 2 skills, and it stops the agent's "fraud check" from being a fake LLM guess.

### 1.9 Where you use the managed abstraction vs. build it yourself
| Capability | Recommendation |
|---|---|
| Vector retrieval | Use managed Knowledge Bases + OpenSearch Serverless — reimplementing chunking/embedding pipelines from scratch teaches less than tuning the managed one well (chunk size, overlap, hybrid search, reranking) |
| Content safety filtering | Use Guardrails — reimplementing a safety classifier is out of scope and duplicates AWS's job |
| Agent orchestration loop | **Compare both**: build a minimal manual tool-calling loop yourself against Claude's Converse API first (so you understand ReAct-style looping), then migrate to AgentCore Runtime for the "real" system. This comparison is one of your strongest interview talking points. |
| Fraud scoring | Build yourself (small XGBoost/scikit-learn model) — this is the one place a real trained model belongs in Project 1 |
| Human-in-the-loop routing | Build yourself with Step Functions — AWS has no single managed abstraction for this, so it's inherently hands-on |

### 1.10 Recommended language/framework
Python 3.12. `boto3` for AWS calls, FastAPI (or plain Lambda handlers) for the API layer, `pytest` for tests, AWS SAM or CDK (Python) for infrastructure as code. Use the Bedrock **Converse API** (not the older InvokeModel per-provider format) for portability.

### 1.11 Recommended Bedrock models
- **Claude (Bedrock-hosted, e.g. a current Claude Sonnet/Haiku tier)** for the orchestrator — strong tool-use and instruction-following, native Converse API tool support.
- A cheaper/faster model (e.g., Amazon Nova Lite or Claude Haiku tier) for the OCR-summarization and low-stakes sub-tasks, to demonstrate deliberate cost/latency tiering rather than using one expensive model for everything.

### 1.12 Agent architecture & tool design
Single orchestrator agent (not a swarm) with 4–5 well-scoped tools, each with a strict JSON schema, explicit error contracts (tool failures must return structured errors the agent can reason about, not raise exceptions that crash the run), and idempotency where relevant (e.g., re-running FraudScoreTool on the same claim ID returns cached result). Keep it single-agent-with-tools rather than multi-agent-swarm: multi-agent adds orchestration complexity that's easy to fake and hard to evaluate meaningfully — a well-designed single agent with real tools is the stronger, more honest engineering demonstration for a portfolio piece.

### 1.13 RAG architecture
Source docs: a small corpus of realistic (synthetic, since real policy docs aren't public) underwriting guideline PDFs you write or adapt from public insurance-regulation text. Chunk by section heading (structure-aware chunking, not fixed-token windows), store chunk metadata (source doc, section, effective date) for citation. Evaluate retrieval precision/recall against a hand-labeled question set before wiring it into the agent.

### 1.14 Data sources/datasets
- Synthetic claims dataset (you generate ~500–1000 synthetic claims with an LLM, following a schema you design, then hand-label ~50 for eval ground truth).
- Public insurance fraud dataset for training the small fraud-scoring model, e.g. **Vehicle Insurance Claim Fraud Detection** (Kaggle, ~15k rows) — used only for the Project 1 fraud tool, not as the whole project.

### 1.15 Evaluation strategy
- **Task accuracy**: agent recommendation vs. your labeled ground truth on the golden set (precision/recall on "escalate to human" decisions).
- **Faithfulness/grounding**: does every policy claim in the output have a valid citation from the Knowledge Base? (score manually on a sample + automate with an LLM-as-judge check, reported with its own limitations noted).
- **Guardrail effectiveness**: run an adversarial prompt set (prompt injection via claim free-text field, PII leakage attempts) and report block rate.
- **Latency/cost per claim**: p50/p95 latency, tokens per claim, $ per claim — track over the eval set.
- Use **Bedrock Model Evaluation** jobs for the underlying model comparison (e.g., compare two candidate models on your golden set) as a Core exam-relevant piece.

### 1.16 Security architecture
- All external input (claim text) treated as untrusted — prompt-injection test suite required before you consider the tool-calling safe.
- Guardrails on both input and output.
- No tool has broader permissions than its single job (see IAM below).
- Secrets (external API keys) in Secrets Manager, never in code or env files committed to git.
- VPC endpoints for Bedrock/DynamoDB/S3 if you want to demonstrate network isolation (optional, adds real IAM/networking depth if you have the hours).

### 1.17 IAM / least privilege
Each Lambda tool gets its own execution role scoped to only the resources it touches (e.g., PolicyLookupTool's role can `dynamodb:GetItem` on the policy table only, nothing else). The orchestrator's AgentCore identity is scoped to invoke only the specific Gateway/tools it needs, not account-wide Bedrock access. Write this as an explicit least-privilege IAM policy document per tool — this is a direct, gradable exam-domain-4 artifact for your portfolio.

### 1.18 Monitoring & observability
Custom CloudWatch metrics: tokens/claim, tool latency per tool, guardrail intervention rate, escalation rate. X-Ray tracing across the Lambda/Bedrock/Step Functions call graph. A CloudWatch dashboard as a deliverable artifact (screenshot it for your portfolio).

### 1.19 Failure handling
- Tool timeout/error → agent falls back to "insufficient information, escalate to human" rather than guessing.
- Guardrail block → structured refusal with reason code, logged.
- Model unavailable (throttling) → retry with backoff, then fall back to a secondary model ID.

### 1.20 Cost/latency optimization
- Prompt caching for the repeated system prompt / tool schema portion of every call.
- Tiered models (cheap model for OCR summarization, stronger model only for the final adjudication reasoning).
- Batch inference for the offline evaluation runs (cheaper than on-demand).

### 1.21 Expected AWS cost — development/testing
Roughly **$15–40 total** for a few weeks of iterative dev: Bedrock on-demand tokens for a project this size are cents to low dollars per day of active testing (Nova Lite / Haiku-tier pricing is sub-$1 per million tokens; Claude Sonnet-tier is a few dollars per million); OpenSearch Serverless has an hourly minimum (~$0.24/hr per OCU-pair) — **this is the single biggest cost risk in Project 1**, budget for it and tear it down when not actively working; Lambda/DynamoDB/API Gateway are effectively free at this scale (free tier).

### 1.22 Expected monthly cost if deployed as a small production system
Roughly **$50–150/month** for light real traffic (few hundred claims/month): OpenSearch Serverless minimum capacity is the dominant fixed cost (~$175/month if left running 24/7 at minimum OCUs — consider Aurora pgvector instead for a portfolio deployment to cut this significantly, or only run OpenSearch during demo windows), Bedrock token costs scale with volume and stay low at this scale, everything else (Lambda, DynamoDB, API Gateway, CloudWatch) stays in low single digits.

**Biggest bill risk**: leaving OpenSearch Serverless collections running continuously, and leaving provisioned-throughput Bedrock capacity (if you ever test it) running. Use on-demand Bedrock, delete/pause the OpenSearch collection between work sessions, and set a CloudWatch billing alarm at $20 as a tripwire.

### 1.23 Step-by-step implementation plan
See the Roadmap in section 3.

### 1.24 Suggested repo structure
```
claimguard/
  infra/                # CDK or SAM (IaC)
  agent/
    orchestrator/        # planning prompts, Converse API loop / AgentCore config
    tools/                # one folder per Lambda tool
  knowledge_base/
    docs/                 # source policy documents
    ingestion/            # chunking scripts
  fraud_model/
    train.py
    model.pkl
  eval/
    golden_set.jsonl
    run_eval.py
    adversarial_prompts.jsonl
  api/                    # API Gateway/Lambda handlers
  tests/
  docs/
    architecture.md
    runbook.md
  README.md
```

### 1.25 Tests to write
Unit tests per tool Lambda (mocked AWS calls), contract tests for tool JSON schemas, an integration test that runs a claim end-to-end against a test AWS account, guardrail regression tests (adversarial prompt set must stay blocked across changes), evaluation-set regression test (accuracy must not silently regress below a threshold when you change prompts).

### 1.26 Final portfolio demo
A 3–5 minute recorded walkthrough: submit a claim, show the agent's tool calls and citations live in CloudWatch/X-Ray, show a guardrail catching an injected prompt, show the CloudWatch cost/latency dashboard, show one claim being escalated to a human. This is far more convincing than a chat UI alone.

### 1.27 Resume bullets
- "Designed and built a multi-tool agentic claims-triage system on Amazon Bedrock AgentCore with RAG-grounded reasoning, custom Guardrails policies, and Step Functions-based human-in-the-loop escalation, achieving [X]% grounding accuracy on a labeled evaluation set."
- "Implemented least-privilege IAM across 5+ Lambda-based agent tools and built a CloudWatch/X-Ray observability stack tracking per-request token cost, latency, and guardrail intervention rate."
- "Built and integrated a supervised fraud-scoring model as an agent tool, bridging classical ML and generative-AI reasoning in a single production workflow."

### 1.28 MLA-C01 objectives reinforced
Generative AI/agentic content across Domains 1–4: prompt engineering & RAG (Domain 2), deployment/orchestration of an agentic workflow (Domain 3), guardrails/security/observability/cost (Domain 4), and even a slice of Domain 1 (data prep for the Knowledge Base corpus and the fraud model's training data).

---

## 2. Project 2 — "DriftWatch": End-to-End SageMaker Pipeline for Customer-Churn Prediction with Automated Retraining

### 2.1 Elevator pitch
A fully automated SageMaker Pipeline that ingests raw customer data, engineers features, trains and tunes a real gradient-boosted model, registers it with approval gating, deploys it behind a monitored real-time endpoint, and automatically retrains when SageMaker Model Monitor detects data drift — with CI/CD triggering the whole pipeline from a git push.

### 2.2 Problem statement
Churn prediction is a canonical, well-understood classical ML problem — which is exactly why it's the right vehicle here: it lets the *engineering* (pipeline, registry, monitoring, retraining, CI/CD) be the hard part, not the modeling.

### 2.3 Commercial relevance
Churn prediction is one of the highest-ROI applied-ML use cases in industry (subscription businesses, telecom, SaaS); the "full lifecycle with monitored retraining" pattern is exactly what companies pay ML engineers to build and operate.

### 2.4 Dataset
**Telco Customer Churn dataset** (originally IBM sample dataset, widely mirrored on Kaggle as "Telco Customer Churn"). ~7,043 rows, ~21 features (tenure, contract type, monthly charges, internet service, payment method, etc.), binary target `Churn` (Yes/No). Public, permissively licensed for learning/portfolio use (verify the specific mirror's license badge before publishing your repo). Reasonable size and feature richness for real preprocessing/feature-engineering work without being a "toy" (Iris/Titanic-tier) dataset — it has genuine class imbalance (~26% churn), mixed categorical/numeric features, and realistic messiness (a `TotalCharges` column that's technically string-typed with blank values), which forces real data-cleaning decisions.

- **Preprocessing**: handle the blank/whitespace `TotalCharges` values (cast to numeric, impute or drop), one-hot/target-encode categoricals, scale numeric features.
- **Split**: stratified 70/15/15 train/val/test on the churn label to preserve class balance across splits.
- **Problem type**: binary classification with class imbalance.
- **Baseline model**: logistic regression (interpretable, fast, a real baseline to beat).
- **Stronger models to compare**: XGBoost (via SageMaker's built-in XGBoost container) and a scikit-learn RandomForest, tuned with SageMaker Automatic Model Tuning (hyperparameter tuning jobs).
- **Metrics**: since classes are imbalanced, prioritize **PR-AUC and F1** on the minority (churn) class over raw accuracy; report ROC-AUC too for comparability, and pick your deployment threshold from a precision/recall tradeoff analysis, not the default 0.5.

### 2.5 SageMaker architecture
```
S3 (raw data) 
  -> SageMaker Processing job (cleaning, feature engineering) -> S3 (features)
  -> SageMaker Pipelines orchestrates:
       ProcessingStep -> TrainingStep (XGBoost, with
         HyperparameterTuningStep) -> EvaluationStep (Processing job
         computing PR-AUC/F1 on held-out test set)
       -> ConditionStep (only proceed if metric beats current
            registered model's metric)
       -> RegisterModelStep (SageMaker Model Registry, pending-approval
            status)
  -> Manual/automated approval -> Deployment (real-time SageMaker
       endpoint with autoscaling, behind an API Gateway/Lambda for a
       simple prediction API)
  -> SageMaker Model Monitor (data quality + model quality monitoring
       schedules) -> CloudWatch alarms on drift
  -> EventBridge rule (drift alarm) -> triggers SageMaker Pipeline
       re-run (automated retraining loop)
  -> CodePipeline/CodeBuild (CI/CD: git push -> lint/test -> trigger
       SageMaker Pipeline execution)
```

### 2.6 SageMaker Pipeline stages (detailed)
1. **Processing (cleaning/feature engineering)** — scikit-learn Processing container running your own preprocessing script.
2. **Training** — XGBoost built-in algorithm container, your own hyperparameter ranges.
3. **Hyperparameter tuning** — SageMaker Automatic Model Tuning (Bayesian) over `max_depth`, `eta`, `subsample`, `scale_pos_weight` (important given class imbalance).
4. **Evaluation** — a Processing step that loads the test set and computes PR-AUC/F1, writes a metrics JSON.
5. **Condition** — pipeline only registers the model if it beats the previous best on the validation metric (this is where you demonstrate real MLOps judgment, not just "always deploy the newest").
6. **Register** — SageMaker Model Registry, model package group, with the evaluation metrics attached as model card content.
7. **Deploy** — a separate deploy script/pipeline (real-time endpoint + optionally a batch transform job for a bulk-scoring demo) triggered on manual approval in the registry.

### 2.7 AWS services used and why
- **SageMaker Processing** — Core. Real feature-engineering code running in a managed container; the "we want to learn X" concept here is separating data prep from notebooks into reproducible, scaled jobs.
- **SageMaker Training Jobs** — Core. Actual model training on managed infra.
- **SageMaker Automatic Model Tuning** — Recommended. Teaches you HPO at scale, not just `GridSearchCV` locally.
- **SageMaker Pipelines** — Core. This *is* the MLOps engineering artifact — orchestration-as-code for the ML lifecycle.
- **SageMaker Model Registry** — Core. Versioning + approval gating.
- **SageMaker Endpoints (real-time) + Batch Transform** — Core. Both inference patterns, so you can speak to the tradeoff from experience.
- **SageMaker Model Monitor** — Core. Direct Domain-4 exam content (drift detection).
- **S3** — Core. Data/artifact storage.
- **IAM** — Core. Pipeline execution role and endpoint role, least privilege.
- **CloudWatch** — Core. Metrics, alarms, drift-triggered alerts.
- **EventBridge** — Recommended. Wires monitoring alarms to automated retraining — this is the piece that turns "I trained a model" into "I built a self-healing ML system."
- **ECR** — Recommended, mainly implicit (SageMaker manages this for built-in containers; becomes explicit/Core if you bring your own container for the custom feature-engineering logic).
- **CodePipeline/CodeBuild** — Recommended. CI/CD from git push to pipeline execution.
- **CloudFormation/CDK** — Recommended for IaC of the whole stack (endpoint config, monitor schedules, EventBridge rules).

Do **not** add: SageMaker Feature Store (real value at team/multi-model scale; for one dataset and one model it's console tourism), SageMaker Ground Truth (no labeling task here), SageMaker Studio notebooks-as-the-deliverable (fine for exploration, but the actual deliverable must be pipeline code, not a notebook).

### 2.8 What you implement yourself
Preprocessing script, feature engineering logic, the SageMaker Pipeline definition (Python SDK, not console wizard), the evaluation script and its condition logic, the Model Monitor baseline/constraint configuration, the EventBridge → pipeline-retrigger wiring, the CI/CD build spec, and the API layer in front of the real-time endpoint.

### 2.9 Training/inference code structure
```
src/
  preprocessing/
    process.py
  training/
    train.py            # XGBoost training entrypoint
  evaluation/
    evaluate.py
  inference/
    inference.py         # endpoint handler
  pipeline/
    pipeline_definition.py
monitoring/
  baseline_config.py
  monitor_schedule.py
```

### 2.10 Model Registry strategy
One model package group (`churn-predictor`), every pipeline run registers a candidate with attached evaluation metrics and a model card (intended use, training data description, known limitations — genuine responsible-AI documentation, not boilerplate). Approval is a manual gate initially; you can demonstrate an *optional* auto-approval condition (metric above threshold) as a stretch goal, but keep manual-by-default so you can narrate the human-oversight decision in interviews.

### 2.11 Deployment architecture
Real-time SageMaker endpoint with a small instance (e.g. `ml.t2.medium` or serverless inference for spiky/low traffic — serverless inference is a strong cost-saving choice for a portfolio project's real traffic pattern), autoscaling policy on invocation count, plus a separate batch transform job you run on-demand for bulk scoring, so you can discuss real-time-vs-batch tradeoffs from direct experience.

### 2.12 Batch vs. real-time
Real-time: use for the interactive demo API. Batch: use for a "score the whole customer base nightly" scenario — implement both so this isn't a hypothetical answer in an interview.

### 2.13 Monitoring strategy
SageMaker Model Monitor Data Quality (input feature drift vs. training baseline) and Model Quality (prediction distribution / accuracy drift, needs ground-truth labels to arrive later — simulate this with a labeled holdout you feed in on a delay) monitoring schedules, CloudWatch alarms on constraint violations.

### 2.14 Drift strategy
Statistical drift detection (Model Monitor's built-in constraints) on key features (`tenure`, `MonthlyCharges`, contract-type distribution). Document a clear threshold policy: what drift magnitude triggers a retrain vs. just an alert.

### 2.15 Retraining strategy
EventBridge rule on the Model Monitor CloudWatch alarm → triggers a new SageMaker Pipeline execution automatically. New model only replaces the deployed one if it passes the evaluation ConditionStep and is manually (or, for the stretch goal, automatically) approved in the registry.

### 2.16 CI/CD strategy
CodePipeline: git push to `main` → CodeBuild runs lint + unit tests → triggers SageMaker Pipeline execution via the SDK → on successful registration, a manual approval action gates production endpoint update.

### 2.17 IAM/security
Separate roles: pipeline execution role (S3 read/write on specific prefixes, SageMaker actions), endpoint role (inference-time permissions only), CI/CD role (least privilege to trigger pipelines, not modify IAM). Document each as a policy JSON in the repo.

### 2.18 IaC recommendations
AWS CDK (Python) for the endpoint, monitor schedules, EventBridge rules, and CI/CD pipeline — keeps everything as reviewable, diffable code, which is itself a portfolio artifact.

### 2.19 Estimated AWS development cost
Roughly **$10–30 total**: training jobs on small instances (`ml.m5.xlarge` or similar) for a dataset this size run minutes, not hours; HPO with a modest number of jobs (10–20) stays in the low single-digit dollars; Processing jobs are similarly cheap; S3/CloudWatch are negligible at this scale.

### 2.20 Estimated monthly cost if deployed
Roughly **$30–60/month** if using a small always-on real-time endpoint instance, or **under $10/month** if you switch to SageMaker Serverless Inference for the portfolio deployment (recommended — it scales to near-zero between demo sessions). Model Monitor adds a small Processing-job cost per scheduled run (hourly/daily schedule — use daily for a portfolio project). CodePipeline/CodeBuild are low-cost at low build frequency.

**Biggest bill risk**: an always-on real-time endpoint left running after you stop actively demoing it — delete it or switch to serverless inference between work sessions, and set a billing alarm.

### 2.21 Step-by-step roadmap
See Roadmap in section 3.

### 2.22 Suggested repo structure
```
driftwatch/
  infra/                 # CDK stack: endpoint, monitor, EventBridge, CI/CD
  src/
    preprocessing/
    training/
    evaluation/
    inference/
    pipeline/
  monitoring/
  data/
    raw/                  # or a download script, not committed data
  tests/
  docs/
    model_card.md
    architecture.md
  buildspec.yml
  README.md
```

### 2.23 Tests/validation
Unit tests for preprocessing transforms (known-input/known-output), a pipeline "dry run" integration test on a tiny data sample, schema validation tests on the Model Monitor baseline, a smoke test that invokes the deployed endpoint and checks response shape/latency.

### 2.24 Final portfolio demo
Show the SageMaker Pipeline DAG executing end-to-end in the console, show a model registered with its metrics/model card, show a synthetic drift event (feed skewed data) triggering Model Monitor and the automated retraining pipeline firing, show the CI/CD pipeline running from a git push.

### 2.25 Resume bullets
- "Built an end-to-end SageMaker Pipeline (processing, tuning, evaluation, conditional registration) for churn prediction, with automated retraining triggered by SageMaker Model Monitor drift alerts via EventBridge."
- "Implemented CI/CD (CodePipeline/CodeBuild) to trigger SageMaker training pipelines on git push, with Model Registry approval gating before production deployment."
- "Deployed and compared real-time and batch inference patterns on SageMaker, optimizing for cost with serverless inference."

### 2.26 MLA-C01 objectives reinforced
Domain 1 (data prep/feature engineering, directly), Domain 2 (model selection, HPO, evaluation, versioning, directly), Domain 3 (deployment, endpoints, autoscaling, CI/CD orchestration, directly), Domain 4 (monitoring, drift, IAM, cost — directly). This project maps to essentially the entire exam guide.

---

## 3. Combined Roadmap

| Phase | Project 1 (ClaimGuard) | Project 2 (DriftWatch) | Skills gained | Est. hours (P1 / P2) |
|---|---|---|---|---|
| 1. Architecture & setup | Design doc, AWS account/IAM baseline, repo scaffold | Same | Systems design, IaC scaffolding | 6 / 4 |
| 2. Data | Build synthetic claims dataset + policy docs | Download/clean Telco churn data | Data generation, schema design, cleaning | 8 / 6 |
| 3. Core implementation | Tool Lambdas, manual tool-loop then AgentCore migration | Processing + training scripts, XGBoost baseline | Agent tool design / classical ML training | 20 / 16 |
| 4. AWS integration | Knowledge Base, Guardrails, Step Functions | SageMaker Pipelines, Model Registry | Managed-service integration | 14 / 14 |
| 5. Evaluation | Golden set, faithfulness scoring, Bedrock Model Evaluation | HPO, metric comparison, threshold selection | Evaluation methodology | 10 / 8 |
| 6. Security | IAM per tool, adversarial prompt testing | IAM per role, model card | Least privilege, responsible AI | 6 / 4 |
| 7. Deployment | API Gateway, production config | Real-time + batch endpoints | Deployment patterns | 8 / 8 |
| 8. Monitoring | CloudWatch/X-Ray dashboard | Model Monitor + EventBridge retraining | Observability, drift handling | 10 / 10 |
| 9. Optimization | Prompt caching, model tiering, cost dashboard | Serverless inference, cost tuning | Cost/latency engineering | 6 / 6 |
| 10. Documentation & portfolio | README, demo recording, architecture diagram | Same | Communication | 6 / 6 |
| **Total (MVP)** | ~55 hrs | ~50 hrs | | |
| **Total (portfolio-strong, incl. CI/CD, adversarial testing, full IaC)** | ~85–100 hrs | ~75–90 hrs | | |

---

## 4. Final Comparison

| Category | Project 1 — ClaimGuard | Project 2 — DriftWatch |
|---|---|---|
| Primary AWS technology | Bedrock (AgentCore, Knowledge Bases, Guardrails) | SageMaker (Pipelines, Registry, Monitor) |
| ML/AI concepts | Agentic reasoning, RAG, grounding, guardrails | Classical ML, HPO, drift, MLOps |
| AWS services (Core count) | ~8 core | ~9 core |
| Engineering difficulty | High (novel orchestration, harder to evaluate) | High (more moving infra pieces, well-trodden patterns) |
| Estimated hours | 55–100 | 50–90 |
| Development cost | $15–40 | $10–30 |
| Monthly running cost | $50–150 (OpenSearch-dominated) | <$10–60 (endpoint-choice-dominated) |
| MLA-C01 coverage | Strong on Domains 2–4, generative-AI content | Strong on all 4 domains, classical-ML core |
| Resume value | High — differentiates you in a crowded "I built a RAG chatbot" market | High — the exact "production MLOps" story most AI/ML Engineer JDs ask for |
| Portfolio/demo value | Very high (visually compelling, tells a business story) | High (less flashy, but deeply credible to hiring managers) |
| Most important skills gained | Agent tool/guardrail engineering, grounding evaluation | Full ML lifecycle automation, drift-triggered retraining |

### Recommendation: build Project 2 (DriftWatch) first.

Reasoning: it maps to a larger fraction of the exam content directly and end-to-end, it's the skill set that's more likely to be missing from an Agentic-AI-focused resume today (you almost certainly already have more Bedrock/agent experience than classical SageMaker MLOps experience, given your day job), and it gives you a trained model and a working Model Registry that Project 1 can *reuse* (the fraud-scoring tool in ClaimGuard). Building DriftWatch first also means Project 1 stops being "another chatbot" and becomes "an agent that calls a model I trained myself" — which is a much stronger, more differentiated portfolio narrative when both projects are read together.
