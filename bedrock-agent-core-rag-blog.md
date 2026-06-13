# How AWS Bedrock AgentCore Handles RAG: A Deep Technical Dive

The most common question I see from engineers building AI applications on AWS isn't "which model should I use?" It's "why is my RAG pipeline returning irrelevant results?"

The answer is almost always the same: a mismatch between how documents were chunked, how retrieval was configured, and what the agent was actually asked to do. Getting RAG right on Bedrock requires understanding the full pipeline — from how you split documents to how AgentCore orchestrates retrieval across multi-agent systems.

AWS has built a lot of surface area here, evolving from a handful of vector store integrations at re:Invent 2023 to a full agentic platform (Bedrock AgentCore, GA October 2025). This post covers every RAG-relevant layer of that stack, with specific configuration recommendations at each step.

---

> **TL;DR**
> - **Knowledge Bases** handle the full ingestion pipeline (parse → chunk → embed → store) and support 8 vector stores, hybrid search, reranking, GraphRAG, and multimodal retrieval.
> - **Bedrock Agents** orchestrate RAG in a ReAct loop — the agent decides when to query a KB, what to ask, and how to combine retrieved context with API calls.
> - **AgentCore** adds a production runtime layer: long-running sessions, semantic tool selection via vector search, and managed memory that complements KB retrieval.
> - The three highest-impact configuration decisions: **chunking strategy**, **hybrid search on by default**, and **reranking**.

---

## The Three Ways to Do RAG on Bedrock

Before diving in, it helps to understand the three distinct modes AWS offers:

| Mode | API | Best For |
|---|---|---|
| **Standalone Knowledge Bases** | `RetrieveAndGenerate` / `Retrieve` | Pure Q&A chatbots; single-step RAG; no tool calls needed |
| **Bedrock Agents + KB Association** | `InvokeAgent` | Multi-step reasoning + RAG + API calls; AWS-managed ReAct orchestration |
| **AgentCore Runtime + Custom Agent + KB Tool** | `InvokeAgentRuntime` | BYO framework (LangGraph, CrewAI, Strands); long-running sessions; multi-agent A2A |

All three converge on the same underlying infrastructure: **Amazon Bedrock Knowledge Bases** for vector storage and retrieval, and **foundation models** on Bedrock for embedding and generation. The difference is in the orchestration layer above them.

---

## The RAG Engine: How Knowledge Bases Works Under the Hood

Knowledge Bases (GA November 2023) is the core RAG primitive across all three modes. It automates the full ingestion-to-retrieval pipeline — and understanding that pipeline is the foundation for every configuration decision downstream.

### The Two-Phase Pipeline

> *Note for editors: insert pipeline diagram image here — ingestion flow (Data Source → Parse → Chunk → Embed → Store) and retrieval flow (Query → Embed → Vector Search → Filter → Rerank → Augment → Response).*

**Ingestion** runs offline when you sync a data source:
1. **Fetch** — Bedrock connects to a configured data source (S3, Confluence, SharePoint, Salesforce, or a web crawler) and pulls raw documents
2. **Parse** — Documents are processed by the default parser, or optionally a Claude-based Foundation Model parser that extracts tables, charts, and images from complex PDFs
3. **Chunk** — Text is split using one of five strategies (covered in the next section)
4. **Embed** — Each chunk is converted to a dense vector using the configured embedding model
5. **Store** — Vectors, raw text, and metadata are written to the vector store index

**Retrieval** runs at query time:
1. **Embed query** — The user's question is vectorized using the same embedding model
2. **Vector search** — ANN similarity search against the index, with optional BM25 keyword search layered on top
3. **Filter** — Metadata filters applied pre- or post-retrieval (explicit or auto-generated from the query)
4. **Rerank** — Optional cross-encoder re-scores the candidate set for deeper relevance
5. **Augment** — Top-K chunks are injected into the LLM prompt
6. **Respond** — The FM generates a response with source citations

---

## Chunking: The Decision That Makes or Breaks Retrieval Quality

Chunking is one of the highest-leverage decisions in any RAG pipeline — yet it's often set to the default and forgotten. Bedrock offers five strategies, selectable at data source creation time. **The choice is permanent.** It cannot be changed after a data source is connected, so think carefully before committing.

### 1. Fixed-Size Chunking

The simplest and most predictable strategy. Splits documents into uniform token-length segments.

- **`maxTokens`**: 1–8,192 (AWS recommends 300)
- **`overlapPercentage`**: 1–99 (AWS recommends 20%)
- Best for: FAQs, product specs, uniform factual documents

### 2. Default Chunking

A preset of fixed-size chunking with AWS-chosen defaults: ~300 tokens, sentence-boundary-aware, 20% overlap. Suitable as a starting point for most document types.

### 3. Semantic Chunking

Groups content by meaning rather than token count. Each sentence is embedded and compared to its neighbor; chunk boundaries form where cosine dissimilarity crosses a threshold.

Key parameters:
- **`breakpointPercentileThreshold`** (50–99, recommend 95): Controls sensitivity to topic shifts. Higher = fewer, larger chunks.
- **`bufferSize`** (0 or 1): At `bufferSize=1`, sentence N's embedding incorporates sentences N-1, N, and N+1 for richer boundary detection.
- **`maxTokens`**: Safety cap on chunk size.
- Best for: Long-form articles, research papers, content with distinct topical sections.

### 4. Hierarchical Chunking

Creates a two-layer parent-child structure. Small child chunks are indexed for precise retrieval; the larger parent chunk is returned as context to the LLM.

- **`maxParentTokenSize`**: AWS recommends 1,500 tokens
- **`maxChildTokenSize`**: AWS recommends 300 tokens
- **`overlapTokens`**: Absolute token count of overlap

This decouples retrieval precision (child vectors) from generation context richness (parent text). Best for complex nested documents: legal contracts, technical manuals, research papers.

### 5. Custom (Lambda) Chunking

For domain-specific formats (source code, call transcripts, regulatory clauses) where generic chunkers break semantic units.

Pipeline:
1. Bedrock parses raw documents → writes to intermediate S3 bucket
2. Your Lambda function reads parsed content and applies custom chunking
3. Lambda writes chunked output back to S3
4. Bedrock reads chunk references → embeds → indexes

This integrates cleanly with LangChain, LlamaIndex, or proprietary splitting logic.

**My take:** Hierarchical chunking is chronically underused. Most teams default to fixed-size at 300 tokens, which works fine for FAQs but fails badly on long technical documents — you retrieve a precise child chunk but the LLM lacks the surrounding context to answer correctly. If your documents have nested structure (contracts, research papers, technical specs), start with hierarchical at the recommended 1,500/300 token split and benchmark from there.

---

## Choosing an Embedding Model (and Why You Can't Change It Later)

The embedding model must be chosen at knowledge base creation and **cannot be changed without re-indexing everything from scratch**. The same model is used for both ingestion and query-time retrieval, so mismatches produce garbage results.

| Model | Dimensions | Max Tokens | Notes |
|---|---|---|---|
| Amazon Titan Text Embeddings V1 | 1,536 (fixed) | 8,192 | Legacy; 25+ languages |
| **Amazon Titan Text Embeddings V2** | 256, 512, or 1,024 | 8,192 | Recommended; configurable dims; binary vectors; 100+ languages |
| Cohere Embed English V3 | 1,024 | 512 | English-optimized precision |
| Cohere Embed Multilingual V3 | 1,024 | 512 | 100+ languages |
| Cohere Embed V4 | 256–1,536 | ~128,000 | Mixed text + images; very long context |
| Amazon Nova Multimodal Embeddings | 256–3,072 | 8,192 | Text, images, video, audio; 200+ languages; GA Nov 2025 |

**Titan V2 dimension accuracy trade-off** (per AWS benchmarks):
- 1,024 → 512 dims: ~99% accuracy retained
- 1,024 → 256 dims: ~97% accuracy retained

Binary vector mode (Titan V2 only) reduces storage by ~32× with minimal quality loss, supported on OpenSearch Serverless and OpenSearch Managed Cluster.

**My take:** Default to Titan V2 at 512 dimensions for most use cases — you retain ~99% of retrieval accuracy vs. the full 1,024 dimensions, cut your index size roughly in half, and get 100+ language support out of the box. Only switch to Cohere Embed English V3 if you have a monolingual English corpus and need to squeeze out every point of precision.

---

## Eight Vector Stores, One Right Choice for Your Use Case

Bedrock Knowledge Bases supports eight vector stores, each with distinct trade-offs:

| Store | Type | Key Strengths | Binary Vectors | Hybrid Search |
|---|---|---|---|---|
| Amazon OpenSearch Serverless | AWS-managed | Default; no infra; FAISS/HNSW/IVF | Yes | Yes |
| Amazon OpenSearch Managed Cluster | AWS-managed | Full OpenSearch control; GA March 2025 | Yes | Yes |
| Amazon Aurora PostgreSQL (pgvector) | AWS-managed | ~90% cheaper than OpenSearch; IVFFlat/HNSW | No | Yes (April 2025) |
| Amazon Neptune Analytics | AWS-managed | GraphRAG — vector + knowledge graph; GA March 2025 | No | No |
| Amazon S3 Vectors | AWS-managed | ~90% cost reduction; up to 2B vectors/index; GA Jan 2026 | No | No |
| Pinecone | Third-party | Existing Pinecone users | No | No |
| MongoDB Atlas | Third-party | Existing MongoDB users; GA May 2024 | No | Yes (April 2025) |
| Redis Enterprise Cloud | Third-party | Low-latency in-memory retrieval | No | No |

**Choosing a vector store:**
- Default / fastest start: **OpenSearch Serverless**
- Cost-sensitive at massive scale: **S3 Vectors** (up to 2 billion vectors, ~$0.04/million writes)
- Multi-hop reasoning across documents: **Neptune Analytics** (GraphRAG)
- Teams already on RDS: **Aurora PostgreSQL**

**My take:** If you're starting a new project with no existing vector infrastructure, don't overthink this — go OpenSearch Serverless and revisit when you have real cost data. The one exception: if you're building a system where documents have rich entity relationships (policy docs, org charts, technical dependency trees), start with Neptune Analytics. Retrofitting GraphRAG later means re-ingesting everything.

---

## Beyond Keyword Search: Semantic, Hybrid, Filtering, and Reranking

### Semantic Search (Default)

Converts the query to a vector using the configured embedding model and runs approximate nearest-neighbor (ANN) search using cosine similarity. The `overrideSearchType` defaults to `SEMANTIC` when omitted.

### Hybrid Search

Executes two parallel searches and fuses the results:

1. **Vector search** — cosine similarity on dense embeddings
2. **BM25 keyword search** — sparse full-text matching on raw chunk text

Results are merged via **score normalization + weighted combination** (min-max or L2 normalization, then weighted average — e.g., 0.3 × BM25 + 0.7 × vector). This is implemented via an OpenSearch normalization-and-combination search pipeline.

**Availability:** OpenSearch Serverless/Managed (since March 2024), Aurora PostgreSQL and MongoDB Atlas (since April 2025).

Activate with: `"overrideSearchType": "HYBRID"` in `vectorSearchConfiguration`.

**When hybrid outperforms semantic-only:** Long-tail queries with rare proper nouns, product codes, or exact terminology that embeddings dilute; brand names; numeric identifiers.

**I'd enable hybrid search by default on every production knowledge base.** The cost overhead is minimal and the recall improvement for edge-case queries is significant. The only reason not to: your vector store doesn't support it (Pinecone, Redis, S3 Vectors are vector-only as of mid-2026).

### Metadata Filtering

Attach custom attributes to documents at ingestion via `.metadata.json` sidecar files (max 10 KB, same S3 folder as source), then filter at query time.

Supported operators: `equals`, `notEquals`, `greaterThan`, `greaterThanOrEquals`, `lessThan`, `lessThanOrEquals`, `stringContains`, `startsWith` (OpenSearch Serverless only), `listContains`, `in`, `notIn`, `andAll`, `orAll`.

**Auto-generated (Implicit) Filters** (GA December 2024): Set `implicitFilterConfiguration` with a model ARN and attribute descriptions, and Bedrock uses an LLM to extract filter values from natural language queries automatically. Query: *"How do I file a claim in Washington?"* → filter: `{state: "Washington"}` — no application-layer parsing needed.

### Query Reformulation

For multi-hop or compound questions, Bedrock decomposes the query into multiple sub-queries, runs each through retrieval independently, pools the results, and re-ranks them before passing context to the FM. Particularly effective for queries like *"How does our return policy differ between domestic and international orders?"*

### Reranking (GA December 2024)

A cross-encoder model re-scores retrieved candidates against the query after initial vector retrieval, producing a more semantically coherent top-K set. The pattern: retrieve a wider candidate pool (e.g., `numberOfResults: 20`), let the reranker score each (query, document) pair jointly with relevance ∈ [0, 1], then return only the top-N reranked results (`numberOfRerankedResults: 5`). The reranker's scores override the original similarity scores entirely.

**Model comparison:**

| Model | Context Window | Languages | Handles Semi-Structured Data |
|---|---|---|---|
| Amazon Rerank 1.0 | Standard | English-focused | No |
| Cohere Rerank 3.5 | 32,000 tokens | 100+ | Yes (JSON, emails, tables) |

Configure via `rerankingConfiguration` in `vectorSearchConfiguration`, or use the standalone `Rerank` API for non-KB retrieval sources.

Reranking is one of the cheapest quality improvements in RAG — typically sub-millisecond latency overhead for dramatically better precision on the final result set. If you're not reranking in production, you're leaving accuracy on the table.

---

## GraphRAG: When Vector Search Isn't Enough

Standard RAG treats documents as independent chunks — it cannot reason about *relationships* between entities across documents. GraphRAG, GA'd March 2025, addresses this.

When you create a knowledge base backed by **Amazon Neptune Analytics**, Bedrock automatically:
1. Extracts entities and relationships from ingested documents
2. Builds a knowledge graph alongside vector embeddings
3. At query time, combines **vector similarity search + graph traversal**

This makes multi-hop questions tractable: *"Which compliance policies apply to the products mentioned in the Q3 risk report?"* A vector-only search would retrieve chunks about each topic independently; GraphRAG can traverse entity connections to surface the relationship.

**Benchmark:** AWS partner Lettria reported up to 35% improvement in answer precision vs. vector-only retrieval (single-partner benchmark; results will vary by data structure).

No graph modeling expertise is required — the entity extraction and graph construction are fully automated.

**One honest caveat:** GraphRAG is genuinely impressive for the right data, but check whether your documents actually have rich entity relationships before switching. For a knowledge base of 500 product FAQs with no cross-references, vector search is already optimal. GraphRAG earns its keep in datasets like regulatory libraries, org structures, or technical dependency graphs — places where multi-hop questions are common and important.

---

## Multimodal RAG: Searching Images, Audio, and Video

GA'd November 2025, multimodal RAG enables knowledge bases to ingest, embed, and retrieve across text, images, audio, and video.

The key enabler is **Amazon Nova Multimodal Embeddings** (GA October 2025):
- Single unified model for text, documents, images, video, and audio
- Output dimensions: 3,072 / 1,024 / 384 / 256 (via Matryoshka Representation Learning)
- Context: up to 8,192 tokens
- 200+ language support
- Leads benchmarks on ActivityNet (video) and TextCaps vs. TwelveLabs, Google Vertex AI, Cohere, Titan

Before multimodal embeddings, complex PDFs with charts, tables, and figures required the **Foundation Model parser** (Claude Sonnet or Haiku) to describe visual elements as text before embedding. That approach still works and is compatible with text-only embedding models.

---

## RAG Over Tables: When Your Data Lives in a Warehouse

All the retrieval methods above target *unstructured* text. But most enterprises keep critical data in structured form — data warehouses, lakehouses, operational databases. Bedrock Knowledge Bases extended RAG to cover structured data in December 2024 via **natural language to SQL generation**.

### How It Works

The flow is: **natural language query → LLM generates SQL (using the table's schema metadata) → SQL executes against Redshift or SageMaker Lakehouse → result set is passed back to the FM → user-friendly response**. No ETL required, no vector index — the data stays in the source system.

**Supported data sources:** Amazon Redshift and Amazon SageMaker Lakehouse.

The data stays in the source system — no ETL, no copying into a vector index. Bedrock reads the schema metadata to understand table structure, generates SQL, executes it, and passes the result set to the foundation model to synthesize a human-readable answer.

### Unstructured vs. Structured RAG — When to Use Each

| Data Type | RAG Approach | When to Choose |
|---|---|---|
| Documents, PDFs, web pages, wikis | Knowledge Bases (vector search) | Semantic/conceptual questions; "what does policy X say about Y?" |
| Tables, data warehouses, lakehouses | Structured data retrieval (SQL generation) | Analytical questions; "how many orders in Q3?", "top 5 customers by revenue" |
| Mixed | Both, in the same agent | Agents can call both a KB lookup and a SQL-generation KB in a single ReAct loop |

A single Bedrock Agent can be associated with both a vector-backed KB and a structured data KB simultaneously — the agent's FM decides which to query based on the nature of the question.

---

## How Bedrock Agents Orchestrate RAG in a ReAct Loop


Standalone Knowledge Bases handle single-step retrieval-then-generate. When you need multi-step reasoning, API calls, database writes, or complex workflows alongside RAG, **Bedrock Agents** is the answer.

### The ReAct Orchestration Loop

Bedrock Agents implement a managed **ReAct (Reason + Act)** loop across four prompt pipeline stages:

1. **Pre-processing:** Validates and contextualizes user input; can apply guardrails before any tool fires.
2. **Orchestration (ReAct loop):** The core reasoning engine. For each iteration, the FM produces:
   - **Thought (Rationale):** Internal chain-of-thought — *"I need the customer's contract terms before I can answer this billing question."*
   - **Action:** Call an action group function OR query a knowledge base, with a natural language query (not keyword search).
   - **Observation:** The chunk text returned from KB retrieval, or the Lambda function response.
   - Loop continues until the FM produces a **Final Answer**.
3. **Knowledge Base Response Generation:** A separate prompt template that controls how retrieved KB chunks are formatted as an observation before re-entering the orchestration loop.
4. **Post-processing:** Formats the final answer for the user.

### Associating Knowledge Bases with Agents

```python
# Associate a KB with an agent via the API
bedrock_agent.associate_agent_knowledge_base(
    agentId="AGENT_ID",
    agentVersion="DRAFT",
    knowledgeBaseId="KB_ID",
    description="Contains product documentation and pricing. Query this when users ask about features, pricing, or compatibility.",
    knowledgeBaseState="ENABLED"
)
```

The `description` field is critical — it's used by the agent's FM for tool selection, determining *when* to query this KB vs. another KB vs. calling an action group.

A single agent can be associated with multiple knowledge bases (e.g., one for product docs, one for policy docs, one for HR handbook), each with independent descriptions.

### Tracing for Explainability

Enable tracing at invocation time to capture the full ReAct trace:

```json
{
  "trace": {
    "orchestrationTrace": {
      "rationale": { "text": "I need to look up the customer's contract to verify their SLA terms before responding." },
      "invocationInput": { "knowledgeBaseLookupInput": { "text": "customer SLA terms enterprise tier", "knowledgeBaseId": "KB_XYZ" } },
      "observation": { "knowledgeBaseLookupOutput": { "retrievedReferences": [...] } }
    }
  }
}
```

This trace is the primary mechanism for auditing why the agent queried a specific KB and what it retrieved.

---

## Inline Agents: Reconfiguring RAG at Runtime

Traditional Bedrock Agents require pre-configuration: create the agent, add action groups and knowledge bases, prepare a version, then invoke it. Changes require creating a new version.

**Inline Agents** (`InvokeInlineAgent` API) flip this model — the entire agent configuration (model, instructions, action groups, knowledge bases, guardrails) is specified at runtime per request.

```python
response = bedrock_agent_runtime.invoke_inline_agent(
    sessionId="session-123",
    inputText="What is our refund policy for international customers?",
    foundationModel="anthropic.claude-3-5-sonnet-20241022-v2:0",
    instruction="You are a customer service agent. Use the knowledge base to answer accurately.",
    knowledgeBases=[{
        "knowledgeBaseId": "KB_ID",
        "description": "Customer service policies and procedures"
    }],
    actionGroups=[...]
)
```

**Use cases where Inline Agents excel:**
- **Per-user role customization:** Dynamically compose agent capabilities based on the authenticated user's role
- **Rapid experimentation:** Test instruction variants without creating agent versions
- **Dynamic routing:** A supervisor layer routes requests to dynamically-configured sub-agents based on query type
- **Multi-tenant SaaS:** Each tenant gets a different knowledge base or instruction set without maintaining N separate agents

**Limitation:** Instructions are not honored if the agent has only one KB, default prompts, no action group, and user input is disabled — a known edge case documented by AWS.

---

## Multi-Agent RAG: Splitting Domain Knowledge Across Specialists

For large-scale agentic systems, Bedrock supports networks of specialized agents orchestrated by a supervisor.

### Architecture

```
User Request
    │
    ▼
Supervisor Agent
  ├── Receives full request
  ├── Plans subtasks
  └── Routes to Sub-Agents:
        ├── Compliance Agent ──> Regulatory KB (GraphRAG on Neptune)
        ├── HR Agent ──────────> Employee Handbook KB (OpenSearch)
        └── Finance Agent ─────> Pricing KB (Aurora PostgreSQL) + Billing API
                                                        │
                                                        ▼
                                              Consolidated Response
```

Each sub-agent runs its own independent ReAct loop with its own associated knowledge bases. Domain-specific RAG stays within the relevant sub-agent.

### Two Orchestration Modes

1. **Supervisor Mode:** Supervisor plans, parallelizes or sequences sub-agents, synthesizes results. Best for complex multi-domain queries.
2. **Supervisor with Routing Mode:** For simpler single-domain queries, routes directly to the best sub-agent without full planning. Lower latency fallback.

### Payload Referencing

A critical performance optimization: instead of embedding large retrieved documents in every inter-agent message (multiplying token costs), the supervisor passes **reference pointers** to retrieved data. Sub-agents fetch data directly, reducing inter-agent token overhead significantly.

---

## Agent Memory: The Other Half of RAG

### Session Memory (In-Session)

All conversation turns within the same `sessionId` share context automatically. The agent does not need to be re-briefed. You can inject structured context per-turn via `sessionState`:
- `sessionAttributes`: Key-value pairs visible to Lambda action group functions
- `promptSessionAttributes`: Key-value pairs injected into the prompt template (accessible by the FM)

### Long-Term Memory (AgentCore Memory)

For context that persists across sessions, **AgentCore Memory** provides managed semantic memory storage with zero infrastructure management.

**How it works:**
1. After a session ends, an async background process reads conversation events
2. An LLM-based extraction pipeline consolidates insights into structured memory records
3. At the start of a new session, relevant memories are retrieved via semantic search and injected into the agent's context

**Four built-in memory strategies:**

| Strategy | What It Stores | Use Case |
|---|---|---|
| **Semantic Memory** | Factual knowledge as vector embeddings | *"User is a data engineer at a healthcare company using Python"* |
| **Summary Memory** | Rolling summaries of past sessions | Long-running projects spanning many conversations |
| **User Preference Memory** | Behavioral patterns and preferences | *"User prefers code examples with type annotations"* |
| **Custom Memory** | Developer-defined extraction logic | Domain-specific entities |

**Retention:** 1–365 days, configurable.

**Memory vs. RAG:** They're complementary. Knowledge Bases hold static organizational knowledge (product docs, policies). Long-term memory holds dynamic per-user learned context. The same agent can leverage both simultaneously — querying the KB for factual grounding while using memory for user-specific personalization.

---

## AgentCore's RAG Capabilities: Runtime, Memory, Gateway, Observability

AgentCore is a new product layer distinct from Bedrock Agents. Rather than a fixed ReAct orchestrator, it's a **production runtime infrastructure** for agents built with *any* framework — AWS's own Strands Agents SDK, LangGraph, CrewAI, LlamaIndex, or custom implementations. Of its seven components, four have direct relevance to how agents discover, retrieve, and reason over knowledge.

### RAG-Relevant AgentCore Components

#### 1. Runtime — Execution Environment for Long-Running RAG Agents

Standard Bedrock Agent invocations are synchronous and short-lived. AgentCore Runtime lifts that constraint:
- **8-hour session windows** — enables agents to iteratively retrieve, reason, and act over extended workflows without re-establishing context
- **Per-user session isolation** via microVMs — each user's retrieval state and in-flight results are fully isolated
- **Stateful MCP server support** — agents can maintain open connections to retrieval services across multiple turns
- **A2A protocol (port 9000)** — sub-agents in a multi-agent system can call each other's retrieval tools directly

#### 2. Memory — Semantic RAG Over User Context

AgentCore Memory is, architecturally, a **RAG system for personalization**. It uses the same vector embedding and similarity search pattern as Knowledge Bases, but over *learned user context* rather than static documents.

**How retrieval works:**
1. Async consolidation pipeline runs after each session, extracting key facts via an LLM
2. Extracted facts are embedded and stored in a managed vector store
3. At the start of each new session, relevant memories are retrieved via **semantic search** and injected into the agent's context — exactly like KB retrieval

**Four memory strategies and their retrieval implications:**

| Strategy | What Gets Indexed | Retrieval Trigger |
|---|---|---|
| **Semantic Memory** | Factual knowledge as embeddings (*"user works in healthcare compliance"*) | Any query — most semantically similar facts retrieved |
| **Summary Memory** | Rolling session summaries | Session start — recent summaries always included |
| **User Preference Memory** | Behavioral patterns (*"prefers concise responses, Python examples"*) | Always retrieved at session start |
| **Custom Memory** | Developer-defined entities | Domain-specific retrieval logic |

**Memory vs. Knowledge Bases — when to use each:**

| Dimension | Knowledge Bases | AgentCore Memory |
|---|---|---|
| **Content type** | Static organizational knowledge (docs, policies, manuals) | Dynamic per-user learned context |
| **Update frequency** | Batch ingestion jobs; scheduled or event-triggered | Continuous — updated after every session automatically |
| **Scope** | Shared across all users | Per-user or per-agent |
| **Retrieval** | Hybrid/semantic search over document chunks | Semantic search over extracted memory records |
| **Best for** | *"What does our refund policy say?"* | *"What did this user ask about last week?"* |

The same agent uses both simultaneously: KB for factual grounding, memory for user-specific personalization. As of May 2026, memory records support metadata filtering (up to 10 indexed keys per record), enabling combined semantic + attribute-filtered memory retrieval — the same pattern as KB metadata filtering.

In practice, I'd reach for Memory when the agent needs to adapt its behavior per user over time — remembering that a specific user is a Python developer, prefers terse answers, or has asked about a particular topic repeatedly. Knowledge Bases handles everything that should be consistent across all users.

#### 3. Gateway — Semantic Tool Selection (RAG for Tool Discovery)

AgentCore Gateway converts REST APIs, Lambda functions, and existing MCP servers into agent-compatible tools. Its most RAG-relevant feature is **semantic tool selection**: when an agent has access to a large pool of tools, Gateway uses **vector similarity search over tool descriptions** to surface the most relevant tools for a given query — rather than passing the full tool list to the LLM on every call.

This is RAG applied to the tool layer:
- Tools are embedded at registration time (their descriptions → vectors)
- At query time, the agent's query is embedded and matched against tool vectors
- Only the top-K most relevant tools are surfaced to the LLM for selection

This solves a real scaling problem: LLMs degrade at tool selection when presented with hundreds of tools. Gateway's semantic retrieval keeps the effective tool set small and relevant per query.

#### 4. Observability — Monitoring RAG Quality in Production

AgentCore Observability (via CloudWatch) captures end-to-end execution traces, making it possible to monitor RAG-specific metrics across the full agent lifecycle:
- Which KB queries fired per session, and what was retrieved
- Latency breakdown: embedding time, vector search time, reranking time, generation time
- Memory retrieval events — what was pulled from long-term memory per session
- Citation tracking — which source documents contributed to each response

### Using RAG with AgentCore

Knowledge Bases are invoked from inside your agent code as a tool:

```python
# Strands Agents SDK example
from strands import Agent
from strands.tools import retrieve_from_knowledge_base

agent = Agent(
    model="us.amazon.nova-pro-v1:0",
    tools=[retrieve_from_knowledge_base(knowledge_base_id="KB_ID")],
    system_prompt="You are a technical support assistant. Use the knowledge base to answer accurately."
)

response = agent("How do I configure VPC peering with custom DNS?")
```

AgentCore Runtime wraps this agent with session isolation, long-running execution, and observability — without changing your agent code.

### When to Choose What

| Scenario | Best Choice |
|---|---|
| Pure RAG chatbot, no tool calls | Standalone KB + `RetrieveAndGenerate` |
| Multi-step reasoning + API calls + RAG | Bedrock Agents + KB association |
| BYO framework (LangGraph/CrewAI) + production scale | AgentCore Runtime + KB tool |
| Multi-agent with domain-specific RAG | Multi-agent Bedrock Agents OR AgentCore with A2A |
| Long-running autonomous tasks (hours) | AgentCore Runtime (8-hour session windows) |
| Large tool pool (100+ tools) with RAG agents | AgentCore Gateway semantic tool selection |
| Per-user personalization alongside KB retrieval | AgentCore Memory + Knowledge Bases together |

If you're just getting started, my honest recommendation is to ignore AgentCore until you've proven out the use case with standalone Knowledge Bases or Bedrock Agents first. AgentCore earns its complexity at production scale — when you need BYO frameworks, multi-agent A2A coordination, or sessions that genuinely run for hours. It's the right tool for that problem, but overkill for a first RAG chatbot.

---

## Guardrails: Catching What RAG Gets Wrong

Production RAG systems need quality and safety controls. Bedrock Guardrails applies at multiple pipeline stages:

### Contextual Grounding (Hallucination Detection)

Available since July 2024, this is Bedrock's purpose-built mechanism for detecting RAG-specific hallucinations:
- **Groundedness check:** Is the response grounded in the retrieved context? Detects claims the LLM added that aren't in the retrieved chunks.
- **Relevance check:** Is the response relevant to the user's query?
- AWS claims detection of 75%+ of AI hallucinations in RAG/summarization responses.

### RAG Evaluation with LLM-as-a-Judge (GA March 2025)

Automated quality scoring for your RAG pipeline across six dimensions:
- **Context Relevance:** Are retrieved chunks relevant to the query?
- **Context Coverage:** Do the chunks cover the answer fully?
- **Correctness:** Is the generated answer factually accurate?
- **Completeness:** Does the answer address all parts of the query?
- **Faithfulness:** Does the answer stay within the retrieved context?
- **Harmfulness / Stereotyping:** Safety metrics

Use this for offline evaluation during development and for A/B testing chunking strategies or retrieval configurations.

### Automated Reasoning Checks (re:Invent 2024)

Encodes domain rules as logical policies and provides *verifiable mathematical proof* that a response complies — not just an LLM judgment. Particularly valuable for compliance, legal, and financial domains where "the model said so" is insufficient.

---

## Timeline: How Bedrock RAG Has Evolved (2023–2026)

| Date | Milestone |
|---|---|
| November 2023 | Knowledge Bases GA (S3, OpenSearch Serverless, Aurora, Pinecone, Redis) |
| March 2024 | Hybrid search GA (OpenSearch Serverless); metadata filtering GA |
| May 2024 | MongoDB Atlas vector store support |
| June 2024 | Amazon Titan Text Embeddings V2 for Knowledge Bases |
| July 2024 | Contextual Grounding in Guardrails; AgentCore Memory preview |
| July 2024 | Advanced chunking (semantic, hierarchical), FM parser, query reformulation GA |
| November 2024 | Inline Agents (`InvokeInlineAgent`) announced |
| December 2024 | re:Invent 2024 — streaming `RetrieveAndGenerateStream`; auto-generated query filters; Rerank API (Amazon Rerank 1.0, Cohere Rerank 3.5); GraphRAG preview; structured data retrieval (SQL generation for Redshift/SageMaker Lakehouse); multi-agent collaboration preview; custom connectors |
| March 2025 | GraphRAG GA with Neptune Analytics; multi-agent collaboration GA; OpenSearch Managed Cluster support; RAG Evaluation GA |
| April 2025 | Hybrid search extended to Aurora PostgreSQL and MongoDB Atlas |
| July 2025 | AgentCore preview launch |
| October 2025 | AgentCore GA (9 AWS regions) |
| October 2025 | Amazon Nova Multimodal Embeddings announced |
| November 2025 | Multimodal retrieval for Knowledge Bases GA |
| December 2025 | re:Invent 2025 — AgentCore Policy (preview), AgentCore Evaluations (preview), Frontier Agents preview |
| January 2026 | Amazon S3 Vectors GA; native Bedrock Knowledge Bases integration |
| May 2026 | AgentCore Long-Term Memory metadata filtering (up to 10 indexed keys per memory record) |

---

## Putting It All Together: A Production RAG Architecture

Here is a representative architecture for a large enterprise RAG system using the full Bedrock stack as of mid-2026:

> *Note for editors: insert architecture diagram image here — Data Sources (S3, Confluence, SharePoint, Web) → Knowledge Bases (Parse/Chunk/Embed with OpenSearch Serverless for text KBs and Neptune Analytics for GraphRAG) → AgentCore Runtime (Supervisor Agent orchestrating Tech Agent + Policy Agent, each with their own KBs, plus AgentCore Long-Term Memory) → Bedrock Guardrails (Contextual Grounding, Automated Reasoning, PII Filter) → User Response with citations.*

The key insight in this architecture: **domain-specific RAG stays within each sub-agent**. The Compliance Agent owns the regulatory KB (on Neptune for GraphRAG), the HR Agent owns the employee handbook KB (on OpenSearch), and the Finance Agent combines a pricing KB with a Redshift structured data KB. The supervisor never directly queries a knowledge base — it coordinates, not retrieves.

---

## Conclusion

AWS Bedrock's RAG stack has matured from a simple document-to-vector pipeline into a comprehensive agentic platform. The key architectural decisions that will shape your implementation:

1. **Chunking strategy** determines retrieval precision more than almost any other parameter — invest time benchmarking semantic vs. hierarchical chunking for your document types.

2. **Hybrid search** should be your default for production systems where users mix semantic questions with exact keyword lookups.

3. **Reranking** provides a meaningful accuracy boost for under-$0.001-per-query cost — enable it on production workloads.

4. **GraphRAG** unlocks multi-hop reasoning that vector-only retrieval cannot achieve; evaluate it if your documents have rich entity relationships.

5. **Structured data retrieval** lets agents answer analytical questions from data warehouses without ETL — if your use case mixes document Q&A with data queries, a single agent can handle both KB types in the same ReAct loop.

6. **AgentCore Gateway's semantic tool selection** is RAG applied to the tool layer — essential when your agent pool grows beyond ~20 tools, as LLMs degrade at tool selection with large flat lists.

7. **Memory vs. Knowledge Bases** serve different retrieval needs: KBs for shared static knowledge, Memory for dynamic per-user context. Use both together for personalized, grounded responses.

8. **Guardrails with contextual grounding** should be non-negotiable in production — hallucination in RAG systems is subtle and difficult to detect without automated verification.

---

## References

- [Amazon Bedrock Knowledge Bases — Product Page](https://aws.amazon.com/bedrock/knowledge-bases/)
- [How Amazon Bedrock Knowledge Bases work — AWS Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-how-it-works.html)
- [How content chunking works — AWS Docs](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-chunking.html)
- [Advanced parsing, chunking, and query reformulation — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-knowledge-bases-now-supports-advanced-parsing-chunking-and-query-reformulation-giving-greater-control-of-accuracy-in-rag-based-applications/)
- [Hybrid Search GA (March 2024) — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/03/knowledge-bases-amazon-bedrock-hybrid-search/)
- [Hybrid Search extended to Aurora/MongoDB (April 2025) — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2025/04/amazon-bedrock-knowledge-bases-hybrid-search-aurora-postgresql-mongo-db-atlas-vector-stores/)
- [Rerank API GA (December 2024) — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/12/amazon-bedrock-rerank-api-accuracy-rag-applications)
- [Cohere Rerank 3.5 in Bedrock — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/cohere-rerank-3-5-is-now-available-in-amazon-bedrock-through-rerank-api/)
- [GraphRAG GA with Neptune Analytics — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/announcing-general-availability-of-amazon-bedrock-knowledge-bases-graphrag-with-amazon-neptune-analytics/)
- [Structured Data Retrieval (December 2024) — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/12/amazon-bedrock-knowledge-bases-structured-data-retrieval)
- [Streaming RetrieveAndGenerateStream — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/12/amazon-bedrock-knowledge-bases-streaming-retrieveandgeneratestream-api/)
- [Auto-Generated Query Filters — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/12/amazon-bedrock-knowledge-bases-auto-generated-query-filters-improved-retrieval)
- [Nova Multimodal Embeddings — AWS News Blog](https://aws.amazon.com/blogs/aws/amazon-nova-multimodal-embeddings-now-available-in-amazon-bedrock/)
- [Multimodal Retrieval GA — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2025/11/multimodal-retrieval-bedrock-knowledge-bases/)
- [Introducing Amazon Bedrock AgentCore — AWS News Blog](https://aws.amazon.com/blogs/aws/introducing-amazon-bedrock-agentcore-securely-deploy-and-operate-ai-agents-at-any-scale/)
- [AgentCore GA — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2025/10/amazon-bedrock-agentcore-available/)
- [InlineAgents (November 2024) — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/11/inlineagents-agents-amazon-bedrock/)
- [Multi-Agent Collaboration GA — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-announces-general-availability-of-multi-agent-collaboration/)
- [Amazon S3 Vectors GA — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-s3-vectors-generally-available/)
- [Titan Text Embeddings V2 for Knowledge Bases — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/06/amazon-titan-text-embeddings-v2-bedrock-knowledge-bases)
- [Guardrails: Contextual Grounding — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2024/07/guardrails-bedrock-hallucinations-safeguard-apps-fm/)
- [RAG Evaluation GA — AWS What's New](https://aws.amazon.com/about-aws/whats-new/2025/03/amazon-bedrock-rag-evaluation-generally-available/)
- [Dive Deep into Vector Data Stores — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/dive-deep-into-vector-data-stores-using-amazon-bedrock-knowledge-bases/)
- [Binary Embeddings with Titan V2 — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/build-cost-effective-rag-applications-with-binary-embeddings-in-amazon-titan-text-embeddings-v2-amazon-opensearch-serverless-and-amazon-bedrock-knowledge-bases/)
- [Building Intelligent Event Agents with AgentCore and Knowledge Bases — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/building-intelligent-event-agents-using-amazon-bedrock-agentcore-and-amazon-bedrock-knowledge-bases/)
