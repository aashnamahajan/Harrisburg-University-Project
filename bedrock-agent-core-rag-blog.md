# How AWS Bedrock AgentCore Handles RAG: A Deep Technical Dive

Retrieval-Augmented Generation (RAG) has become the dominant pattern for grounding large language models in accurate, up-to-date organizational knowledge. Amazon Web Services has invested deeply in making RAG not just possible, but production-ready at enterprise scale — evolving from a handful of vector store integrations at re:Invent 2023 to a full agentic platform (Bedrock AgentCore, GA October 2025) that handles everything from document ingestion through multi-agent orchestration. This post walks through every layer of the stack.

---

## The Big Picture: Three Ways to Do RAG on Bedrock

Before diving into architecture, it helps to understand the three distinct modes AWS offers:

| Mode | API | Best For |
|---|---|---|
| **Standalone Knowledge Bases** | `RetrieveAndGenerate` / `Retrieve` | Pure Q&A chatbots; single-step RAG; no tool calls needed |
| **Bedrock Agents + KB Association** | `InvokeAgent` | Multi-step reasoning + RAG + API calls; AWS-managed ReAct orchestration |
| **AgentCore Runtime + Custom Agent + KB Tool** | `InvokeAgentRuntime` | BYO framework (LangGraph, CrewAI, Strands); long-running sessions; multi-agent A2A |

All three converge on the same underlying infrastructure: **Amazon Bedrock Knowledge Bases** for vector storage and retrieval, and **foundation models** on Bedrock for embedding and generation. The difference is in the orchestration layer above them.

---

## Part 1: Bedrock Knowledge Bases — The RAG Engine

Knowledge Bases reached general availability in November 2023 and is the core RAG primitive across all three modes. It automates the full ingestion-to-retrieval pipeline.

### The Two-Phase Pipeline

#### Ingestion (Offline)

```
Data Source (S3, Confluence, SharePoint, Salesforce, Web Crawler)
    │
    ▼
  Parse  →  Foundation Model parser (Claude) or BDA parser for PDFs
    │        with tables, charts, images; Default parser for plain text
    ▼
  Chunk  →  One of five strategies (see below)
    │
    ▼
  Embed  →  Embedding model converts each chunk to a dense vector
    │
    ▼
  Store  →  Vectors + text + metadata written to vector store index
```

#### Retrieval (Runtime)

```
User Query
    │
    ▼
Embed Query  →  Same embedding model as ingestion
    │
    ▼
Vector Search  →  ANN similarity search (+ optional BM25 for hybrid)
    │
    ▼
Metadata Filter  →  Pre- or post-retrieval (explicit or auto-generated)
    │
    ▼
Rerank  →  Optional cross-encoder (Amazon Rerank 1.0 or Cohere Rerank 3.5)
    │
    ▼
Prompt Augmentation  →  Top-K chunks injected into LLM context
    │
    ▼
Response + Citations
```

---

## Part 2: Chunking Strategies

Chunking is one of the highest-leverage decisions in any RAG pipeline. Bedrock offers five strategies, selectable at data source creation time (the choice is permanent — it cannot be changed after a data source is connected).

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

---

## Part 3: Embedding Models

The embedding model must be chosen at knowledge base creation and cannot be changed without re-indexing. The same model is used for both ingestion and query-time retrieval.

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

---

## Part 4: Vector Store Integration

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

---

## Part 5: Retrieval Methods

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

### Metadata Filtering

Attach custom attributes to documents at ingestion via `.metadata.json` sidecar files (max 10 KB, same S3 folder as source), then filter at query time.

Supported operators: `equals`, `notEquals`, `greaterThan`, `greaterThanOrEquals`, `lessThan`, `lessThanOrEquals`, `stringContains`, `startsWith` (OpenSearch Serverless only), `listContains`, `in`, `notIn`, `andAll`, `orAll`.

**Auto-generated (Implicit) Filters** (GA December 2024): Set `implicitFilterConfiguration` with a model ARN and attribute descriptions, and Bedrock uses an LLM to extract filter values from natural language queries automatically. Query: *"How do I file a claim in Washington?"* → filter: `{state: "Washington"}` — no application-layer parsing needed.

### Query Reformulation

For multi-hop or compound questions, Bedrock decomposes the query into multiple sub-queries, runs each through retrieval independently, pools the results, and re-ranks them before passing context to the FM. Particularly effective for queries like *"How does our return policy differ between domestic and international orders?"*

### Reranking (GA December 2024)

A cross-encoder model re-scores retrieved candidates against the query after initial vector retrieval, producing a more semantically coherent top-K set.

```
Retrieve API (numberOfResults: 20)
    │
    ▼
Initial top-20 candidates (by cosine similarity)
    │
    ▼
Reranker (Amazon Rerank 1.0 or Cohere Rerank 3.5)
  → Jointly encodes (query, document) pairs
  → Scores each pair: relevance ∈ [0, 1]
    │
    ▼
Top-5 reranked results returned (numberOfRerankedResults: 5)
```

**Model comparison:**

| Model | Context Window | Languages | Handles Semi-Structured Data |
|---|---|---|---|
| Amazon Rerank 1.0 | Standard | English-focused | No |
| Cohere Rerank 3.5 | 32,000 tokens | 100+ | Yes (JSON, emails, tables) |

Configure via `rerankingConfiguration` in `vectorSearchConfiguration`, or use the standalone `Rerank` API for non-KB retrieval sources.

---

## Part 6: GraphRAG with Neptune Analytics

Standard RAG treats documents as independent chunks — it cannot reason about *relationships* between entities across documents. GraphRAG, GA'd March 2025, addresses this.

When you create a knowledge base backed by **Amazon Neptune Analytics**, Bedrock automatically:
1. Extracts entities and relationships from ingested documents
2. Builds a knowledge graph alongside vector embeddings
3. At query time, combines **vector similarity search + graph traversal**

This makes multi-hop questions tractable: *"Which compliance policies apply to the products mentioned in the Q3 risk report?"* A vector-only search would retrieve chunks about each topic independently; GraphRAG can traverse entity connections to surface the relationship.

**Benchmark:** AWS partner Lettria reported up to 35% improvement in answer precision vs. vector-only retrieval (single-partner benchmark; results will vary by data structure).

No graph modeling expertise is required — the entity extraction and graph construction are fully automated.

---

## Part 7: Multimodal RAG

GA'd November 2025, multimodal RAG enables knowledge bases to ingest, embed, and retrieve across text, images, audio, and video.

The key enabler is **Amazon Nova Multimodal Embeddings** (GA October 2025):
- Single unified model for text, documents, images, video, and audio
- Output dimensions: 3,072 / 1,024 / 384 / 256 (via Matryoshka Representation Learning)
- Context: up to 8,192 tokens
- 200+ language support
- Leads benchmarks on ActivityNet (video) and TextCaps vs. TwelveLabs, Google Vertex AI, Cohere, Titan

Before multimodal embeddings, complex PDFs with charts, tables, and figures required the **Foundation Model parser** (Claude Sonnet or Haiku) to describe visual elements as text before embedding. That approach still works and is compatible with text-only embedding models.

---

## Part 8: Bedrock Agents — Orchestrating RAG in a ReAct Loop

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

## Part 9: Inline Agents (GA November 2024)

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

## Part 10: Multi-Agent Collaboration with RAG (GA March 2025)

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

## Part 11: Agent Memory

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

## Part 12: Amazon Bedrock AgentCore (GA October 2025)

AgentCore is a new product layer distinct from Bedrock Agents. Rather than a fixed ReAct orchestrator, it's a **production runtime infrastructure** for agents built with *any* framework — AWS's own Strands Agents SDK, LangGraph, CrewAI, LlamaIndex, or custom implementations.

### Seven Components

| Component | What It Provides |
|---|---|
| **Runtime** | Serverless container execution; 8-hour max session windows; per-user session isolation (microVMs); Agent-to-Agent (A2A) protocol on port 9000; stateful MCP server support |
| **Memory** | Managed semantic/summary/preference memory (no infra to manage); async consolidation; cross-agent memory sharing; metadata on memory records (up to 10 indexed keys, May 2026) |
| **Gateway** | Converts REST APIs, Lambda functions, and existing MCP servers into agent-compatible tools; semantic tool selection via vector similarity over tool descriptions; IAM + OAuth authorization |
| **Code Interpreter** | Sandboxed multi-language code execution environment |
| **Browser** | Secure cloud-hosted browser runtime for web automation tasks |
| **Observability** | Real-time end-to-end execution traces via CloudWatch |
| **Identity** | IAM and OAuth-based agent-to-tool and agent-to-agent authorization; integrates with Cognito, Microsoft Entra ID, Okta |

### Using RAG with AgentCore

With AgentCore, you invoke Knowledge Bases from *inside your agent code* as a tool:

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

AgentCore Runtime wraps this agent with session isolation, long-running execution support, and observability — without changing your agent code.

### When to Choose What

| Scenario | Best Choice |
|---|---|
| Pure RAG chatbot, no tool calls | Standalone KB + `RetrieveAndGenerate` |
| Multi-step reasoning + API calls + RAG | Bedrock Agents + KB association |
| BYO framework (LangGraph/CrewAI) + production scale | AgentCore Runtime + KB tool |
| Multi-agent with domain-specific RAG | Multi-agent Bedrock Agents OR AgentCore with A2A |
| Long-running autonomous tasks (hours) | AgentCore Runtime (8-hour session windows) |

---

## Part 13: Guardrails for RAG

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

## Part 14: Key Timeline — From GA to AgentCore

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

```
                        ┌─────────────────────────────────────┐
                        │        Data Sources                 │
                        │  S3 │ Confluence │ SharePoint │ Web │
                        └──────────────┬──────────────────────┘
                                       │ Ingestion Pipeline
                                       ▼
                        ┌─────────────────────────────────────┐
                        │       Amazon Bedrock Knowledge Bases│
                        │                                     │
                        │  Parse (Claude FM Parser / BDA)     │
                        │  Chunk (Hierarchical / Semantic)    │
                        │  Embed (Titan V2 / Nova Multimodal) │
                        │                                     │
                        │  ┌─────────────┐  ┌─────────────┐  │
                        │  │  OpenSearch │  │   Neptune   │  │
                        │  │ Serverless  │  │  Analytics  │  │
                        │  │ (Text KBs)  │  │  (GraphRAG) │  │
                        │  └─────────────┘  └─────────────┘  │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │      AgentCore Runtime              │
                        │  (Session isolation, 8hr windows,  │
                        │   A2A protocol, Observability)      │
                        │                                     │
                        │  ┌─────────────────────────────┐   │
                        │  │    Supervisor Agent          │   │
                        │  │    (ReAct / LangGraph)       │   │
                        │  └──────┬──────────┬───────────┘   │
                        │         │          │                │
                        │  ┌──────▼──┐  ┌───▼──────┐         │
                        │  │  Tech   │  │  Policy  │         │
                        │  │  Agent  │  │  Agent   │         │
                        │  │  +KB    │  │  +KB     │         │
                        │  └─────────┘  └──────────┘         │
                        │                                     │
                        │  AgentCore Memory (Long-Term)       │
                        └──────────────┬──────────────────────┘
                                       │
                        ┌──────────────▼──────────────────────┐
                        │         Bedrock Guardrails          │
                        │  Contextual Grounding │ PII Filter  │
                        │  Automated Reasoning  │ Hallucination│
                        └──────────────┬──────────────────────┘
                                       │
                                  User Response
                               (with citations)
```

---

## Conclusion

AWS Bedrock's RAG stack has matured from a simple document-to-vector pipeline into a comprehensive agentic platform. The key architectural decisions that will shape your implementation:

1. **Chunking strategy** determines retrieval precision more than almost any other parameter — invest time benchmarking semantic vs. hierarchical chunking for your document types.

2. **Hybrid search** should be your default for production systems where users mix semantic questions with exact keyword lookups.

3. **Reranking** provides a meaningful accuracy boost for under-$0.001-per-query cost — enable it on production workloads.

4. **GraphRAG** unlocks multi-hop reasoning that vector-only retrieval cannot achieve; evaluate it if your documents have rich entity relationships.

5. **AgentCore vs. Bedrock Agents** is a framework flexibility question: choose Bedrock Agents for AWS-managed ReAct with minimal code, choose AgentCore when you need BYO framework, long-running sessions, or A2A multi-agent coordination.

6. **Guardrails with contextual grounding** should be non-negotiable in production — hallucination in RAG systems is subtle and difficult to detect without automated verification.

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
