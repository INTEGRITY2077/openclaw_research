# OpenClaw Memory Management Code Review

## 1. Architecture Overview

OpenClaw's memory management is a multi-layered system spanning 6 major subsystems:

```
                    ┌─────────────────────┐
                    │   Agent Tools        │
                    │  (memory_search,     │
                    │   memory_get)        │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐  ┌────▼─────┐  ┌──────▼──────────┐
    │ Memory Index   │  │ Context  │  │  Memory Flush   │
    │ Manager        │  │ Pruning  │  │  (Pre-Compaction)│
    │ (Vector+FTS)   │  │          │  │                  │
    └─────────┬──────┘  └────┬─────┘  └──────┬──────────┘
              │              │               │
    ┌─────────▼──────┐  ┌───▼──────┐  ┌─────▼───────────┐
    │ SQLite DB      │  │ Session  │  │  Compaction      │
    │ (chunks, FTS5, │  │ Transcript│  │  (Summarization) │
    │  sqlite-vec)   │  │ (JSONL)  │  │                  │
    └────────────────┘  └──────────┘  └─────────────────┘
```

**Core Files:**
- `src/memory/manager.ts` (~1500 LOC) - Central `MemoryIndexManager`
- `src/agents/compaction.ts` (305 LOC) - Session summarization
- `src/agents/pi-extensions/context-pruning/pruner.ts` (347 LOC) - In-flight context trimming
- `src/auto-reply/reply/memory-flush.ts` (106 LOC) - Pre-compaction memory write
- `src/agents/pi-embedded-runner/compact.ts` (498 LOC) - Compaction executor

---

## 2. Subsystem-Level Findings

### 2.1 MemoryIndexManager (`src/memory/manager.ts`)

**Strengths:**
- **Singleton cache pattern** (`INDEX_CACHE`) prevents duplicate managers for the same agent+workspace combination. Cache key includes serialized settings, ensuring config changes create new instances.
- **Graceful degradation**: sqlite-vec and FTS5 are optional. If unavailable, the system falls back to in-memory cosine similarity or keyword-only search respectively.
- **Incremental indexing**: Hash-based change detection (`sync-memory-files.ts:56`) avoids re-embedding unchanged files. This is critical for controlling embedding API costs.
- **File watcher with debounce** (`ensureWatcher`, line ~848): Chokidar watches MEMORY.md and memory/ directory with `awaitWriteFinish` to avoid partial-write indexing.

**Concerns:**

1. **Module-level global cache without eviction** (`manager.ts:112`)
   ```typescript
   const INDEX_CACHE = new Map<string, MemoryIndexManager>();
   ```
   `INDEX_CACHE` is a module-level `Map` that grows without bound. While `close()` deletes the entry, if managers are created for many agent/workspace combinations without being closed (e.g., in long-running gateway processes), this becomes a memory leak. The cache key includes the full serialized settings JSON, which can cause duplicate entries for semantically identical configs.

2. **`EMBEDDING_APPROX_CHARS_PER_TOKEN = 1`** (`manager.ts:97`)
   This constant is suspicious. Standard English text is typically 3-4 characters per token. Setting this to 1 means the batching logic (`EMBEDDING_BATCH_MAX_TOKENS = 8000`) would batch by ~8000 characters rather than ~8000 tokens. This results in very small batches (about 2000 actual tokens per batch), which increases API round trips unnecessarily.

3. **Synchronous SQLite in an async context** (`manager.ts:140`)
   The class uses `DatabaseSync` (Node.js synchronous SQLite API) throughout, including in async methods like `search()` and `sync()`. This blocks the event loop during database operations. For small memory indices this is tolerable, but with large chunk counts (thousands of embedded documents), operations like the full cosine similarity fallback (`manager-search.ts:71-93`) that load all chunks into memory and compute pairwise similarity will cause noticeable latency.

4. **Race condition in `sync()` coalescing** (`manager.ts:395-407`)
   ```typescript
   async sync(params?) {
     if (this.syncing) return this.syncing;
     this.syncing = this.runSync(params).finally(() => { this.syncing = null; });
     return this.syncing;
   }
   ```
   If `sync()` is called with `force: true` while a non-forced sync is running, the forced sync is silently dropped. The caller gets back the promise of the non-forced sync, which may not perform the full reindex the caller expects.

5. **Symlink traversal guard is incomplete** (`internal.ts:61-62`, `internal.ts:114`)
   `walkDir` correctly skips symlinks at the entry level, but `normalizeExtraMemoryPaths` resolves paths with `path.resolve()` which does not resolve symlinks. If a user provides a path like `../symlink-to-sensitive-dir/`, it would pass path validation but the actual resolved directory might be outside the workspace. The `readFile` method (`manager.ts:409-472`) does check `lstat` for symlinks on the final file, which mitigates the risk partially.

### 2.2 Hybrid Search (`src/memory/hybrid.ts`, `src/memory/manager-search.ts`)

**Strengths:**
- Clean separation between vector and keyword search with configurable weight merging.
- `bm25RankToScore` normalization (`hybrid.ts:36-39`) correctly converts BM25 rank (where lower is better) to a 0-1 score.
- FTS5 query builder (`hybrid.ts:23-34`) sanitizes input by stripping non-alphanumeric characters, preventing FTS injection.

**Concerns:**

6. **Full table scan fallback** (`manager-search.ts:71-93`)
   When sqlite-vec is unavailable, `searchVector` falls back to loading **all** chunks from the database, deserializing their embeddings, and computing cosine similarity in JavaScript. For a large index this is O(n) in both memory and compute. There is no pagination or limit on the fallback query.

7. **Dimension mismatch silently degrades** (`internal.ts:258-277`)
   `cosineSimilarity` uses `Math.min(a.length, b.length)` and silently ignores extra dimensions. If the embedding model changes (e.g., OpenAI `text-embedding-3-small` at 1536 dims vs Gemini at different dims), old chunks remain in the index with mismatched dimensions. The search will silently compute partial cosine similarity, returning degraded but non-obvious results.

### 2.3 Context Compaction (`src/agents/compaction.ts`)

**Strengths:**
- **Progressive summarization** (`summarizeWithFallback`): Three-tier fallback (full -> partial excluding oversized -> descriptive note). This is robust against context window limits.
- **Adaptive chunk ratio** (`computeAdaptiveChunkRatio`): Dynamically reduces chunk size when average message tokens exceed 10% of context window, preventing summarization attempts on messages that are too large.
- **Safety margin** (`SAFETY_MARGIN = 1.2`): 20% buffer accounts for `estimateTokens()` inaccuracy. This is a pragmatic choice.

**Concerns:**

8. **O(n^2) token estimation in `pruneHistoryForContextShare`** (`compaction.ts:330`)
   ```typescript
   while (keptMessages.length > 0 && estimateMessagesTokens(keptMessages) > budgetTokens) {
     const chunks = splitMessagesByTokenShare(keptMessages, parts);
     ...
   }
   ```
   Each loop iteration calls `estimateMessagesTokens()` on `keptMessages`, which iterates all messages. `splitMessagesByTokenShare` also iterates all messages. In the worst case, this is O(n * k) where k is the number of drop rounds. Could be optimized by maintaining a running token total.

9. **No cap on summarization API calls** (`summarizeInStages`, `compaction.ts:243-304`)
   `summarizeInStages` splits messages, summarizes each chunk (sequential API call per chunk), then merges summaries with another API call. For very long histories with many parts, this generates an unbounded number of API calls. The `parts` parameter defaults to 2 but is caller-controllable.

### 2.4 Context Pruning (`src/agents/pi-extensions/context-pruning/pruner.ts`)

**Strengths:**
- **Non-destructive**: Operates on a copy of the messages array (shallow clone via `messages.slice()`), leaving the persisted session untouched.
- **Image safety**: Explicitly skips pruning tool results that contain image blocks.
- **Two-stage approach**: Soft trim (head+tail with truncation note) -> hard clear (replace with placeholder). Progressive degradation under pressure.
- **Bootstrap protection** (`pruner.ts:253-257`): Never prunes messages before the first user message, protecting identity/config reads.

**Concerns:**

10. **Character-based estimation vs token-based** (`pruner.ts:7`)
    ```typescript
    const CHARS_PER_TOKEN_ESTIMATE = 4;
    ```
    The pruner uses character length * 4 to estimate tokens, while the compaction system uses `estimateTokens()` from the Pi library. These different estimation approaches could cause the pruner and compactor to disagree about when context is "full", potentially leading to redundant pruning or missed pruning.

11. **Shallow clone with index mutation** (`pruner.ts:296-298`)
    ```typescript
    if (!next) { next = messages.slice(); }
    next[i] = updated as unknown as AgentMessage;
    ```
    `messages.slice()` is a shallow copy - individual message objects are shared. While this function only replaces array elements (doesn't mutate objects in place), the multiple `as unknown as AgentMessage` casts indicate type friction between the `ToolResultMessage` and `AgentMessage` types. This suggests the type hierarchy could be cleaner.

### 2.5 Memory Flush (`src/auto-reply/reply/memory-flush.ts`, `agent-runner-memory.ts`)

**Strengths:**
- **One flush per compaction cycle**: Tracked via `memoryFlushCompactionCount` in session metadata, preventing redundant flush runs.
- **Sandbox awareness**: Checks `workspaceAccess === "rw"` before attempting flush.
- **Silent by default**: Uses `SILENT_REPLY_TOKEN` so the flush doesn't generate visible output to users.

**Concerns:**

12. **Threshold calculation edge case** (`memory-flush.ts:90-91`)
    ```typescript
    const threshold = Math.max(0, contextWindow - reserveTokens - softThreshold);
    if (threshold <= 0) return false;
    ```
    If `reserveTokens + softThreshold >= contextWindow`, flush is disabled entirely. With default values (`softThresholdTokens=4000`, `reserveTokensFloor` from Pi settings), this could silently disable memory flush for small context window models (e.g., some 8K models).

13. **Error swallowed silently** (`agent-runner-memory.ts:196-198`)
    ```typescript
    } catch (err) {
      logVerbose(`memory flush run failed: ${String(err)}`);
    }
    ```
    The entire memory flush failure is logged only at verbose level. If the flush consistently fails (e.g., bad API key, rate limits), the user has no visibility into the issue unless verbose logging is enabled.

### 2.6 Compaction Executor (`src/agents/pi-embedded-runner/compact.ts`)

**Strengths:**
- **Write lock** (`compact.ts:357-358`): `acquireSessionWriteLock` prevents concurrent compaction/write operations on the same session file.
- **Lane queueing** (`compact.ts:487-497`): `compactEmbeddedPiSession` wraps the direct call in session+global lane queues, preventing deadlocks between concurrent operations.
- **Proper cleanup** in finally blocks: `sessionManager.flushPendingToolResults()`, `session.dispose()`, lock release, and `process.chdir(prevCwd)`.

**Concerns:**

14. **`process.chdir()` is not process-safe** (`compact.ts:186, 478`)
    ```typescript
    process.chdir(effectiveWorkspace);
    // ... async operations ...
    process.chdir(prevCwd);
    ```
    `process.chdir` changes the working directory for the entire Node.js process. If multiple compaction operations run concurrently (even with lane queueing), any interleaving between the `chdir` calls and async operations could cause one operation to see another's working directory. The lane queueing mitigates this, but only if all code paths that do compaction go through the lane.

15. **Token estimation sanity check may mask issues** (`compact.ts:445-447`)
    ```typescript
    if (tokensAfter > result.tokensBefore) {
      tokensAfter = undefined; // Don't trust the estimate
    }
    ```
    If tokens increase after compaction, the estimate is silently discarded. However, tokens increasing after compaction is a genuine bug (summary + kept messages should be smaller). Setting `undefined` hides a potential issue where the compaction didn't actually reduce context.

### 2.7 Schema & Database (`src/memory/memory-schema.ts`)

**Strengths:**
- Migration-friendly: `ensureColumn` uses `PRAGMA table_info` to check before `ALTER TABLE`, making the schema safely additive.
- FTS5 creation is try/catch wrapped, allowing the system to degrade gracefully on SQLite builds without FTS5.

**Concerns:**

16. **SQL injection via table name interpolation** (`memory-schema.ts:39, 58-67`)
    ```typescript
    params.db.exec(`CREATE TABLE IF NOT EXISTS ${params.embeddingCacheTable} ...`);
    params.db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS ${params.ftsTable} USING fts5(...)`);
    ```
    Table names are interpolated directly into SQL strings. While these are currently hardcoded constants (`EMBEDDING_CACHE_TABLE`, `FTS_TABLE`), the function signature accepts arbitrary strings. If a caller ever passes user-controlled input, this becomes SQL injection. The `ensureColumn` function (`memory-schema.ts:91`) has the same pattern with `table` and `column`.

---

## 3. Cross-Cutting Concerns

### 3.1 Token Estimation Inconsistency
Three different token estimation approaches coexist:
- `estimateTokens()` from `@mariozechner/pi-coding-agent` (compaction)
- `CHARS_PER_TOKEN_ESTIMATE = 4` (context pruning)
- `EMBEDDING_APPROX_CHARS_PER_TOKEN = 1` (memory index batching)

These inconsistencies mean the same content can be estimated at vastly different token counts depending on which subsystem is measuring. This is a source of subtle bugs where subsystems disagree about when action is needed.

### 3.2 Error Handling Strategy
The codebase consistently uses a "swallow and degrade" pattern:
- Empty `catch {}` blocks in `walkDir`, `listMemoryFiles`, `normalizeExtraMemoryPaths`
- Verbose-only logging for memory flush failures
- Fallback chains for embedding providers

This makes the system resilient but hard to debug. When memory search returns poor results, there's no easy way to determine if FTS5 failed to load, embeddings are from the wrong model, or the vector extension timed out.

### 3.3 Concurrency Model
The system uses multiple concurrency control mechanisms:
- **Lane queueing** for compaction (per-session + global)
- **Write locks** for session files
- **Promise coalescing** for sync operations
- **Debounced timers** for file watching

These layers are well-designed for the gateway's concurrent request model, but the interaction between them is complex. The documentation of which locks are held when calling which functions is implicit (via function names and JSDoc), rather than enforced by types.

---

## 4. Positive Patterns Worth Highlighting

1. **Fallback chain architecture**: Embedding provider resolution (local -> remote -> none) is well-structured and makes the system usable across different deployment environments.

2. **Hash-based incremental indexing**: Using SHA-256 content hashes to avoid re-embedding unchanged files is correct and cost-effective.

3. **Separation of search concerns**: The split between `manager-search.ts` (raw DB queries), `hybrid.ts` (score merging), and `manager.ts` (orchestration) keeps each layer focused.

4. **Pre-compaction memory flush**: The idea of giving the agent a chance to persist important memories before context is compacted is architecturally sound and unique.

5. **Non-destructive context pruning**: Pruning operates on copies and only affects the current request, not the persisted transcript.

---

## 5. Summary of Priority Issues

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1 | Medium | `manager.ts:112` | INDEX_CACHE unbounded growth |
| 2 | Low | `manager.ts:97` | EMBEDDING_APPROX_CHARS_PER_TOKEN=1 seems miscalibrated |
| 4 | Medium | `manager.ts:395-407` | Force-sync coalesced with non-force sync |
| 6 | Medium | `manager-search.ts:71-93` | Full table scan fallback with no size limit |
| 7 | Medium | `internal.ts:258-277` | Silent dimension mismatch in cosine similarity |
| 8 | Low | `compaction.ts:330` | Quadratic token estimation in prune loop |
| 10 | Low | Cross-cutting | Inconsistent token estimation across subsystems |
| 12 | Low | `memory-flush.ts:90-91` | Flush silently disabled on small context windows |
| 14 | Medium | `compact.ts:186` | process.chdir not safe for concurrent operations |
| 16 | Low | `memory-schema.ts` | SQL table/column name interpolation |

---

*Review generated: 2026-02-11*
*Scope: `src/memory/`, `src/agents/compaction.ts`, `src/agents/pi-extensions/context-pruning/`, `src/auto-reply/reply/memory-flush.ts`, `src/agents/pi-embedded-runner/compact.ts`*
