import time

from rag_store import Embedder, build_or_load_store
from rag_utils import project_paths

DEFAULT_LLM = "llama3.2"
DEFAULT_TOP_K = 4
DEFAULT_MIN_SCORE = 0.2
DEFAULT_MAX_CONTEXT_CHARS = 4000

# The exact string the system prompt tells the model to use when the context
# does not contain the answer. Evaluation looks for it, so it lives in one place.
NO_CONTEXT_ANSWER = "I don't know based on the provided documents."


# ===========================================================================
# 1. The retriever
# ===========================================================================

class Retriever:
    """Question text -> the chunks most likely to contain the answer."""

    def __init__(self, store, embedder, top_k=DEFAULT_TOP_K, min_score=DEFAULT_MIN_SCORE):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query, top_k=None, min_score=None, source=None):
        """Retrieve the best chunks for a query, best first.

        `source` restricts the answer to one file. We over-fetch before
        filtering, because filtering a top-4 list by source usually leaves you
        with nothing.
        """
        top_k = self.top_k if top_k is None else top_k
        min_score = self.min_score if min_score is None else min_score
        # The query goes through the SAME model as the documents. Two models =
        # two unrelated vector spaces = confident nonsense.
        query_vector = self.embedder.encode_query(query)
        fetch = top_k * 5 if source else top_k          # over-fetch, then filter
        hits = self.store.search(query_vector, top_k=fetch, min_score=min_score)
        if source:
            hits = [hit for hit in hits if hit["metadata"].get("source") == source]
        hits = hits[:top_k]
        for rank, hit in enumerate(hits, start=1):      # renumber after filtering
            hit["rank"] = rank
        return hits


# ===========================================================================
# 2. Prompt construction
# ===========================================================================

SYSTEM_PROMPT = """You are a careful assistant for a technical knowledge base.

Rules:
1. Answer using ONLY the numbered context passages provided by the user.
2. If the context does not contain the answer, reply exactly:
   "I don't know based on the provided documents."
3. Cite the passage number in square brackets after each claim, like [2].
4. Never invent facts, numbers, file names or citations.
5. Be concise: two to five sentences unless the question asks for more."""

USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Answer (cite passages as [1], [2], ...):"""


def build_context(hits, max_chars=DEFAULT_MAX_CONTEXT_CHARS):
    """Retrieved chunks -> a numbered, source-tagged context block.

    The character budget is not optional. The context window is finite and every
    token costs money and latency. When the budget runs out we stop adding whole
    passages rather than truncating one mid-sentence, so everything the model
    sees is complete.
    """
    blocks, used = [], 0
    for rank, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        page = f", page {meta['page']}" if meta.get("page") else ""
        header = f"[{rank}] (source: {meta.get('source', 'unknown')}{page})"
        block = f"{header}\n{hit['text'].strip()}"
        if used + len(block) > max_chars and blocks:      # keep at least one passage
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) if blocks else "(no relevant passages found)"


def build_prompt(question, hits, max_chars=DEFAULT_MAX_CONTEXT_CHARS):
    """The two halves of what we send: standing rules, and this specific turn."""
    return {
        "system": SYSTEM_PROMPT,
        "user": USER_TEMPLATE.format(context=build_context(hits, max_chars),
                                     question=question),
    }


# ===========================================================================
# 3. Generation
# ===========================================================================

def generate_answer(prompt, model=DEFAULT_LLM, temperature=0.0, max_tokens=400):
    """Send a {system, user} prompt to a local Ollama model, get text back.

    temperature=0.0 on purpose: a RAG answer should be a faithful reading of the
    context, not a creative one, and a deterministic model makes the evaluation
    below mean something.
    """
    import ollama  # lazy: importing it should not be the cost of importing this module

    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": prompt["system"]},
                  {"role": "user", "content": prompt["user"]}],
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return response["message"]["content"].strip()


class OllamaGenerator:
    """The LLM behind a one-call interface: prompt dict -> answer string.

    RAGPipeline only ever calls `generator(prompt)`, so swapping in a different
    backend - an API model, or a canned function in a test - means passing a
    different callable, not editing the pipeline.
    """

    def __init__(self, model=DEFAULT_LLM, temperature=0.0, max_tokens=400):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, prompt, max_tokens=None):
        return generate_answer(prompt, model=self.model,
                               temperature=self.temperature,
                               max_tokens=self.max_tokens if max_tokens is None else max_tokens)


# ===========================================================================
# 4. The complete pipeline
# ===========================================================================

def format_sources(hits):
    """De-duplicated, human-readable source list."""
    seen, sources = set(), []
    for hit in hits:
        meta = hit.get("metadata", {})
        label = meta.get("source", "unknown")
        if meta.get("page"):
            label += f" page {meta['page']}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


class RAGPipeline:
    """retrieve -> prompt -> generate -> cite, with the intermediates kept."""

    def __init__(self, retriever, generator=None,
                 max_context_chars=DEFAULT_MAX_CONTEXT_CHARS):
        self.retriever = retriever
        self.generator = generator if generator is not None else OllamaGenerator()
        self.max_context_chars = max_context_chars

    def ask(self, question, top_k=None, min_score=None, max_context_chars=None,
            verbose=False):
        """question -> {answer, sources, hits, prompt, grounded, timings}"""
        started = time.perf_counter()

        # 1. retrieve
        hits = self.retriever.retrieve(question, top_k=top_k, min_score=min_score)
        retrieved_at = time.perf_counter()
        if verbose:
            print(f"  retrieved {len(hits)} chunks in "
                  f"{(retrieved_at - started) * 1000:.0f} ms")

        # short-circuit: nothing to ground on, so do not pay for a generation
        # that can only hallucinate. This is the system saying "I don't know".
        if not hits:
            return {"question": question, "answer": NO_CONTEXT_ANSWER, "sources": [],
                    "hits": [], "prompt": None, "grounded": False,
                    "retrieval_s": retrieved_at - started, "generation_s": 0.0}

        # 2. build the prompt
        prompt = build_prompt(question, hits,
                              self.max_context_chars if max_context_chars is None
                              else max_context_chars)

        # 3. generate
        answer = self.generator(prompt)
        finished = time.perf_counter()

        # 4. return everything, not just the answer: the hits and the prompt are
        #    what you will need the moment the answer looks wrong.
        return {"question": question, "answer": answer, "sources": format_sources(hits),
                "hits": hits, "prompt": prompt, "grounded": True,
                "retrieval_s": retrieved_at - started,
                "generation_s": finished - retrieved_at}

    __call__ = ask


def format_result(result):
    """The answer plus its receipts, as a printable string."""
    lines = [result["answer"]]
    if result["hits"]:
        lines.append("")
        lines.append("Sources:")
        for hit in result["hits"]:
            meta = hit.get("metadata", {})
            page = f" page {meta['page']}" if meta.get("page") else ""
            lines.append(f"  [{hit['rank']}] {meta.get('source', 'unknown')}{page}  "
                         f"(score {hit['score']:.3f})")
    else:
        lines.append("")
        lines.append("Sources: none - nothing in the knowledge base was relevant.")
    lines.append("")
    lines.append(f"({result['retrieval_s'] * 1000:.1f} ms retrieval, "
                 f"{result['generation_s']:.1f} s generation)  "
                 f"<- retrieval is essentially free; the LLM is the whole latency budget")
    return "\n".join(lines)


def print_result(result):
    print(format_result(result))


# ===========================================================================
# 5. Evaluation
# ===========================================================================

# question, expected keywords in the answer, expected source file, should refuse?
TEST_SET = [
    ("What is the attention mechanism?",
     ["quer", "key", "value"], "ai_course.pdf", False),
    ("What is deep learning?",
     ["layer"], "ai_course.pdf", False),
    ("Which index type should I use below 50,000 vectors?",
     ["flat"], "documentation.txt", False),
    ("How do I install FAISS with pip?",
     ["faiss-cpu"], "documentation.txt", False),
    ("What is the maximum value of k on GPU?",
     ["2048"], "documentation.txt", False),
    # --- questions the corpus cannot answer: the system must refuse ---
    ("How do I configure a Kubernetes ingress controller?", [], None, True),
    ("What is the boiling point of mercury?", [], None, True),
    ("Who won the 2027 Champions League final?", [], None, True),
]


def is_refusal(answer):
    """Did the model decline to answer? Checked loosely - models paraphrase."""
    lowered = answer.lower()
    return "don't know" in lowered or "do not know" in lowered


def evaluate(pipeline, test_set=TEST_SET, top_k=DEFAULT_TOP_K, min_score=DEFAULT_MIN_SCORE):
    """Run the test set and score two things separately.

    hit@k answers "did retrieval find the right document?" and the keyword check
    answers "did the LLM use it?". Collapsing them into one number is how you
    end up tuning the prompt to fix a chunking problem.
    """
    rows = []
    for question, keywords, expected_source, should_refuse in test_set:
        hits = pipeline.retriever.retrieve(question, top_k=top_k, min_score=min_score)
        retrieved_sources = {hit["metadata"].get("source") for hit in hits}
        hit_at_k = expected_source in retrieved_sources if expected_source else None
        try:
            answer = pipeline.ask(question, top_k=top_k, min_score=min_score)["answer"]
        except Exception as error:                      # a dead LLM is a result too
            answer = f"<generation failed: {error}>"
        refused = is_refusal(answer)
        found = [k for k in keywords if k.lower() in answer.lower()]
        if should_refuse:
            passed = refused
        else:
            passed = bool(keywords) and len(found) == len(keywords) and not refused
        rows.append({"question": question, "hit@k": hit_at_k, "refused": refused,
                     "found": found, "expected": keywords, "pass": passed,
                     "top_score": hits[0]["score"] if hits else 0.0,
                     "answer": answer})
    return rows


def format_evaluation(rows):
    """The scoreboard: one line per question, then the two totals that matter."""
    lines = [f"{'pass':>5} {'hit@k':>6} {'top':>6}  question", "-" * 78]
    for row in rows:
        hit = "-" if row["hit@k"] is None else ("yes" if row["hit@k"] else "NO")
        lines.append(f"{'PASS' if row['pass'] else 'FAIL':>5} {hit:>6} "
                     f"{row['top_score']:>6.2f}  {row['question'][:56]}")

    answerable = [row for row in rows if row["expected"]]
    refusals = [row for row in rows if not row["expected"]]
    lines.append("-" * 78)
    if answerable:
        lines.append(
            f"answerable questions: {sum(r['pass'] for r in answerable)}/{len(answerable)} correct, "
            f"retrieval hit@k {sum(bool(r['hit@k']) for r in answerable)}/{len(answerable)}")
    if refusals:
        lines.append(f"unanswerable questions: {sum(r['pass'] for r in refusals)}/"
                     f"{len(refusals)} correctly refused")
    return "\n".join(lines)


def format_failures(rows):
    """Read the failures. A PASS/FAIL table tells you nothing about WHY.

    hit@k = yes but wrong answer -> generation problem (prompt / model)
    hit@k = NO                   -> retrieval problem (chunking / top_k / embedding)
    refused but corpus has it    -> min_score too high, or phrasing mismatch
    """
    lines = []
    for row in rows:
        if row["pass"]:
            continue
        lines.append("=" * 78)
        lines.append(f"Q: {row['question']}")
        lines.append(f"   retrieval hit@k: {row['hit@k']}   "
                     f"top score: {row['top_score']:.3f}")
        lines.append(f"   expected keywords: {row['expected']}   found: {row['found']}")
        lines.append(f"   answer: {row['answer'][:400]}")
        lines.append("")
    return "\n".join(lines) if lines else "(everything passed)"


def sweep_retrieval(retriever, test_set=TEST_SET, top_ks=(1, 2, 4, 8),
                    min_scores=(0.0, 0.2, 0.35)):
    """Tune top_k and min_score without paying for a single LLM call.

    Two columns, pulling in opposite directions: hit@k wants to go up, and
    false context (passages handed over for a question the corpus cannot
    answer) wants to go down. Picking the knee is the whole exercise.
    """
    answerable = sum(1 for row in test_set if row[2])
    unanswerable = sum(1 for row in test_set if row[3])
    lines = [f"{'top_k':>6}{'min_score':>11}{'hit@k':>8}{'false ctx':>11}", "-" * 36]
    for top_k in top_ks:
        for min_score in min_scores:
            hit_count = false_context = 0
            for question, _, expected_source, should_refuse in test_set:
                hits = retriever.retrieve(question, top_k=top_k, min_score=min_score)
                sources = {hit["metadata"].get("source") for hit in hits}
                if expected_source and expected_source in sources:
                    hit_count += 1
                if should_refuse and hits:      # context for an unanswerable question
                    false_context += 1
            lines.append(f"{top_k:>6}{min_score:>11}{hit_count:>4}/{answerable:<3}"
                         f"{false_context:>7}/{unanswerable}")
    return "\n".join(lines)


# ===========================================================================
# 6. Convenience
# ===========================================================================

def build_pipeline(paths=None, embedder=None, model=DEFAULT_LLM, top_k=DEFAULT_TOP_K,
                   min_score=DEFAULT_MIN_SCORE, generator=None, rebuild=False):
    """Documents on disk -> a pipeline you can ask questions. One call.

    Reuses Session 2's saved store when it was built with the same embedding
    model, and rebuilds it otherwise - vectors from two models are not
    comparable, so a mismatched store is worse than no store.
    """
    paths = project_paths() if paths is None else paths
    embedder = Embedder() if embedder is None else embedder
    store = build_or_load_store(paths, embedder, rebuild=rebuild)
    if len(store):
        try:
            store.build_faiss()             # optional: search() falls back to numpy
        except ImportError:
            pass
    retriever = Retriever(store, embedder, top_k=top_k, min_score=min_score)
    if generator is None:
        generator = OllamaGenerator(model=model)
    return RAGPipeline(retriever, generator)


if __name__ == "__main__":
    rag = build_pipeline()
    print(f"knowledge base: {len(rag.retriever.store)} chunks, "
          f"{rag.retriever.store.dimension} dims, "
          f"model {rag.retriever.store.model_name}")
    print()
    print_result(rag.ask("What is the attention mechanism?"))
    print()
    print(format_evaluation(evaluate(rag)))
