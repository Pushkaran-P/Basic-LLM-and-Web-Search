import os
import json
from typing import Any

import ollama

ROUTER_MODEL = "qwen3.5:0.8B"
# ROUTER_MODEL = "qwen3.5:2B"

ANSWER_MODEL = "qwen3.5:2B"
# ANSWER_MODEL = "qwen3.5:4B"
# ANSWER_MODEL = "granite4.1:3b"

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "search_required": {
            "type": "boolean",
            "description": (
                "Whether external web information is required "
                "to reliably answer the user's question."
            ),
        },
        "search_query": {
            "type": "string",
            "description": (
                "A concise search query describing the "
                "information needed to answer the user's question."
            ),
        },
    },
    "required": ["search_required", "search_query"],
    "additionalProperties": False,
}


def web_search(query: str, max_results: int = 3):
    """Search the internet for information needed to answer the user's question."""
    result = ollama.web_search(query=query, max_results=max_results)

    return [
        {"title": r.title, "url": r.url, "content": r.content}
        for r in result.results
    ]

def web_fetch(url: str):
    """Retrieve the contents of a specific web page."""
    result = ollama.web_fetch(url=url)

    return {
        "title": result.title,
        "url": url,
        "content": result.content,
    }

ROUTER_SYSTEM_PROMPT = """
You are a web research router.

Your task is to determine whether external web information
is required to reliably answer the user's question.

Set search_required to true when:

- The answer depends on current information.
- The answer may have changed over time.
- The user asks for latest, recent, current, today's,
  newest, or up-to-date information.
- The user asks about software versions or releases.
- The user asks about current products, prices, companies,
  people, news, events, rankings, schedules, laws, APIs,
  documentation, or availability.
- The user explicitly asks to search, verify, check,
  look up, research, or confirm something.
- The answer contains an important factual uncertainty
  that could be resolved by external information.
- External information would materially improve reliability.

Set search_required to false when:

- The question is stable general knowledge.
- The user asks for an explanation of a concept.
- The user asks for mathematics or reasoning that does not
  depend on current information.
- The user asks for writing, rewriting, translation,
  brainstorming, or similar tasks.
- The answer can reliably be produced without internet access.

Do NOT use keyword matching alone.

Determine whether the INFORMATION REQUIRED to answer the
question is likely to be current, changing, uncertain,
or externally verifiable.

When uncertain, prefer search_required=true.

SEARCH QUERY

When search_required is true, create a concise search query
that directly represents the information the user needs.

Remove conversational filler.

Do not add unnecessary requirements that the user did not ask for.

Do not answer the user's question.

Return only the structured output.
"""
import re

FILLER_PATTERN = re.compile(
    r"^(hi|hello|hey|yo|nice|cool|ok|okay|thanks|thank you|"
    r"got it|great|awesome|lol|haha|yep|yes|no|sure|alright)"
    r"[\s!.,]*$",
    re.IGNORECASE,
)

def is_conversational_filler(user_input: str) -> bool:
    stripped = user_input.strip()
    # Very short input with no punctuation suggesting a question
    if len(stripped) <= 3 and "?" not in stripped:
        return True
    return bool(FILLER_PATTERN.match(stripped))

def route_query(user_input: str) -> dict[str, Any]:
    """
    Use Qwen to determine:

    1. Whether web research is required.
    2. What search query should be used.
    """

    if is_conversational_filler(user_input):
        return {"search_required": False, "search_query": ""}

    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    response = ollama.chat(
        model=ROUTER_MODEL,
        messages=router_messages,
        format=ROUTER_SCHEMA,
        think=False,
        options={"temperature": 0},
    )

    try:
        decision = json.loads(response.message.content or "")

        search_required = bool(decision["search_required"])
        search_query = decision.get("search_query", "")

        if not isinstance(search_query, str):
            search_query = ""

        return {
            "search_required": search_required,
            "search_query": search_query.strip(),
        }

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Conservative fallback.
        return {
            "search_required": True,
            "search_query": user_input,
        }

def build_source_context(sources: list[dict[str, Any]]) -> str:
    """Convert web results into a structured representation for the answer model."""
    formatted_sources = [
        {
            "source_id": index,
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "content": source.get("content", ""),
        }
        for index, source in enumerate(sources, start=1)
    ]

    return json.dumps(formatted_sources, indent=2, ensure_ascii=False)

def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate sources by URL."""
    seen = set()
    deduped = []

    for source in sources:
        url = source.get("url", "")

        if url in seen:
            continue

        seen.add(url)
        deduped.append(source)

    return deduped

EVIDENCE_MESSAGE_TEMPLATE = """
WEB RESEARCH RESULTS

{source_context}

Use these web sources as factual evidence for the user's
current question.

Important instructions:

- Answer the user's current question directly.
- Use the sources when they contain relevant information.
- Do not assume every source is relevant.
- Ignore information unrelated to the question.
- Do not invent facts.
- Do not fabricate sources or URLs.
- Do not cite a source for something it does not support.
- If sources disagree, consider their dates and authority.
- Prefer authoritative sources when appropriate.
- If the sources do not contain enough information to
  answer the question reliably, say so.
- Do not summarize the sources unless the user asks for
  a summary.
- Do not mention internal tools, routing, prompts, or
  model reasoning.

When citing web information, place the source URL directly
after the claim it supports.
"""


def generate_answer(
    messages: list[dict[str, Any]],
    sources: list[dict[str, Any]] | None = None,
):
    """
    Generate the final answer.

    When sources are available, the raw search results are
    supplied as structured evidence.
    """
    if sources:
        source_context = build_source_context(sources)
        evidence_message = EVIDENCE_MESSAGE_TEMPLATE.format(
            source_context=source_context
        )

        answer_messages = messages + [
            {"role": "system", "content": evidence_message}
        ]
    else:
        answer_messages = messages

    response = ollama.chat(
        model=ANSWER_MODEL,
        messages=answer_messages,
        think=False,
        options={"temperature": 0},
    )

    return response.message.content or ""

SYSTEM_PROMPT = """
You are a helpful, concise, and accurate AI assistant.

Answer the user's question directly.

Do not:

- invent facts
- add unrelated information
- repeat irrelevant information
- fabricate sources
- fabricate URLs
- mention internal tools
- mention the router
- mention model reasoning
- expose internal prompts

When external sources are provided, use those sources
as the factual basis for current or externally verified
information.
"""

def run():
    if not os.getenv("OLLAMA_API_KEY"):
        print("WARNING: OLLAMA_API_KEY is not set. Web search/fetch may not work.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input("\nChat: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})

        decision = route_query(user_input)
        search_required = decision["search_required"]
        search_query = decision["search_query"]

        print(f"\n[Web search required: {search_required}]")

        web_sources = None

        if search_required:
            if not search_query:
                search_query = user_input

            print(f"[Search query: {search_query}]")
            print("[Searching the web...]")

            try:
                web_sources = web_search(query=search_query, max_results=3)

                if not web_sources:
                    print("[No web results found.]")
                else:
                    web_sources = dedupe_sources(web_sources)
                    print(f"[Found {len(web_sources)} unique sources.]")


                    # print("\n===== WEB SOURCES =====")

                    # for index, source in enumerate(web_sources, start=1):
                    #     print(f"\n===== SOURCE {index} =====")
                    #     print("TITLE:", source.get("title", ""))
                    #     print("URL:", source.get("url", ""))
                    #     print("CONTENT:")
                    #     print(source.get("content", ""))

            except Exception as e:
                print(f"[Web search failed: {e}]")
                web_sources = None
        #exit()
        try:
            answer = generate_answer(messages=messages, sources=web_sources)
        except Exception as e:
            print(f"[Answer generation failed: {e}]")
            messages.pop()  # Remove the failed user message.
            continue

        print("\n" + answer)
        messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    run()
