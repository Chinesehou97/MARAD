GLOBAL_HEADER = ""

OUTPUT_PROTOCOL = """Strictly output the following three blocks (no code fences, no extra text):
AGENT: <agent_tag>
FACT_UPDATE: { ...JSON object... }
RULE_LINES: [ ...JSON array... ]
- FACT_UPDATE must be a JSON object and include the agent field (consistent with AGENT line).
- RULE_LINES must be a JSON array; each item must include: rule_line, agent; evidence is optional.
- If there are no valid rules, output RULE_LINES as an empty array []."""


def wrap_prompt(core_instructions: str, ctx_text: str, agent_tag: str = "") -> str:
    tag_line = f"AGENT: {agent_tag}" if agent_tag else "AGENT: <agent_tag>"
    parts = [
        GLOBAL_HEADER.strip(),
        "",
        core_instructions.strip(),
        "",
        "===== DATA =====",
        ctx_text.strip(),
        "",
        "===== OUTPUT FORMAT =====",
        tag_line,
        OUTPUT_PROTOCOL.strip(),
    ]
    return "\n".join(parts)
