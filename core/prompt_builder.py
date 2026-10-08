from typing import List, Dict


def build_code_gen_prompt(rules_text: str) -> str:
    prompt = [
        "You translate constraints into JSON config and must use only the input rules.",
        "Output must be strict JSON and pass schema validation.",
        "",
        "Label parsing:",
        "- base_class = last token with trailing digits removed (apple2->apple; red wire1->wire).",
        "- subtype = remaining prefix tokens joined; may be empty (red wire1->red).",
        "- Use base_class/subtype only; do not use numeric suffixes.",
        "",
        "JSON output template:",
        "{",
        "  \"derived\": {},",
        "  \"rules\": []",
        "}",
        "derived must be an empty object; no derived features are supported.",
        "",
        "Allowed primitives (field names must match exactly; no null; id is unique):",
        "A) count: type,id,container,target,(subtype),(exact|min|max),(filter)",
        "   container=whole_image or {\"inside\":\"<base_class>\"}",
        "   filter fields only diag,subtype,base_class; ops ==,!=,<,<=,>,>=,&&,||,!.",
        "B) imply: type,id,if,then (if/then are arrays of rules)",
        "Output rules: JSON only; no pseudocode/placeholders; no new fields/types/relations; numbers must come from input rules.",
        "",
        "Input Rules:",
        rules_text,
    ]
    return "\n".join(prompt)


def build_rule_translate_prompt(
    rule_lines: List[str],
    agent_tag: str,
    support_lines: List[str] | None = None,
) -> str:
    rules_block = "\n".join(f"- {line}" for line in rule_lines) if rule_lines else "- (none)"
    support_block = ""
    if support_lines:
        support_block = "\n\nSupporting rules (read-only):\n" + "\n".join(
            f"- {line}" for line in support_lines
        )

    input_text = (
        f"Rules for translation (agent={agent_tag}):\n{rules_block}{support_block}"
    )

    header = [
        "You are translating RuleMemory lines into executable JSON rules.",
        f"Current agent: {agent_tag}. Only translate the Rules for translation section.",
        "Supporting rules are read-only hints for numbers/edges; do NOT translate them.",
        f"All rule ids must start with '{agent_tag}_' and be unique.",
        "Skip any rule line that is a failure/negation or cannot be expressed by the schema.",
    ]

    if agent_tag.startswith("agent1"):
        header += [
            "Agent1 hints:",
            "- 'each sample must have N <base_class>' -> count(exact=N, container=whole_image, target=<base_class>).",
            "- 'each sample must contain <base_class>' -> count(min=1, container=whole_image, target=<base_class>).",
            "- 'each sample has N A or M B or K C' -> translate OR using imply + count(min/max).",
            "  For two options (A,N) or (B,M):",
            "  - if count(max=N-1,target=A) then count(exact=M,target=B)",
            "  - if count(min=N+1,target=A) then count(exact=M,target=B)",
            "  For 3+ options, pick the last option as fallback and emit imply rules for all",
            "  combinations of (<=N-1)/(>=N+1) on the other options, with THEN enforcing",
            "  the fallback exact count.",
        ]
    return "\n".join(header) + "\n\n" + build_code_gen_prompt(input_text)


def build_llm_contents(prompt_text: str):
    """
    Wrap chat-completions content. Current version sends text only (no images).
    """
    return [{"type": "text", "text": prompt_text}]
