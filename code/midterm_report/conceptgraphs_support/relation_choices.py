"""Versioned relationship inference contract; saved training prompts stay intact."""
import re

LABELS = ("a_on_b", "b_on_a", "a_in_b", "b_in_a", "next_to", "nearby", "none")
FORMAT = "multiple_choice_v3"
AUDITED_FORMATS = ("multiple_choice_v2", FORMAT)
UPSTREAM_RELATIONS = {"a_on_b": "a on b", "b_on_a": "b on a", "a_in_b": "a in b",
                      "b_in_a": "b in a", "next_to": "next to", "nearby": "nearby",
                      "none": "none of these"}
SWAPPED_LABELS = {"a_on_b": "b_on_a", "b_on_a": "a_on_b", "a_in_b": "b_in_a",
                  "b_in_a": "a_in_b", "next_to": "next_to", "nearby": "nearby", "none": "none"}


def edge_semantics(a, b, relation):
    """Arrows run from supported/contained subject to supporter/container."""
    label = "_".join(str(relation).casefold().split())
    if label in ("a_on_b", "a_in_b"):
        return a, b, "on" if label == "a_on_b" else "in", True
    if label in ("b_on_a", "b_in_a"):
        return b, a, "on" if label == "b_on_a" else "in", True
    if label in ("nearby", "next_to"):
        return a, b, label.replace("_", " "), False
    raise ValueError(f"Not a drawable relation: {relation!r}")


PROMPT = """Look at the two marked objects. Object A has a RED box; object B has a CYAN box.
Choose exactly ONE of these seven options:
1. a_on_b: A (RED) rests ON and is supported by B (CYAN).
2. b_on_a: B (CYAN) rests ON and is supported by A (RED).
3. a_in_b: A (RED) is physically contained INSIDE B (CYAN).
4. b_in_a: B (CYAN) is physically contained INSIDE A (RED).
5. next_to: A and B are immediate side-by-side neighbors with a small gap and no intervening object.
6. nearby: A and B are physically close, but none of options 1-5 applies.
7. none: None of the relationships above is supported by the image.
ON means physical support, not merely being above another object in the image. IN means containment, not overlapping boxes.
Prefer containment over support, support over next_to, and next_to over nearby when applicable.
For example, if the CYAN object rests on the RED object, choose b_on_a, not a_on_b.
Reply ONLY with the chosen option's exact label: a_on_b, b_on_a, a_in_b, b_in_a, next_to, nearby, or none.
Do not reply with just "on" or "in". Do not give an explanation."""


def parse_choice(answer):
    """Accept only a single unambiguous option; never invent missing direction."""
    text = str(answer).strip().casefold().strip(" \t\r\n.\"'`")
    canonical = re.sub(r"\s+", "_", text)
    if canonical in LABELS:
        return canonical
    match = re.fullmatch(r"(?:option\s*)?[([]?([1-7])[)\].]?", text)
    if match:
        return LABELS[int(match.group(1))-1]
    match = re.fullmatch(r"([1-7])[.):]\s*([a-z_]+)", text)
    if match and LABELS[int(match.group(1))-1] == match.group(2):
        return match.group(2)
    return None


class LabelTokenConstraint:
    """A generation trie allowing only complete canonical label continuations."""
    def __init__(self, tokenizer, labels, prompt_length, eos_ids, max_new_tokens):
        self.prompt_length, self.trie = prompt_length, {}
        self.eos_ids = [int(i) for i in (eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]) if i is not None]
        if not self.eos_ids:
            raise ValueError("Choice decoding requires an EOS token")
        for tokens in label_token_sequences(tokenizer, labels):
            if len(tokens)+1 > max_new_tokens:
                raise ValueError("max_new_tokens is too small for a complete choice plus EOS")
            node = self.trie
            for token in tokens:
                node = node.setdefault(int(token), {})
            for token in self.eos_ids:
                node.setdefault(token, {})

    def __call__(self, batch_id, input_ids):
        suffix = input_ids[self.prompt_length:]
        suffix = suffix.tolist() if hasattr(suffix, "tolist") else list(suffix)
        node = self.trie
        for token in suffix:
            if int(token) not in node:
                raise ValueError("Generated tokens left the allowed choice trie")
            node = node[int(token)]
        if not node:
            return self.eos_ids
        return list(node)


def label_token_sequences(tokenizer, labels):
    """Require lossless token round-trips before any constrained prediction."""
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Choice labels must be nonempty and unique")
    sequences = []
    for label in labels:
        tokens = tuple(tokenizer.encode(label, add_special_tokens=False))
        decoded = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not tokens or tokens in sequences or decoded != label:
            raise ValueError(f"Choice label cannot round-trip exactly through tokenizer: {label!r}")
        sequences.append(tokens)
    return sequences


def inference_settings(config):
    """Resolve the effective prompt and record the protocol explicitly."""
    config = dict(config)
    mode = config.get("relationship_response_format", FORMAT)
    if mode in (FORMAT, "multiple_choice", "multiple_choice_v1", "multiple_choice_v2"):
        config["relationship_response_format"] = FORMAT
        config["relation_prompt"] = config.get("relation_choice_prompt") or PROMPT
        config["relationship_choice_decoding"] = config.get("relationship_choice_decoding", "constrained_greedy")
        if config["relationship_choice_decoding"] not in ("sequence_score", "constrained_greedy"):
            raise ValueError("Unknown relationship_choice_decoding")
        experimental = config["relationship_choice_decoding"] == "sequence_score"
        config["relationship_protocol"] = "experimental_sequence_score_v1" if experimental else "canonical_label_generation_v1_restored"
        config["relationship_score_rule"] = "mean_log_probability_including_eos_v1" if experimental else None
        config["verify_swapped_boxes"] = config.get("verify_swapped_boxes", True)
        if not isinstance(config["verify_swapped_boxes"], bool):
            raise ValueError("verify_swapped_boxes must be a boolean")
    elif mode != "legacy":
        raise ValueError("relationship_response_format must be multiple_choice_v3 or legacy")
    return config
