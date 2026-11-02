"""Explicit relationship choices for inference; saved training prompts stay intact."""
import re

LABELS = ("a_on_b", "b_on_a", "a_in_b", "b_in_a", "next_to", "nearby", "none")
FORMAT = "multiple_choice_v1"
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
        sequences = set()
        for label in labels:
            tokens = tuple(tokenizer.encode(label, add_special_tokens=False))
            if not tokens or tokens in sequences:
                raise ValueError("Choice labels must have distinct nonempty token sequences")
            if len(tokens)+1 > max_new_tokens:
                raise ValueError("max_new_tokens is too small for a complete choice plus EOS")
            sequences.add(tokens)
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


def inference_settings(config):
    """Resolve the effective prompt and record the protocol explicitly."""
    config = dict(config)
    mode = config.get("relationship_response_format", FORMAT)
    if mode in (FORMAT, "multiple_choice"):
        config["relationship_response_format"] = FORMAT
        config["relation_prompt"] = config.get("relation_choice_prompt") or PROMPT
    elif mode != "legacy":
        raise ValueError("relationship_response_format must be multiple_choice_v1 or legacy")
    return config
