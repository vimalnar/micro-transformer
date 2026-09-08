"""Observable coverage, not a claim of exhaustive language understanding."""

from collections import Counter, defaultdict


class Coverage:
    def __init__(self):
        self.variants = defaultdict(Counter)
        self.answers = defaultdict(Counter)
        self.features = Counter()
        self.move_histogram = Counter()

    def add(self, record):
        task, variant = record["task"], record.get("variant", record["structural_template"])
        self.variants[task][variant] += 1
        answer = " ".join(record["answer_tokens"]) or "<statement>"
        self.answers[f"{task}/{variant}"][answer] += 1
        tokens = record["tokens"]
        if "?" not in tokens:
            return
        boundary = tokens.index("?")
        start = max((i + 1 for i, token in enumerate(tokens[:boundary]) if token == "."), default=0)
        # Imported lazily to avoid a generator/coverage initialization cycle.
        from .generator import AGENTS, KINDS, IDENTIFIERS
        question = tokens[start:boundary]
        objects = [(a, b) for a, b in zip(question, question[1:]) if a in KINDS and b in IDENTIFIERS]
        if not objects:
            return
        target = objects[0]
        statements, current = [], []
        for token in tokens[:start]:
            if token == ".":
                statements.append(current); current = []
            else:
                current.append(token)
        direct_moves = sum(len(s) >= 6 and s[0] in AGENTS and s[1] == "move" and tuple(s[2:4]) == target
                           and not any(c in s for c in ("before", "after", "and", "while")) for s in statements)
        self.move_histogram[direct_moves] += 1
        self.features["questions_with_repeated_direct_target_moves"] += direct_moves >= 2
        first = next(((a, b) for a, b in zip(tokens[:start], tokens[1:start]) if a in KINDS and b in IDENTIFIERS), None)
        self.features["questions_targeting_other_than_first_entity"] += first is not None and first != target

    def report(self, registry=None, selected_tasks=None):
        missing, unbalanced = [], []
        if registry:
            for task in (selected_tasks if selected_tasks is not None else registry.tasks):
                spec = registry.tasks[task]
                for variant in spec.variants:
                    if not self.variants[task][variant]: missing.append(f"{task}/{variant}")
                for variant in spec.boolean_variants:
                    counts = self.answers[f"{task}/{variant}"]
                    if not counts["yes"] or not counts["no"] or abs(counts["yes"] - counts["no"]) > 1:
                        unbalanced.append({"task_variant": f"{task}/{variant}", "yes": counts["yes"], "no": counts["no"]})
        return {"variant_counts": {k: dict(sorted(v.items())) for k, v in sorted(self.variants.items())},
                "answer_counts_by_variant": {k: dict(sorted(v.items())) for k, v in sorted(self.answers.items())},
                "features": dict(self.features),
                "queried_object_direct_move_histogram": dict(sorted(self.move_histogram.items())),
                "missing_variants": missing, "unbalanced_boolean_variants": unbalanced,
                "capability_coverage_passed": not missing and not unbalanced,
                "note": "Measured templates, answer balance and event features; not exhaustive semantic coverage."}
