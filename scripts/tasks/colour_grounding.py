"""Optional colour curriculum, using only existing Micro-Transformer semantics/tokens.

Contrasting paint values are explicit; object identity, order and irrelevant
actions vary. Mix this family with the broad base profile, not in its place.
"""

from micro_transformer.data.task_registry import TaskSpec
from micro_transformer.data.task_families import Scenario

VARIANTS = ("single", "repaint", "distractor", "mixed", "same", "unknown")


def register(registry, api):
    def build(ctx, occurrence):
        variant, cycle = VARIANTS[occurrence % len(VARIANTS)], occurrence // len(VARIANTS)
        s = Scenario(api, ctx, "colour_grounding", variant)
        obj, actor = s.obj(), s.actor()
        colour = api.COLOURS[cycle % len(api.COLOURS)]
        different = s.rng.choice([c for c in api.COLOURS if c != colour])
        # Include truly minimal episodes AND longer equivalent forms. Rejection
        # of duplicate minimal forms can fall back to the richer forms.
        if s.rng.random() < .5:
            s.event(s.actor(), "move", *obj, "to", s.location())
        question = ["what", "colour", *obj]
        expected = [colour]
        if variant == "unknown":
            s.event(actor, "move", *obj, "to", s.location())
            other = s.obj()
            s.event(s.actor(), "paint", *other, colour)
            expected = ["unknown"]
        elif variant == "same":
            other = s.obj()
            truth = cycle % 2 == 0
            colour = api.COLOURS[cycle // 2 % len(api.COLOURS)]
            different = s.rng.choice([c for c in api.COLOURS if c != colour])
            s.event(actor, "paint", *obj, colour)
            s.event(s.actor(), "paint", *other, colour if truth else different)
            question = ["is", *obj, "same", "colour", *other]
            expected = ["yes" if truth else "no"]
        else:
            if variant == "repaint":
                s.event(s.actor(), "paint", *obj, different)
            other = s.obj() if variant == "distractor" else None
            target_last = bool(cycle // len(api.COLOURS) % 2)
            if other and target_last:
                s.event(s.actor(), "paint", *other, different)
            s.event(actor, "paint", *obj, colour)
            if other and not target_last:
                s.event(s.actor(), "paint", *other, different)
            if variant == "mixed":
                holder, receiver = actor, s.actor()
                s.event(holder, "take", *obj)
                s.event(holder, "give", *obj, "to", receiver)
                s.event(receiver, "drop", *obj, "at", s.location())
        return s.finish(question, expected)

    registry.register(TaskSpec("colour_grounding", "1.0.0",
                               "Minimal and contrasting colour grounding mixed with base skills",
                               VARIANTS, build, ("same",)))
