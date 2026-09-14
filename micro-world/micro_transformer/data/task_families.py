"""Broad v4 tasks: compose executable events, then query the reference interpreter.

No expected answer is hard-coded into an episode. Builder expectations (for
balanced booleans) are asserted against the interpreter before serialization.
The v1 vocabulary and semantics remain unchanged.
"""

from .task_registry import TaskSpec


class Scenario:
    def __init__(self, api, ctx, task, variant):
        self.api, self.ctx, self.task, self.variant = api, ctx, task, variant
        self.rng = ctx.rng
        self.state = api.ReferenceInterpreter()
        self.events = []
        self.used = set()
        self.primary = None

    def obj(self, kinds=None):
        key, tokens = self.ctx.object(kinds or self.api.KINDS, self.used)
        self.used.add(key)
        if self.primary is None:
            self.primary = key
        return tokens

    def actor(self):
        return self.rng.choice(self.api.AGENTS)

    def location(self, excluding=None):
        return self.rng.choice([x for x in self.api.LOCATIONS if x != excluding])

    def event(self, *tokens):
        parts = list(tokens)
        self.state.execute(parts)
        self.events.append(parts)

    def distract(self, maximum=2):
        # Distractors can precede, interrupt, or follow target events. Objects
        # are distinct, but often share a kind or identifier with the target.
        for _ in range(self.rng.randint(0, maximum)):
            obj = self.obj()
            actor = self.actor()
            self.event(actor, "move", *obj, "to", self.location())
            if self.rng.random() < .35:
                self.event(actor, "paint", *obj, self.rng.choice(self.api.COLOURS))

    def updates(self, minimum=1):
        if self.ctx.suite == "challenge":
            return self.rng.randint(5, 7)
        upper = {"simple": 2, "mixed": 4, "hard": 6}[self.ctx.difficulty]
        return self.rng.randint(minimum, max(minimum, upper))

    def finish(self, question=(), expected=None):
        answer = self.state.answer(list(question))
        if expected is not None and answer != expected:
            raise ValueError(f"{self.task}/{self.variant}: expected {expected}, derived {answer}")
        tokens = [token for event in self.events for token in (*event, ".")]
        if question:
            tokens.extend([*question, "?", *answer])
        tokens.append("|")
        return self.api.Episode(tokens, answer, self.task, len(self.events),
                                f"{self.task}:{self.variant}", self.primary,
                                variant=self.variant, task_version="1.0.0", suite=self.ctx.suite,
                                _events=[event[:] for event in self.events], _question=list(question))


def state_task(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    s = Scenario(api, ctx, task, variant)
    obj, actor = s.obj(), s.actor()
    s.distract(2 if task == "delayed_recall" else 1)
    s.event(actor, "move", *obj, "to", s.location())
    if variant == "moves":
        for _ in range(s.updates() - 1):
            s.distract(1)
            previous = s.state.locations.get(":".join(obj))
            s.event(s.actor(), "move", *obj, "to", s.location(previous))
    elif variant == "take_drop":
        for _ in range(s.updates()):
            s.event(s.actor(), "take", *obj)
            s.event(s.actor(), "drop", *obj, "at", s.location())
    elif variant == "swap":
        other = s.obj()
        s.event(actor, "move", *other, "to", s.location())
        for _ in range(s.updates()):
            s.event(s.actor(), "swap", *obj, "with", *other)
    elif variant == "containment":
        box = s.obj(api.CONTAINERS)
        s.event(actor, "put", *obj, "inside", *box)
        if occurrence // len(variants) % 2:
            s.event(actor, "remove", *obj, "from", *box, "to", s.location())
    elif variant == "unknown":
        if occurrence // len(variants) % 2:
            s.event(actor, "take", *obj)
        else:
            obj = s.obj()
    s.distract(6 if task == "delayed_recall" else 2)
    return s.finish(["where", *obj])


def ownership(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    s = Scenario(api, ctx, task, variant)
    obj, actor = s.obj(), s.actor()
    s.distract(1)
    s.event(actor, "move", *obj, "to", s.location())
    if variant != "unowned":
        # The taker need not be the mover. No transfer is required in 'take'.
        holder = s.actor()
        s.event(holder, "take", *obj)
        if variant == "transfers":
            for _ in range(s.updates()):
                receiver = s.rng.choice([a for a in api.AGENTS if a != holder])
                s.event(holder, "give", *obj, "to", receiver)
                holder = receiver
                s.distract(1)
        elif variant == "drop_retake":
            s.event(holder, "drop", *obj, "at", s.location())
            if occurrence // len(variants) % 2:
                s.event(s.actor(), "take", *obj)
        elif variant == "relocate":
            s.event(s.actor(), "move", *obj, "to", s.location())
    s.distract(2)
    return s.finish(["who", "has", *obj])


def properties(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    cycle = occurrence // len(variants)
    truth = cycle % 2 == 0
    s = Scenario(api, ctx, task, variant)
    obj, actor = s.obj(), s.actor()
    s.distract(1)
    s.event(actor, "move", *obj, "to", s.location())
    if variant == "colour":
        for _ in range(s.updates()):
            s.event(s.actor(), "paint", *obj, s.rng.choice(api.COLOURS))
            s.distract(1)
        return s.finish(["what", "colour", *obj])
    if variant == "unknown_colour":
        s.distract(2)
        return s.finish(["what", "colour", *obj])
    if variant == "same_colour":
        other = s.obj()
        first = s.rng.choice(api.COLOURS)
        second = first if truth else s.rng.choice([c for c in api.COLOURS if c != first])
        s.event(actor, "paint", *obj, first)
        s.event(actor, "paint", *other, second)
        question = ["is", *obj, "same", "colour", *other]
    else:
        attribute = api.ATTRIBUTES[(cycle // 2) % len(api.ATTRIBUTES)]
        negated = variant == "negated_attribute"
        value = truth != negated
        if attribute == "visible":
            if not value:
                box = s.obj(api.CONTAINERS)
                s.event(actor, "hide", *obj, "in", *box)
        elif attribute in {"locked", "closed"}:
            yes, no = ("lock", "unlock") if attribute == "locked" else ("close", "open")
            s.event(actor, yes if not value else no, *obj)
            s.event(actor, yes if value else no, *obj)
        elif attribute == "hidden":
            box = s.obj(api.CONTAINERS)
            s.event(actor, "hide", *obj, "in", *box)
            if not value:
                s.event(actor, "find", *obj)
        elif value:
            s.event(*obj, "is", attribute)
        else:
            other = s.obj()
            s.event(*other, "is", attribute)
        question = ["is", *obj, *(["not"] if negated else []), attribute]
    s.distract(2)
    return s.finish(question, ["yes" if truth else "no"])


def conditionals(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    cycle = occurrence // len(variants)
    truth, negate = cycle % 2 == 0, variant == "negated"
    s = Scenario(api, ctx, task, variant)
    obj, actor = s.obj(("door", "gate", "box")), s.actor()
    s.distract(1)
    s.event(actor, "move", *obj, "to", s.location())
    locked = bool(s.rng.getrandbits(1))
    s.event(actor, "lock" if locked else "unlock", *obj)
    branch_true = locked != negate
    # Randomize the condition independently of the desired result, so neither
    # branch position nor condition polarity predicts the answer.
    chosen = "close" if truth else "open"
    opposite = "open" if truth else "close"
    left, right = (chosen, opposite) if branch_true else (opposite, chosen)
    s.event("if", *obj, *(["not"] if negate else []), "locked", "then", actor, left, *obj,
            "else", actor, right, *obj)
    s.distract(2)
    return s.finish(["is", *obj, "closed"], ["yes" if truth else "no"])


def epistemic(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    s = Scenario(api, ctx, task, variant)
    obj = s.obj()
    mover, observer = s.rng.sample(api.AGENTS, 2)
    s.distract(1)
    first, second = s.rng.sample(api.LOCATIONS, 2)
    s.event(mover, "move", *obj, "to", first)
    if task == "belief":
        if variant != "unknown":
            s.event(observer, "sees", mover, "move", *obj, "to", first)
        if variant in {"stale", "updated"}:
            for _ in range(s.updates()):
                second = s.location(s.state.locations.get(":".join(obj)))
                s.event(mover, "move", *obj, "to", second)
                if variant == "updated":
                    s.event(observer, "sees", mover, "move", *obj, "to", second)
        question = ["where", observer, "believes", *obj]
    else:
        if variant != "unknown":
            s.event(mover, "tell", observer, *obj, "at", first)
        if variant in {"stale", "updated"}:
            s.event(mover, "move", *obj, "to", second)
            if variant == "updated":
                s.event(mover, "tell", observer, *obj, "at", second)
        if variant == "explicit":
            s.event(observer, "knows", *obj, "at", second)
        question = ["where", observer, "knows", *obj]
    s.distract(2)
    return s.finish(question)


def spatial(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    truth = occurrence // len(variants) % 2 == 0
    s = Scenario(api, ctx, task, variant)
    first = s.obj(api.CONTAINERS if variant == "contains" else None)
    second, actor = s.obj(), s.actor()
    s.distract(1)
    s.event(actor, "move", *first, "to", s.location())
    if variant == "contains":
        s.event(actor, "put", *second, "inside", *first)
        if not truth:
            s.event(actor, "remove", *second, "from", *first, "to", s.location())
        question = ["does", *first, "contains", *second]
    else:
        relation = "behind" if variant == "which" else variant
        s.event(*first, relation if truth else ("beside" if relation != "beside" else "behind"), *second)
        question = (["which", *first, "is", relation, *second] if variant == "which"
                    else ["does", *first, relation, *second])
    s.distract(2)
    expected = (first if truth else ["unknown"]) if variant == "which" else ["yes" if truth else "no"]
    return s.finish(question, expected)


def quantifiers(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    truth = occurrence // len(variants) % 2 == 0
    s = Scenario(api, ctx, task, variant)
    first = s.obj()
    kind = first[0]
    # Every held-out kind has at least one available entity. 'all' can be false
    # even for a single known member. The identity split still limits the range
    # of possible counts; report that limitation separately from answer balance.
    available = [i for i in api.IDENTIFIERS if api.assigned_split(f"{kind}:{i}") == ctx.split]
    number = s.rng.randint(1, min(5, len(available)))
    objects = [first] + [s.obj((kind,)) for _ in range(number - 1)]
    location, other = s.rng.sample(api.LOCATIONS, 2)
    if variant == "count":
        matches = s.rng.randint(0, number)
    elif variant == "some":
        matches = s.rng.randint(1, number) if truth else 0
    elif variant == "none":
        matches = 0 if truth else s.rng.randint(1, number)
    else:
        matches = number if truth else s.rng.randint(0, number - 1)
    s.rng.shuffle(objects)
    for index, obj in enumerate(objects):
        s.event(s.actor(), "move", *obj, "to", location if index < matches else other)
    # Query the final arrangement, including a correction/reversal when possible.
    if s.rng.random() < .5:
        obj = objects[0]
        final = s.state.locations[":".join(obj)]
        s.event(s.actor(), "move", *obj, "to", other if final == location else location)
        s.event(s.actor(), "move", *obj, "to", final)
    question = ["count", kind, "at", location] if variant == "count" else ["is", variant, kind, "at", location]
    return s.finish(question, None if variant == "count" else ["yes" if truth else "no"])


def temporal(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    s = Scenario(api, ctx, task, variant)
    obj, actor = s.obj(), s.actor()
    s.distract(1)
    first, second = s.rng.sample(api.LOCATIONS, 2)
    if variant in {"before", "after", "and", "while"}:
        # Same-target cases make event ordering consequential, rather than two
        # unrelated actions. Also retain independent-object examples.
        other = obj if occurrence // len(variants) % 2 else s.obj()
        s.event(actor, "move", *obj, "to", first, variant, s.actor(), "move", *other, "to", second)
    elif variant == "again":
        s.event(actor, "move", *obj, "to", first)
        s.event(actor, "move", *obj, "from", first, "to", second, "again")
    elif variant == "only":
        s.event("only", actor, "move", *obj, "to", first)
    else:
        other = s.obj()
        s.event(actor, "take", *other)
        s.event(actor, "move", *obj, "to", first, "because", actor, "has", *other)
    s.distract(2)
    return s.finish(["where", *obj])


def abilities(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    truth = occurrence // len(variants) % 2 == 0
    s = Scenario(api, ctx, task, variant)
    obj, actor = s.obj(("door", "gate", "box")), s.actor()
    s.distract(1)
    s.event(actor, "move", *obj, "to", s.location())
    if variant == "open":
        for _ in range(s.updates() - 1):
            s.event(actor, s.rng.choice(("lock", "unlock")), *obj)
        s.event(actor, "unlock" if truth else "lock", *obj)
        question = ["can", s.actor(), "open", *obj]
    else:
        box = s.obj(api.CONTAINERS)
        s.event(actor, "hide", *obj, "in", *box)
        if truth:
            s.event(actor, "find", *obj)
        question = ["is", *obj, "visible"]
    s.distract(2)
    return s.finish(question, ["yes" if truth else "no"])


def composition(api, ctx, occurrence, task, variants):
    variant = variants[occurrence % len(variants)]
    s = Scenario(api, ctx, task, variant)
    obj, actor = s.obj(), s.actor()
    s.distract(1)
    s.event(actor, "build", *obj, "at", s.location())
    s.event(actor, "paint", *obj, s.rng.choice(api.COLOURS))
    holder = None
    for _ in range(s.updates(2)):
        action = s.rng.choice(("move", "paint", "take", "drop", "give"))
        if action == "paint":
            s.event(s.actor(), "paint", *obj, s.rng.choice(api.COLOURS))
        elif action in {"take", "give"}:
            if action == "give" and holder:
                receiver = s.actor()
                s.event(holder, "give", *obj, "to", receiver)
                holder = receiver
            else:
                holder = s.actor()
                s.event(holder, "take", *obj)
        else:
            s.event(s.actor(), action, *obj, "to" if action == "move" else "at", s.location())
            holder = None
        s.distract(1)
    question = {"location": ["where", *obj], "owner": ["who", "has", *obj],
                "colour": ["what", "colour", *obj]}[variant]
    return s.finish(question)


def statements(api, ctx, occurrence, task, variants):
    s = Scenario(api, ctx, task, "events")
    # Preserve the full action/word inventory in statement-only examples.
    obj, actor = s.obj(), s.actor()
    s.event(actor, "build", *obj, "at", s.location())
    s.event(actor, "follow", s.actor(), "to", s.location())
    box = s.obj(api.CONTAINERS)
    s.event(actor, "hide", *obj, "in", *box)
    s.event(*box, "contains", *obj)
    s.event(actor, "find", *obj)
    s.event(actor, "remove", *obj, "from", *box, "to", s.location())
    return s.finish()


FAMILIES = (
    ("statement_only", statements, ("events",), (), "Executable event sequences without a question"),
    ("state_tracking", state_task, ("moves", "take_drop", "swap", "containment", "unknown"), (), "Current location after updates and removals"),
    ("delayed_recall", state_task, ("moves", "take_drop", "swap"), (), "Updated target state amid intervening events"),
    ("ownership", ownership, ("take", "transfers", "drop_retake", "relocate", "unowned"), (), "Ownership with zero or multiple transfers"),
    ("property", properties, ("colour", "attribute", "negated_attribute", "same_colour", "unknown_colour"), ("attribute", "negated_attribute", "same_colour"), "Colour and property updates, negation and missing facts"),
    ("conditional", conditionals, ("positive", "negated"), ("positive", "negated"), "Independent branch conditions and resulting states"),
    ("belief", epistemic, ("current", "stale", "updated", "unknown"), (), "Observed, stale, updated and absent observations"),
    ("spatial", spatial, ("behind", "beside", "with", "contains", "which"), ("behind", "beside", "with", "contains"), "Positive and negative relations and containment changes"),
    ("quantifier", quantifiers, ("count", "some", "none", "all"), ("some", "none", "all"), "Counts including zero and balanced quantifier statements"),
    ("temporal", temporal, ("before", "after", "and", "while", "again", "only", "because"), (), "Order-sensitive same-target and independent events"),
    ("communication", epistemic, ("current", "stale", "updated", "unknown", "explicit"), (), "Information that may be absent, changed or stale"),
    ("ability", abilities, ("open", "visible"), ("open", "visible"), "Balanced action eligibility and visibility after updates"),
    ("composition", composition, ("location", "owner", "colour"), (), "Mixed actions followed by different query types"),
)


def register(registry, api):
    for name, function, variants, boolean, description in FAMILIES:
        def build(ctx, occurrence, name=name, function=function, variants=variants):
            return function(api, ctx, occurrence, name, variants)
        registry.register(TaskSpec(name, "1.0.0", description, variants, build, boolean))
