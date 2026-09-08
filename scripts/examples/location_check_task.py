"""Example extension: truth-balanced spatial checks without adding vocabulary.

Load explicitly with --task-module scripts/examples/location_check_task.py and
--profile-file scripts/examples/location_check_profile.json.
"""

from micro_transformer.data.task_registry import TaskSpec
from micro_transformer.data.task_families import Scenario


def register(registry, api):
    def build(ctx, occurrence):
        s = Scenario(api, ctx, "location_check", "contains")
        box, obj, actor = s.obj(api.CONTAINERS), s.obj(), s.actor()
        truth = occurrence % 2 == 0
        s.event(actor, "move", *box, "to", s.location())
        s.event(actor, "put", *obj, "inside", *box)
        if not truth:
            s.event(actor, "remove", *obj, "from", *box, "to", s.location())
        s.distract(2)
        return s.finish(["does", *box, "contains", *obj], ["yes" if truth else "no"])
    registry.register(TaskSpec("location_check", "1.0.0", "Example custom containment task",
                               ("contains",), build, ("contains",)))
