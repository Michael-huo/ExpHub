from __future__ import annotations

from exphub.pipeline.decode.image_gen import core as image_gen_core
from exphub.pipeline.decode.sequence_merge import core as sequence_merge_core


def run_image_gen(runtime):
    return image_gen_core.run(runtime)


def run_sequence_merge(runtime):
    return sequence_merge_core.run(runtime)


def run(runtime):
    run_image_gen(runtime)
    return run_sequence_merge(runtime)
