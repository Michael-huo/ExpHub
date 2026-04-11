from __future__ import annotations

def run_image_gen(runtime):
    from . import frames_generate

    return frames_generate.run(runtime)


def run_sequence_merge(runtime):
    from . import sequence_merge

    return sequence_merge.run(runtime)


def run(runtime):
    run_image_gen(runtime)
    return run_sequence_merge(runtime)
