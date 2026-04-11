from __future__ import annotations


class PromptBackend(object):
    name = ""

    def load(self):
        # type: () -> None
        raise NotImplementedError

    def generate(self, image_paths, instruction):
        # type: (list, str) -> str
        raise NotImplementedError

    def meta(self):
        # type: () -> dict
        return {}
