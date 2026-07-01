from __future__ import annotations

import unittest

from exphub.execution_plan import ExecutionPlanError, build_execution_plan


class ExecutionPlanTests(unittest.TestCase):
    def test_infer_omitted_step_resolves_all_without_viewer(self):
        plan = build_execution_plan(mode="infer", requested_step=None, seed=12345)
        self.assertIsNone(plan.requested_step)
        self.assertEqual(plan.resolved_step, "all")
        self.assertEqual(plan.stages, ("prepare", "encode", "decode", "eval"))
        self.assertFalse(plan.droid_live_viewer)

    def test_infer_explicit_all_disables_viewer(self):
        plan = build_execution_plan(mode="infer", requested_step="all", seed=12345)
        self.assertEqual(plan.requested_step, "all")
        self.assertEqual(plan.resolved_step, "all")
        self.assertFalse(plan.droid_live_viewer)

    def test_infer_explicit_eval_enables_viewer(self):
        plan = build_execution_plan(mode="infer", requested_step="eval", seed=12345)
        self.assertEqual(plan.requested_step, "eval")
        self.assertEqual(plan.resolved_step, "eval")
        self.assertEqual(plan.stages, ("eval",))
        self.assertTrue(plan.droid_live_viewer)

    def test_train_all_expands_to_train_pipeline(self):
        plan = build_execution_plan(mode="train", requested_step=None, seed=12345)
        self.assertEqual(plan.resolved_step, "all")
        self.assertEqual(plan.stages, ("prepare", "encode", "lora"))
        self.assertFalse(plan.droid_live_viewer)

    def test_invalid_mode_step_pairs_fail(self):
        cases = [
            ("train", "decode", "train + decode is invalid"),
            ("train", "eval", "train + eval is invalid"),
            ("infer", "lora", "infer + lora is invalid"),
        ]
        for mode, step, message in cases:
            with self.subTest(mode=mode, step=step):
                with self.assertRaises(ExecutionPlanError) as ctx:
                    build_execution_plan(mode=mode, requested_step=step, seed=12345)
                self.assertEqual(str(ctx.exception), message)

    def test_experiments_only_allowed_for_infer_all(self):
        plan = build_execution_plan(  
            mode="infer",
            requested_step=None,
            experiments=("motion-benchmark", "image-quality"),
            seed=12345,
        )
        self.assertEqual(plan.experiments, ("motion-benchmark", "image-quality"))

        with self.assertRaisesRegex(ExecutionPlanError, "only allowed for infer \\+ all"):
            build_execution_plan(mode="infer", requested_step="eval", experiments=("image-quality",), seed=12345)
        with self.assertRaisesRegex(ExecutionPlanError, "only allowed for infer \\+ all"):
            build_execution_plan(mode="train", requested_step=None, experiments=("image-quality",), seed=12345)

    def test_duplicate_experiments_fail(self):
        with self.assertRaisesRegex(ExecutionPlanError, "duplicate experiment: motion-benchmark"):
            build_execution_plan(
                mode="infer",
                requested_step=None,
                experiments=("motion-benchmark", "motion-benchmark"),
                seed=12345,
            )

    def test_negative_seed_fails(self):
        with self.assertRaisesRegex(ExecutionPlanError, "non-negative integer"):
            build_execution_plan(mode="infer", requested_step=None, seed=-1)


if __name__ == "__main__":
    unittest.main()
