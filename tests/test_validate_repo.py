import unittest

from scripts import validate_repo


class ValidateRepoTests(unittest.TestCase):
    def test_build_checks_matches_expected_order(self):
        checks = validate_repo.build_checks("python")

        self.assertEqual(
            [check["name"] for check in checks],
            ["unit_tests", "dataset_catalog", "notebooks", "lab_smoke"],
        )

    def test_build_checks_can_skip_lab_smoke_test(self):
        checks = validate_repo.build_checks("python", include_lab=False)

        self.assertEqual(
            [check["name"] for check in checks],
            ["unit_tests", "dataset_catalog", "notebooks"],
        )
