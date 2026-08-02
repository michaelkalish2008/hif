"""Unit tests for the region-sensitivity rank computation (harness helpers)."""

import pytest

from hif.validation.harness import compute_answer_cell_rank, scale_cell


class TestComputeAnswerCellRank:
    def test_rank_1_when_answer_cell_has_highest_jsd(self):
        cell_jsd = {"0,0": 0.01, "0,1": 0.42, "1,0": 0.03, "1,1": 0.02}
        assert compute_answer_cell_rank(cell_jsd, {"row": 0, "col": 1}) == 1

    def test_rank_last_when_answer_cell_has_lowest_jsd(self):
        cell_jsd = {"0,0": 0.30, "0,1": 0.20, "1,0": 0.10, "1,1": 0.05}
        assert compute_answer_cell_rank(cell_jsd, {"row": 1, "col": 1}) == 4

    def test_middle_rank_4x4(self):
        cell_jsd = {f"{r},{c}": (r * 4 + c) / 100.0
                    for r in range(4) for c in range(4)}
        # value 0.12 for (3,0): ranks below 0.15, 0.14, 0.13 -> rank 4 of 16
        assert compute_answer_cell_rank(cell_jsd, {"row": 3, "col": 0}) == 4

    def test_missing_answer_cell_raises(self):
        with pytest.raises(KeyError):
            compute_answer_cell_rank({"0,0": 0.1}, {"row": 3, "col": 3})

    def test_tied_values_still_return_a_valid_rank(self):
        cell_jsd = {"0,0": 0.2, "0,1": 0.2, "1,0": 0.2, "1,1": 0.2}
        rank = compute_answer_cell_rank(cell_jsd, {"row": 1, "col": 0})
        assert 1 <= rank <= 4


class TestScaleCell:
    def test_identity_when_grids_match(self):
        cell = {"row": 3, "col": 1}
        assert scale_cell(cell, {"rows": 4, "cols": 4}, (4, 4)) == cell

    def test_4x4_to_2x2(self):
        g = {"rows": 4, "cols": 4}
        assert scale_cell({"row": 0, "col": 0}, g, (2, 2)) == {"row": 0, "col": 0}
        assert scale_cell({"row": 1, "col": 2}, g, (2, 2)) == {"row": 0, "col": 1}
        assert scale_cell({"row": 3, "col": 3}, g, (2, 2)) == {"row": 1, "col": 1}

    def test_missing_corpus_grid_assumes_harness_grid(self):
        assert scale_cell({"row": 2, "col": 1}, None, (4, 4)) == {"row": 2, "col": 1}
