from clay.penalties.edge_cross import segment_cross


class TestSegmentCross:
    def test_no_cross(self):
        assert not segment_cross(0, 0, 4, 0, 0, 1, 4, 1)

    def test_cross(self):
        assert segment_cross(0, 0, 4, 4, 0, 4, 4, 0)

    def test_touching_at_endpoint(self):
        assert not segment_cross(0, 0, 4, 4, 4, 4, 8, 8)

    def test_collinear_non_overlapping(self):
        assert not segment_cross(0, 0, 4, 4, 5, 5, 6, 6)

    def test_collinear_overlapping(self):
        assert not segment_cross(0, 0, 4, 4, 2, 2, 6, 6)