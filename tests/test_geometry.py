from clay.geometry import point_to_segment_distance


class TestPointToSegmentDistance:
    def test_perpendicular_inside(self):
        dist = point_to_segment_distance(0, 0, 4, 0, 2, 3)
        assert abs(dist - 3) < 1e-6

    def test_perpendicular_outside(self):
        dist = point_to_segment_distance(0, 0, 4, 0, -1, 3)
        expected = ((-1)**2 + 3**2) ** 0.5
        assert abs(dist - expected) < 1e-6

    def test_closest_to_start(self):
        dist = point_to_segment_distance(0, 0, 4, 0, -1, -1)
        expected = (1**2 + 1**2) ** 0.5
        assert abs(dist - expected) < 1e-6

    def test_closest_to_end(self):
        dist = point_to_segment_distance(0, 0, 4, 0, 5, -1)
        expected = (1**2 + 1**2) ** 0.5
        assert abs(dist - expected) < 1e-6
