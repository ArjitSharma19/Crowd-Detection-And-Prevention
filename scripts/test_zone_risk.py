import unittest
import numpy as np
import sys
import os

# Append the project workspace root to the python module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.density import get_zone_risk_scores


class TestZoneRiskScores(unittest.TestCase):
    def setUp(self):
        # We will assume a total capacity limit of 90, meaning each cell limit is 10
        self.max_people_allowed = 90
        # caution at 70% (7 people per cell), danger at 100% (10 people per cell)
        self.caution_threshold = 70.0

    def test_shared_base_logic_yolo(self):
        # Active model is YOLO, no speed/variance stats provided -> should rely purely on capacity
        zone_grid = np.zeros((3, 3), dtype=np.float32)
        # Set cell values to represent different capacity brackets
        zone_grid[0, 0] = 5.0   # 50% capacity -> Safe
        zone_grid[0, 1] = 8.0   # 80% capacity -> Caution
        zone_grid[0, 2] = 11.0  # 110% capacity -> Danger
        
        result = get_zone_risk_scores(
            zone_grid=zone_grid,
            active_model="YOLO",
            max_people_allowed=self.max_people_allowed,
            caution_threshold=self.caution_threshold,
            avg_speeds=None,
            dir_variances=None,
            previous_zone_grids=None
        )
        
        # Check coordinates and results
        self.assertEqual(result[0, 0]['risk_tier'], 'safe')
        self.assertEqual(result[0, 0]['trigger_reason'], 'capacity')
        self.assertAlmostEqual(result[0, 0]['capacity_pct'], 50.0)
        
        self.assertEqual(result[0, 1]['risk_tier'], 'caution')
        self.assertEqual(result[0, 1]['trigger_reason'], 'capacity')
        self.assertAlmostEqual(result[0, 1]['capacity_pct'], 80.0)
        
        self.assertEqual(result[0, 2]['risk_tier'], 'danger')
        self.assertEqual(result[0, 2]['trigger_reason'], 'capacity')
        self.assertAlmostEqual(result[0, 2]['capacity_pct'], 110.0)

    def test_shared_base_logic_csrnet(self):
        # Active model is CSRNet, no historical grids provided -> should rely on capacity
        zone_grid = np.zeros((3, 3), dtype=np.float32)
        zone_grid[1, 0] = 5.0   # 50% -> Safe
        zone_grid[1, 1] = 8.0   # 80% -> Caution
        zone_grid[1, 2] = 11.0  # 110% -> Danger
        
        result = get_zone_risk_scores(
            zone_grid=zone_grid,
            active_model="CSRNet",
            max_people_allowed=self.max_people_allowed,
            caution_threshold=self.caution_threshold,
            avg_speeds=None,
            dir_variances=None,
            previous_zone_grids=None
        )
        
        self.assertEqual(result[1, 0]['risk_tier'], 'safe')
        self.assertEqual(result[1, 1]['risk_tier'], 'caution')
        self.assertEqual(result[1, 2]['risk_tier'], 'danger')

    def test_yolo_velocity_elevation_chaotic(self):
        # Test chaotic elevation (+1 tier)
        zone_grid = np.zeros((3, 3), dtype=np.float32)
        zone_grid[0, 0] = 5.0  # Base capacity is 50% (Safe)
        
        avg_speeds = np.zeros((3, 3), dtype=np.float32)
        dir_variances = np.zeros((3, 3), dtype=np.float32)
        
        # High variance (0.6 > 0.4), but normal speed (30 < 50)
        avg_speeds[0, 0] = 30.0
        dir_variances[0, 0] = 0.6
        
        result = get_zone_risk_scores(
            zone_grid=zone_grid,
            active_model="YOLO",
            max_people_allowed=self.max_people_allowed,
            caution_threshold=self.caution_threshold,
            avg_speeds=avg_speeds,
            dir_variances=dir_variances,
            previous_zone_grids=None
        )
        
        # Base was Safe, elevated by 1 tier -> should be Caution
        self.assertEqual(result[0, 0]['risk_tier'], 'caution')
        self.assertEqual(result[0, 0]['trigger_reason'], 'velocity')
        self.assertAlmostEqual(result[0, 0]['velocity_variance'], 0.6)

    def test_yolo_velocity_elevation_critical(self):
        # Test chaotic + fast elevation (+2 tiers/direct to Danger)
        zone_grid = np.zeros((3, 3), dtype=np.float32)
        zone_grid[0, 0] = 3.0  # Base capacity is 30% (Safe)
        
        avg_speeds = np.zeros((3, 3), dtype=np.float32)
        dir_variances = np.zeros((3, 3), dtype=np.float32)
        
        # High variance (0.6 > 0.4) AND high speed (60 > 50)
        avg_speeds[0, 0] = 60.0
        dir_variances[0, 0] = 0.6
        
        result = get_zone_risk_scores(
            zone_grid=zone_grid,
            active_model="YOLO",
            max_people_allowed=self.max_people_allowed,
            caution_threshold=self.caution_threshold,
            avg_speeds=avg_speeds,
            dir_variances=dir_variances,
            previous_zone_grids=None
        )
        
        # Base was Safe, elevated to Danger directly
        self.assertEqual(result[0, 0]['risk_tier'], 'danger')
        self.assertEqual(result[0, 0]['trigger_reason'], 'velocity+speed')

    def test_csrnet_density_surge(self):
        # Test CSRNet surge elevation (+1 tier)
        # We need historical zone grids to calculate moving averages
        # Let's say caution threshold is 7.0 (70% of 10 limit)
        # Frame history: [d_t-3, d_t-2, d_t-1, d_t]
        # For a cell at (0, 0), let's simulate a rapid surge:
        g1 = np.ones((3, 3), dtype=np.float32) * 1.0  # t-3
        g2 = np.ones((3, 3), dtype=np.float32) * 1.0  # t-2
        g3 = np.ones((3, 3), dtype=np.float32) * 2.0  # t-1
        g4 = np.ones((3, 3), dtype=np.float32) * 5.0  # t (Current)
        
        # History list:
        history = [g1, g2, g3, g4]
        
        # Smoothed current: mean(g2, g3, g4) = mean(1.0, 2.0, 5.0) = 2.667
        # Smoothed previous: mean(g1, g2, g3) = mean(1.0, 1.0, 2.0) = 1.333
        # Growth rate = (2.667 - 1.333) / 1.333 = 1.000 (100% growth) -> > 20% DENSITY_SURGE_THRESHOLD
        
        # Current cell count is 5.0. Base capacity = 50% (Safe).
        # Surge detection should elevate it by 1 tier -> Caution.
        result = get_zone_risk_scores(
            zone_grid=g4,
            active_model="CSRNet",
            max_people_allowed=self.max_people_allowed,
            caution_threshold=self.caution_threshold,
            avg_speeds=None,
            dir_variances=None,
            previous_zone_grids=history
        )
        
        self.assertEqual(result[0, 0]['risk_tier'], 'caution')
        self.assertEqual(result[0, 0]['trigger_reason'], 'density_surge')


if __name__ == "__main__":
    unittest.main()
