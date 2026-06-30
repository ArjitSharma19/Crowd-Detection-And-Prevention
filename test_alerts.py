"""
Unit tests for the AlertManager class within the Crowd Detection & Prevention system.
These tests verify risk tier evaluation (danger, caution, safe), capacity thresholds,
and instantaneous status transitions under different configurations.
"""

import unittest
import time
from src.alerts import AlertManager

class TestAlertManager(unittest.TestCase):
    """
    Test suite for AlertManager.
    Ensures that capacity, density, and delays are handled correctly to trigger the right alert levels.
    """
    def setUp(self):
        # Initialize AlertManager with standard limits matching mainapp config:
        # max_capacity=25, caution_at=70, density_limit=5.0, trigger_delay_seconds=20.0
        self.manager = AlertManager(
            max_capacity=25,
            caution_at=70,
            density_limit=5.0,
            trigger_delay_seconds=20.0
        )

    def test_risk_tier_capacity_danger(self):
        # capacity_percentage = 1320 must return "danger"
        self.assertEqual(self.manager.get_risk_tier(1320.0), "danger")
        # capacity_percentage = 100 must return "danger"
        self.assertEqual(self.manager.get_risk_tier(100.0), "danger")
        # capacity_percentage = 500 must return "danger"
        self.assertEqual(self.manager.get_risk_tier(500.0), "danger")

    def test_risk_tier_capacity_caution(self):
        # capacity_percentage = 70 must return "caution"
        self.assertEqual(self.manager.get_risk_tier(70.0), "caution")
        # capacity_percentage = 85 must return "caution"
        self.assertEqual(self.manager.get_risk_tier(85.0), "caution")

    def test_risk_tier_capacity_safe(self):
        # capacity_percentage = 69 must return "safe"
        self.assertEqual(self.manager.get_risk_tier(69.0), "safe")
        # capacity_percentage = 0 must return "safe"
        self.assertEqual(self.manager.get_risk_tier(0.0), "safe")

    def test_risk_tier_density_ignored(self):
        # peak_density must not affect risk tier; at 50% capacity it must be "safe"
        self.assertEqual(self.manager.get_risk_tier(50.0, peak_density=5.0), "safe")
        self.assertEqual(self.manager.get_risk_tier(50.0, peak_density=6.0), "safe")
        self.assertEqual(self.manager.get_risk_tier(50.0, peak_density=3.5), "safe")

    def test_instantaneous_update(self):
        # Initial status should be NORMAL
        self.assertEqual(self.manager.current_status, "NORMAL")
        
        # Set trigger delay to 0 to simulate instantaneous update
        self.manager.trigger_delay_seconds = 0.0
        
        # Call update with severe over-capacity (e.g. 330 people against limit of 25)
        # Should change status to CRITICAL instantaneously when trigger_delay_seconds is 0.0
        result = self.manager.update(current_count=330, peak_density=0.0)
        self.assertEqual(result["status"], "CRITICAL")
        self.assertEqual(self.manager.current_status, "CRITICAL")
        self.assertIn("CRITICAL: Capacity exceeded", result["message"])

        # Call update with caution-level capacity (e.g., 20 people against limit of 25 -> 80% capacity)
        # Should change status to WARNING instantaneously
        result = self.manager.update(current_count=20, peak_density=0.0)
        self.assertEqual(result["status"], "WARNING")
        self.assertEqual(self.manager.current_status, "WARNING")
        self.assertIn("Warning — capacity at 80%", result["message"])

        # Call update with safe-level capacity (e.g., 5 people against limit of 25 -> 20% capacity)
        # Should change status to NORMAL instantaneously
        result = self.manager.update(current_count=5, peak_density=0.0)
        self.assertEqual(result["status"], "NORMAL")
        self.assertEqual(self.manager.current_status, "NORMAL")
        self.assertIn("safe parameters", result["message"])

if __name__ == "__main__":
    unittest.main()
