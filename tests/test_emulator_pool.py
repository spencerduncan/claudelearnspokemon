"""
Test suite for EmulatorPool basic functionality

Tests core emulator pool operations as specified in the design document.
"""

import time
import unittest
from unittest.mock import MagicMock

from claudelearnspokemon.emulator_pool import EmulatorInstance, EmulatorPool, EmulatorState


class TestEmulatorPoolBasic(unittest.TestCase):
    """Test basic EmulatorPool functionality"""

    def setUp(self):
        """Set up test environment"""
        self.pool = EmulatorPool(pool_size=4, default_timeout=2.0)

    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self.pool, "_initialized") and self.pool._initialized:
            self.pool.shutdown()

    def test_emulator_pool_starts_containers_on_sequential_ports(self):
        """Test pool initializes containers on sequential ports"""
        self.pool.initialize(pool_size=4)

        # Verify initialization state
        self.assertTrue(self.pool._initialized)
        self.assertEqual(len(self.pool._emulators), 4)

        # Verify sequential ports starting from 8081
        expected_ports = [8081, 8082, 8083, 8084]
        actual_ports = sorted(self.pool._emulators.keys())
        self.assertEqual(actual_ports, expected_ports)

        # Verify all emulators are available
        for port, emulator in self.pool._emulators.items():
            self.assertEqual(emulator.state, EmulatorState.AVAILABLE)
            self.assertIsNone(emulator.owner_thread_id)
            self.assertIn(port, self.pool._available_ports)

    def test_emulator_pool_tracks_emulator_availability(self):
        """Test pool correctly tracks which emulators are available/busy"""
        self.pool.initialize()

        # Initially all should be available
        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 4)
        self.assertEqual(status["busy_count"], 0)

        # Acquire one emulator
        client1 = self.pool.acquire(timeout=1.0)
        self.assertIsNotNone(client1)

        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 3)
        self.assertEqual(status["busy_count"], 1)

        # Acquire another
        client2 = self.pool.acquire(timeout=1.0)
        self.assertIsNotNone(client2)

        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 2)
        self.assertEqual(status["busy_count"], 2)

        # Release one
        self.pool.release(client1)

        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 3)
        self.assertEqual(status["busy_count"], 1)

        # Release the other
        self.pool.release(client2)

        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 4)
        self.assertEqual(status["busy_count"], 0)

    def test_emulator_pool_blocks_acquisition_when_all_busy(self):
        """Test pool blocks acquisition when all emulators are busy"""
        self.pool.initialize()

        # Acquire all emulators
        clients = []
        for i in range(4):
            client = self.pool.acquire(timeout=1.0)
            self.assertIsNotNone(client, f"Should acquire emulator {i}")
            clients.append(client)

        # Verify all are busy
        status = self.pool.get_status()
        self.assertEqual(status["busy_count"], 4)
        self.assertEqual(status["available_count"], 0)

        # Try to acquire one more - should timeout
        start_time = time.time()
        blocked_client = self.pool.acquire(timeout=0.5)
        elapsed = time.time() - start_time

        self.assertIsNone(blocked_client)
        self.assertAlmostEqual(elapsed, 0.5, delta=0.2)

        # Release all
        for client in clients:
            self.pool.release(client)

    def test_emulator_pool_restarts_failed_emulator_automatically(self):
        """Test pool can restart failed emulator instances"""
        self.pool.initialize()

        # Get initial status
        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 4)

        # Simulate failure by manually setting emulator state
        test_port = 8081
        self.pool._emulators[test_port].state = EmulatorState.FAILED
        self.pool._available_ports.discard(test_port)

        # Verify failed state
        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 3)

        # Restart the failed emulator
        restart_success = self.pool.restart_emulator(test_port)
        self.assertTrue(restart_success)

        # Verify it's available again
        status = self.pool.get_status()
        self.assertEqual(status["available_count"], 4)

        emulator = self.pool._emulators[test_port]
        self.assertEqual(emulator.state, EmulatorState.AVAILABLE)
        self.assertIn(test_port, self.pool._available_ports)

    def test_emulator_pool_maintains_checkpoint_isolation_between_instances(self):
        """Test emulators maintain checkpoint isolation"""
        self.pool.initialize()

        # Acquire two emulators
        client1 = self.pool.acquire()
        client2 = self.pool.acquire()

        self.assertIsNotNone(client1)
        self.assertIsNotNone(client2)

        # Mock script and checkpoint objects
        mock_script1 = MagicMock()
        mock_script1.id = "script_1"

        mock_script2 = MagicMock()
        mock_script2.id = "script_2"

        # Execute scripts on different emulators with different checkpoints
        result1 = self.pool.execute_script(client1, mock_script1, "checkpoint_A")
        result2 = self.pool.execute_script(client2, mock_script2, "checkpoint_B")

        # Verify isolation - different ports and checkpoints
        self.assertNotEqual(result1["port"], result2["port"])
        self.assertEqual(result1["checkpoint_id"], "checkpoint_A")
        self.assertEqual(result2["checkpoint_id"], "checkpoint_B")

        # Clean up
        self.pool.release(client1)
        self.pool.release(client2)

    def test_emulator_pool_gracefully_shuts_down_all_containers(self):
        """Test pool gracefully shuts down all containers"""
        self.pool.initialize()

        # Acquire some emulators
        client1 = self.pool.acquire()
        client2 = self.pool.acquire()

        self.assertIsNotNone(client1)
        self.assertIsNotNone(client2)

        # Get status before shutdown
        status = self.pool.get_status()
        self.assertEqual(status["busy_count"], 2)
        self.assertEqual(status["available_count"], 2)

        # Shutdown
        self.pool.shutdown()

        # Verify shutdown state
        self.assertTrue(self.pool._shutdown)
        self.assertEqual(len(self.pool._emulators), 0)
        self.assertEqual(len(self.pool._available_ports), 0)
        self.assertEqual(len(self.pool._busy_ports), 0)

        # Verify new acquisitions fail
        client = self.pool.acquire(timeout=0.5)
        self.assertIsNone(client)

    def test_emulator_pool_health_check(self):
        """Test health check functionality"""
        self.pool.initialize()

        # All should be healthy initially
        health = self.pool.health_check()
        expected_ports = [8081, 8082, 8083, 8084]

        self.assertEqual(set(health.keys()), set(expected_ports))
        for port, is_healthy in health.items():
            self.assertTrue(is_healthy, f"Port {port} should be healthy")

        # Simulate failure
        self.pool._emulators[8081].state = EmulatorState.FAILED

        health = self.pool.health_check()
        self.assertFalse(health[8081], "Failed emulator should be unhealthy")
        self.assertTrue(health[8082], "Other emulators should be healthy")

    def test_emulator_pool_restart_emulator_conditions(self):
        """Test conditions for emulator restart"""
        self.pool.initialize()

        # Should not restart non-existent emulator
        success = self.pool.restart_emulator(9999)
        self.assertFalse(success)

        # Should not restart busy emulator
        client = self.pool.acquire()
        port = None
        for p, emulator in self.pool._emulators.items():
            if emulator.client == client:
                port = p
                break

        self.assertIsNotNone(port)
        success = self.pool.restart_emulator(port)
        self.assertFalse(success)

        # Should restart available emulator
        self.pool.release(client)
        success = self.pool.restart_emulator(port)
        self.assertTrue(success)

    def test_emulator_pool_execute_script_ownership_validation(self):
        """Test script execution validates client ownership"""
        self.pool.initialize()

        # Acquire emulator in one "thread" context
        client = self.pool.acquire()
        self.assertIsNotNone(client)

        # Mock script
        mock_script = MagicMock()
        mock_script.id = "test_script"

        # Should succeed with correct owner
        result = self.pool.execute_script(client, mock_script, "checkpoint_1")
        self.assertTrue(result["success"])

        # Clean up
        self.pool.release(client)


class TestEmulatorInstance(unittest.TestCase):
    """Test EmulatorInstance data class"""

    def test_emulator_instance_creation(self):
        """Test EmulatorInstance initialization"""
        emulator = EmulatorInstance(port=8081)

        self.assertEqual(emulator.port, 8081)
        self.assertIsNone(emulator.client)
        self.assertEqual(emulator.state, EmulatorState.AVAILABLE)
        self.assertIsNone(emulator.owner_thread_id)
        self.assertIsNone(emulator.acquired_at)
        self.assertTrue(emulator.is_available())

    def test_emulator_instance_availability_check(self):
        """Test availability check logic"""
        emulator = EmulatorInstance(port=8081)

        # Initially available
        self.assertTrue(emulator.is_available())

        # Not available when busy
        emulator.state = EmulatorState.BUSY
        self.assertFalse(emulator.is_available())

        # Not available when owned
        emulator.state = EmulatorState.AVAILABLE
        emulator.owner_thread_id = 12345
        self.assertFalse(emulator.is_available())

        # Not available when failed
        emulator.owner_thread_id = None
        emulator.state = EmulatorState.FAILED
        self.assertFalse(emulator.is_available())

        # Available when state is available and no owner
        emulator.state = EmulatorState.AVAILABLE
        emulator.owner_thread_id = None
        self.assertTrue(emulator.is_available())


class TestEmulatorPoolConfiguration(unittest.TestCase):
    """Test EmulatorPool configuration and edge cases"""

    def test_custom_pool_size_and_timeout(self):
        """Test custom pool size and timeout configuration"""
        pool = EmulatorPool(pool_size=2, default_timeout=1.5)

        self.assertEqual(pool.pool_size, 2)
        self.assertEqual(pool.default_timeout, 1.5)

        pool.initialize()

        # Should have 2 emulators
        self.assertEqual(len(pool._emulators), 2)
        expected_ports = [8081, 8082]
        actual_ports = sorted(pool._emulators.keys())
        self.assertEqual(actual_ports, expected_ports)

        pool.shutdown()

    def test_double_initialization_handling(self):
        """Test handling of double initialization"""
        pool = EmulatorPool(pool_size=2)

        # First initialization
        pool.initialize()
        self.assertTrue(pool._initialized)
        original_emulator_count = len(pool._emulators)

        # Second initialization should be harmless
        pool.initialize()
        self.assertTrue(pool._initialized)
        self.assertEqual(len(pool._emulators), original_emulator_count)

        pool.shutdown()

    def test_acquisition_before_initialization(self):
        """Test acquisition behavior before pool is initialized"""
        pool = EmulatorPool(pool_size=2)

        # Should wait for initialization
        start_time = time.time()
        client = pool.acquire(timeout=1.0)
        elapsed = time.time() - start_time

        self.assertIsNone(client)
        self.assertAlmostEqual(elapsed, 1.0, delta=0.2)

    def test_double_shutdown_handling(self):
        """Test handling of double shutdown"""
        pool = EmulatorPool(pool_size=2)
        pool.initialize()

        # First shutdown
        pool.shutdown()
        self.assertTrue(pool._shutdown)

        # Second shutdown should be harmless
        pool.shutdown()  # Should not raise exception
        self.assertTrue(pool._shutdown)


if __name__ == "__main__":
    unittest.main()
