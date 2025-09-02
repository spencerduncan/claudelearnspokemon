"""
Test suite for compatibility validation module.

Tests the validation functions:
- Each validation function tested independently
- Clear test names that reveal intent
- Comprehensive edge case coverage
- Error message validation for user experience

"""

import pytest

from src.claudelearnspokemon.compatibility.validation import (
    ValidationError,
    validate_adapter_type_selection,
    validate_client_creation_parameters,
    validate_container_identifier,
    validate_input_delay_parameter,
    validate_server_url_port,
    validate_timeout_parameter,
)


@pytest.mark.fast
@pytest.mark.medium
class TestServerUrlPortValidation:
    """Test server port validation with comprehensive edge cases."""

    def test_valid_integer_ports(self):
        """Test validation passes for valid integer ports."""
        # Common valid ports
        assert validate_server_url_port(8080) == 8080
        assert validate_server_url_port(8081) == 8081
        assert validate_server_url_port(3000) == 3000
        assert validate_server_url_port(65535) == 65535  # Max valid port
        assert validate_server_url_port(1) == 1  # Min valid port

    def test_valid_string_ports(self):
        """Test validation converts valid string ports to integers."""
        assert validate_server_url_port("8080") == 8080
        assert validate_server_url_port("8081") == 8081
        assert validate_server_url_port("3000") == 3000

    def test_invalid_port_zero(self):
        """Test validation rejects port zero with clear error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_server_url_port(0)
        assert "Invalid port: 0. Must be positive integer." in str(exc_info.value)

    def test_invalid_negative_port(self):
        """Test validation rejects negative ports with clear error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_server_url_port(-1)
        assert "Invalid port: -1. Must be positive integer." in str(exc_info.value)

    def test_invalid_string_port(self):
        """Test validation rejects non-numeric strings with clear error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_server_url_port("invalid")
        assert "Invalid port: invalid. Must be positive integer." in str(exc_info.value)

    def test_invalid_float_port(self):
        """Test validation rejects float values properly."""
        # Should convert 8080.0 to 8080 (valid)
        assert validate_server_url_port(8080.0) == 8080

        # Should reject 8080.5 (invalid)
        with pytest.raises(ValidationError) as exc_info:
            validate_server_url_port(8080.5)
        assert "Invalid port: 8080.5. Must be positive integer." in str(exc_info.value)

    def test_invalid_none_port(self):
        """Test validation rejects None with clear error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_server_url_port(None)
        assert "Invalid port: None. Must be positive integer." in str(exc_info.value)


@pytest.mark.fast
@pytest.mark.medium
class TestContainerIdentifierValidation:
    """Test container ID validation following clean code principles."""

    def test_valid_container_ids(self):
        """Test validation passes for valid container IDs."""
        assert validate_container_identifier("abc123def456") == "abc123def456"
        assert validate_container_identifier("short") == "short"
        assert (
            validate_container_identifier("container_with_underscores")
            == "container_with_underscores"
        )
        assert validate_container_identifier("UPPERCASE") == "UPPERCASE"

    def test_valid_container_id_with_whitespace(self):
        """Test validation trims whitespace from container IDs."""
        assert validate_container_identifier("  abc123  ") == "abc123"
        assert validate_container_identifier("\tcontainer\n") == "container"

    def test_invalid_empty_container_id(self):
        """Test validation rejects empty container ID with exact error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_container_identifier("")
        assert "Container ID cannot be empty" in str(exc_info.value)

    def test_invalid_whitespace_only_container_id(self):
        """Test validation rejects whitespace-only container ID after trimming."""
        with pytest.raises(ValidationError) as exc_info:
            validate_container_identifier("   ")
        assert "Container ID cannot be empty" in str(exc_info.value)

    def test_invalid_none_container_id(self):
        """Test validation rejects None container ID with clear error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_container_identifier(None)
        assert "Container ID must be string" in str(exc_info.value)

    def test_invalid_numeric_container_id(self):
        """Test validation rejects numeric container ID."""
        with pytest.raises(ValidationError) as exc_info:
            validate_container_identifier(12345)
        assert "Container ID must be string, got int" in str(exc_info.value)


@pytest.mark.fast
@pytest.mark.medium
class TestAdapterTypeSelectionValidation:
    """Test adapter type validation with comprehensive coverage."""

    def test_valid_adapter_types(self):
        """Test validation passes for all supported adapter types."""
        assert validate_adapter_type_selection("auto") == "auto"
        assert validate_adapter_type_selection("benchflow") == "benchflow"
        assert validate_adapter_type_selection("direct") == "direct"
        assert validate_adapter_type_selection("fallback") == "fallback"

    def test_invalid_adapter_type(self):
        """Test validation rejects unsupported adapter types with clear error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_adapter_type_selection("invalid")
        assert "Invalid adapter_type: invalid" in str(exc_info.value)

    def test_invalid_adapter_type_case_sensitive(self):
        """Test validation is case sensitive as expected."""
        with pytest.raises(ValidationError) as exc_info:
            validate_adapter_type_selection("AUTO")
        assert "Invalid adapter_type: AUTO" in str(exc_info.value)

    def test_invalid_numeric_adapter_type(self):
        """Test validation rejects numeric adapter type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_adapter_type_selection(123)
        assert "Adapter type must be string, got int" in str(exc_info.value)

    def test_invalid_none_adapter_type(self):
        """Test validation rejects None adapter type."""
        with pytest.raises(ValidationError) as exc_info:
            validate_adapter_type_selection(None)
        assert "Adapter type must be string, got NoneType" in str(exc_info.value)


@pytest.mark.fast
@pytest.mark.medium
class TestTimeoutParameterValidation:
    """Test timeout validation with edge cases and clear naming."""

    def test_valid_timeout_values(self):
        """Test validation passes for valid timeout values."""
        assert validate_timeout_parameter(3.0) == 3.0
        assert validate_timeout_parameter(5.5) == 5.5
        assert validate_timeout_parameter(1) == 1.0  # int converted to float
        assert validate_timeout_parameter(0.1) == 0.1  # small positive value

    def test_valid_timeout_with_custom_parameter_name(self):
        """Test validation uses custom parameter name in error messages."""
        assert validate_timeout_parameter(3.0, "detection_timeout") == 3.0

    def test_invalid_zero_timeout(self):
        """Test validation rejects zero timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_parameter(0.0)
        assert "Invalid timeout: 0.0. Must be positive number." in str(exc_info.value)

    def test_invalid_negative_timeout(self):
        """Test validation rejects negative timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_parameter(-1.5)
        assert "Invalid timeout: -1.5. Must be positive number." in str(exc_info.value)

    def test_invalid_string_timeout(self):
        """Test validation rejects string timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_parameter("invalid")
        assert "Invalid timeout: invalid. Must be positive number." in str(exc_info.value)

    def test_invalid_timeout_with_custom_parameter_name(self):
        """Test validation uses custom parameter name in error messages."""
        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_parameter(-1.0, "custom_timeout")
        assert "Invalid custom_timeout: -1.0. Must be positive number." in str(exc_info.value)


@pytest.mark.fast
@pytest.mark.medium
class TestInputDelayParameterValidation:
    """Test input delay validation allowing zero but rejecting negative values."""

    def test_valid_input_delay_values(self):
        """Test validation passes for valid input delay values."""
        assert validate_input_delay_parameter(0.05) == 0.05
        assert validate_input_delay_parameter(0.1) == 0.1
        assert validate_input_delay_parameter(1.0) == 1.0
        assert validate_input_delay_parameter(0.0) == 0.0  # Zero is valid for input delay
        assert validate_input_delay_parameter(0) == 0.0  # int zero converted to float

    def test_invalid_negative_input_delay(self):
        """Test validation rejects negative input delay."""
        with pytest.raises(ValidationError) as exc_info:
            validate_input_delay_parameter(-0.1)
        assert "Invalid input_delay: -0.1. Must be non-negative number." in str(exc_info.value)

    def test_invalid_string_input_delay(self):
        """Test validation rejects string input delay."""
        with pytest.raises(ValidationError) as exc_info:
            validate_input_delay_parameter("invalid")
        assert "Invalid input_delay: invalid. Must be non-negative number." in str(exc_info.value)

    def test_invalid_none_input_delay(self):
        """Test validation rejects None input delay."""
        with pytest.raises(ValidationError) as exc_info:
            validate_input_delay_parameter(None)
        assert "Invalid input_delay: None. Must be non-negative number." in str(exc_info.value)


@pytest.mark.fast
@pytest.mark.medium
class TestClientCreationParametersValidation:
    """Test the main coordinator validation function."""

    def test_valid_parameter_set(self):
        """Test validation passes for complete valid parameter set."""
        params = validate_client_creation_parameters(
            port=8081,
            container_id="abc123",
            adapter_type="auto",
            input_delay=0.05,
            detection_timeout=3.0,
        )

        expected = {
            "port": 8081,
            "container_id": "abc123",
            "adapter_type": "auto",
            "input_delay": 0.05,
            "detection_timeout": 3.0,
        }
        assert params == expected

    def test_valid_parameter_set_with_defaults(self):
        """Test validation works with default parameters."""
        params = validate_client_creation_parameters(
            port="8082",  # string port
            container_id="  container  ",  # whitespace
        )

        # Should use defaults and clean up values
        assert params["port"] == 8082
        assert params["container_id"] == "container"
        assert params["adapter_type"] == "auto"
        assert params["input_delay"] == 0.05
        assert params["detection_timeout"] == 3.0

    def test_validation_fails_fast_on_invalid_port(self):
        """Test validation fails immediately on invalid port (fail fast principle)."""
        with pytest.raises(ValidationError) as exc_info:
            validate_client_creation_parameters(
                port=-1,  # Invalid port - should fail first
                container_id="",  # Also invalid, but port checked first
                adapter_type="invalid",  # Also invalid
            )
        # Should get port error first (fail fast)
        assert "Invalid port: -1" in str(exc_info.value)

    def test_validation_fails_on_invalid_container_id(self):
        """Test validation fails on invalid container ID when port is valid."""
        with pytest.raises(ValidationError) as exc_info:
            validate_client_creation_parameters(
                port=8081,  # Valid
                container_id="",  # Invalid - should fail here
                adapter_type="invalid",  # Also invalid, but container checked first
            )
        assert "Container ID cannot be empty" in str(exc_info.value)

    def test_validation_fails_on_invalid_adapter_type(self):
        """Test validation fails on invalid adapter type when other params valid."""
        with pytest.raises(ValidationError) as exc_info:
            validate_client_creation_parameters(
                port=8081,  # Valid
                container_id="abc123",  # Valid
                adapter_type="invalid",  # Invalid - should fail here
            )
        assert "Invalid adapter_type: invalid" in str(exc_info.value)

    def test_validation_fails_on_invalid_input_delay(self):
        """Test validation fails on invalid input delay."""
        with pytest.raises(ValidationError) as exc_info:
            validate_client_creation_parameters(
                port=8081,  # Valid
                container_id="abc123",  # Valid
                adapter_type="auto",  # Valid
                input_delay=-0.1,  # Invalid - should fail here
            )
        assert "Invalid input_delay: -0.1" in str(exc_info.value)

    def test_validation_fails_on_invalid_detection_timeout(self):
        """Test validation fails on invalid detection timeout."""
        with pytest.raises(ValidationError) as exc_info:
            validate_client_creation_parameters(
                port=8081,  # Valid
                container_id="abc123",  # Valid
                adapter_type="auto",  # Valid
                input_delay=0.05,  # Valid
                detection_timeout=-1.0,  # Invalid - should fail here
            )
        assert "Invalid detection_timeout: -1.0" in str(exc_info.value)

    def test_all_parameter_types_converted_properly(self):
        """Test all parameters are converted to proper types."""
        params = validate_client_creation_parameters(
            port="8081",  # string -> int
            container_id="  abc123  ",  # whitespace stripped
            adapter_type="benchflow",  # string preserved
            input_delay="0.1",  # string -> float
            detection_timeout=5,  # int -> float
        )

        # Check types are correct
        assert isinstance(params["port"], int)
        assert isinstance(params["container_id"], str)
        assert isinstance(params["adapter_type"], str)
        assert isinstance(params["input_delay"], float)
        assert isinstance(params["detection_timeout"], float)

        # Check values are correct
        assert params["port"] == 8081
        assert params["container_id"] == "abc123"
        assert params["adapter_type"] == "benchflow"
        assert params["input_delay"] == 0.1
        assert params["detection_timeout"] == 5.0


@pytest.mark.fast
@pytest.mark.medium
class TestValidationErrorMessages:
    """Test that error messages are clear and actionable."""

    def test_error_messages_are_descriptive(self):
        """Test all error messages provide clear guidance for users."""
        # Port validation error
        try:
            validate_server_url_port(-1)
        except ValidationError as e:
            assert "Invalid port: -1. Must be positive integer." == str(e)

        # Container ID error
        try:
            validate_container_identifier("")
        except ValidationError as e:
            assert "Container ID cannot be empty" == str(e)

        # Adapter type error
        try:
            validate_adapter_type_selection("invalid")
        except ValidationError as e:
            assert "Invalid adapter_type: invalid" == str(e)

    def test_error_messages_include_actual_values(self):
        """Test error messages include the actual invalid values for debugging."""
        with pytest.raises(ValidationError) as exc_info:
            validate_server_url_port("not_a_number")
        assert "not_a_number" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            validate_timeout_parameter(-5.5, "custom_timeout")
        assert "-5.5" in str(exc_info.value)
        assert "custom_timeout" in str(exc_info.value)

    def test_validation_error_inherits_from_exception(self):
        """Test ValidationError is properly structured for exception handling."""
        assert issubclass(ValidationError, Exception)

        # Should be catchable as Exception
        try:
            validate_server_url_port(-1)
        except Exception as e:
            assert isinstance(e, ValidationError)
