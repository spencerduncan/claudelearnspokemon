"""
Parameter validation module for Pokemon Gym factory operations.

Provides focused, single-responsibility validation functions following Clean Code principles.
Each function validates one specific parameter with clear error messages and intent-revealing names.

Author: Uncle Bot - Clean Code Craftsmanship Applied
"""


class ValidationError(Exception):
    """
    Validation-specific exception for parameter validation errors.

    Provides clear, actionable error messages for production debugging.
    """

    pass


def validate_server_url_port(port: int | str) -> int:
    """
    Validate server port parameter for Pokemon Gym connections.

    Ensures port is a positive integer suitable for HTTP server connections.
    Follows the principle: "Functions should do one thing."

    Args:
        port: Port number to validate (int or convertible string)

    Returns:
        int: Validated port number

    Raises:
        ValidationError: If port is invalid with specific reason

    Examples:
        >>> validate_server_url_port(8081)
        8081
        >>> validate_server_url_port("8082")
        8082
        >>> validate_server_url_port(-1)
        ValidationError: Invalid port: -1. Must be positive integer.
    """
    # Check for float values that aren't whole numbers
    if isinstance(port, float) and port != int(port):
        raise ValidationError(f"Invalid port: {port}. Must be positive integer.")

    try:
        port_int = int(port)
    except (ValueError, TypeError) as err:
        raise ValidationError(f"Invalid port: {port}. Must be positive integer.") from err

    if port_int <= 0:
        raise ValidationError(f"Invalid port: {port_int}. Must be positive integer.")

    return port_int


def validate_container_identifier(container_id: str) -> str:
    """
    Validate Docker container identifier parameter.

    Ensures container ID is not empty and suitable for Docker operations.
    Clean Code principle: "Use intention-revealing names."

    Args:
        container_id: Docker container ID to validate

    Returns:
        str: Validated container ID

    Raises:
        ValidationError: If container ID is invalid

    Examples:
        >>> validate_container_identifier("abc123def456")
        "abc123def456"
        >>> validate_container_identifier("")
        ValidationError: Container ID cannot be empty
    """
    # Check type first - fail fast principle
    if not isinstance(container_id, str):
        raise ValidationError(f"Container ID must be string, got {type(container_id).__name__}")

    # Trim whitespace and then check if empty
    cleaned_id = container_id.strip()
    if not cleaned_id:
        raise ValidationError("Container ID cannot be empty")

    return cleaned_id


def validate_adapter_type_selection(adapter_type: str) -> str:
    """
    Validate adapter type parameter for Pokemon Gym client selection.

    Ensures adapter type is one of the supported values for transparent client creation.
    Single Responsibility: Only validates adapter type, nothing else.

    Args:
        adapter_type: Type of adapter to validate

    Returns:
        str: Validated adapter type

    Raises:
        ValidationError: If adapter type is not supported

    Examples:
        >>> validate_adapter_type_selection("auto")
        "auto"
        >>> validate_adapter_type_selection("invalid")
        ValidationError: Invalid adapter_type: invalid
    """
    SUPPORTED_ADAPTER_TYPES = ("auto", "benchflow", "direct", "fallback")

    if not isinstance(adapter_type, str):
        raise ValidationError(f"Adapter type must be string, got {type(adapter_type).__name__}")

    if adapter_type not in SUPPORTED_ADAPTER_TYPES:
        raise ValidationError(f"Invalid adapter_type: {adapter_type}")

    return adapter_type


def validate_timeout_parameter(timeout: float | int, parameter_name: str = "timeout") -> float:
    """
    Validate timeout parameter for network operations.

    Ensures timeout is positive number suitable for network requests.
    Generic function for any timeout validation needs.

    Args:
        timeout: Timeout value to validate
        parameter_name: Name of parameter for error messages

    Returns:
        float: Validated timeout value

    Raises:
        ValidationError: If timeout is invalid

    Examples:
        >>> validate_timeout_parameter(3.0)
        3.0
        >>> validate_timeout_parameter(-1.0)
        ValidationError: Invalid timeout: -1.0. Must be positive number.
    """
    try:
        timeout_float = float(timeout)
    except (ValueError, TypeError) as err:
        raise ValidationError(
            f"Invalid {parameter_name}: {timeout}. Must be positive number."
        ) from err

    if timeout_float <= 0:
        raise ValidationError(
            f"Invalid {parameter_name}: {timeout_float}. Must be positive number."
        )

    return timeout_float


def validate_input_delay_parameter(input_delay: float | int) -> float:
    """
    Validate input delay parameter for Pokemon Gym operations.

    Ensures input delay is non-negative number suitable for timing control.
    Specific validation for input delay with appropriate constraints.

    Args:
        input_delay: Input delay value to validate

    Returns:
        float: Validated input delay

    Raises:
        ValidationError: If input delay is invalid

    Examples:
        >>> validate_input_delay_parameter(0.05)
        0.05
        >>> validate_input_delay_parameter(0.0)  # Zero is valid
        0.0
        >>> validate_input_delay_parameter(-0.1)
        ValidationError: Invalid input_delay: -0.1. Must be non-negative number.
    """
    try:
        delay_float = float(input_delay)
    except (ValueError, TypeError) as err:
        raise ValidationError(
            f"Invalid input_delay: {input_delay}. Must be non-negative number."
        ) from err

    if delay_float < 0:
        raise ValidationError(f"Invalid input_delay: {delay_float}. Must be non-negative number.")

    return delay_float


def validate_client_creation_parameters(
    port: int | str,
    container_id: str,
    adapter_type: str = "auto",
    input_delay: float | int = 0.05,
    detection_timeout: float | int = 3.0,
) -> dict[str, int | str | float]:
    """
    Validate all parameters for Pokemon client creation in one coordinated operation.

    This is the main validation coordinator that applies Single Responsibility Principle
    by delegating each specific validation to focused functions.

    Clean Code principle: "Functions should do one thing" - this function coordinates
    validation but doesn't implement the validation logic itself.

    Args:
        port: HTTP port for emulator communication
        container_id: Docker container ID for this emulator
        adapter_type: Type selection - "auto", "benchflow", "direct", or "fallback"
        input_delay: Delay between sequential inputs for benchflow adapter
        detection_timeout: Timeout for server type detection

    Returns:
        dict: Validated parameters ready for client creation

    Raises:
        ValidationError: If any parameter is invalid with specific details

    Example:
        >>> params = validate_client_creation_parameters(
        ...     port=8081,
        ...     container_id="abc123",
        ...     adapter_type="auto"
        ... )
        >>> params["port"]
        8081
    """
    # Apply validation in logical order - fail fast on most basic errors first
    validated_port = validate_server_url_port(port)
    validated_container_id = validate_container_identifier(container_id)
    validated_adapter_type = validate_adapter_type_selection(adapter_type)
    validated_input_delay = validate_input_delay_parameter(input_delay)
    validated_detection_timeout = validate_timeout_parameter(detection_timeout, "detection_timeout")

    return {
        "port": validated_port,
        "container_id": validated_container_id,
        "adapter_type": validated_adapter_type,
        "input_delay": validated_input_delay,
        "detection_timeout": validated_detection_timeout,
    }


# Uncle Bob's Clean Code principle: "The ratio of time spent reading versus writing is well over 10 to 1."
# These validation functions are optimized for readability and maintainability.
# Each function has a single, clear purpose with intention-revealing names.
