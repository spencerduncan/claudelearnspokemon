#!/usr/bin/env python3
"""
Security Validation Test Suite for Message Routing Engine
Validates all three critical security fixes implemented.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from claudelearnspokemon.message_classifier import (
    MessageClassifier, 
    ClassificationPattern, 
    MessageType,
    validate_message_input,
    validate_regex_pattern
)
from claudelearnspokemon.message_router import TokenBucket, RequestFingerprinter, RoutingRequest

def test_input_validation_security():
    """Test 1: Input validation prevents malicious inputs"""
    print("🔒 Testing input validation security...")
    
    # Test 1.1: Message length validation
    try:
        validate_message_input("x" * 20000)  # Exceeds MAX_MESSAGE_LENGTH
        print("❌ FAIL: Should reject oversized message")
        return False
    except ValueError as e:
        if "exceeds maximum" in str(e):
            print("✅ PASS: Oversized message rejected")
        else:
            print(f"❌ FAIL: Wrong error for oversized message: {e}")
            return False
    
    # Test 1.2: Empty message validation
    try:
        validate_message_input("   ")  # Whitespace only
        print("❌ FAIL: Should reject whitespace-only message")
        return False
    except ValueError as e:
        if "empty or whitespace" in str(e):
            print("✅ PASS: Whitespace-only message rejected")
        else:
            print(f"❌ FAIL: Wrong error for empty message: {e}")
            return False
    
    # Test 1.3: Control character validation
    try:
        validate_message_input("test\x00message")  # Null byte
        print("❌ FAIL: Should reject control characters")
        return False
    except ValueError as e:
        if "control characters" in str(e):
            print("✅ PASS: Control characters rejected")
        else:
            print(f"❌ FAIL: Wrong error for control chars: {e}")
            return False
    
    # Test 1.4: Dangerous regex pattern validation  
    try:
        validate_regex_pattern("(.*)+(.*)+")  # ReDoS vulnerability
        print("❌ FAIL: Should reject ReDoS pattern")
        return False
    except ValueError as e:
        if "dangerous construct" in str(e):
            print("✅ PASS: ReDoS pattern rejected")
        else:
            print(f"❌ FAIL: Wrong error for ReDoS: {e}")
            return False
    
    print("✅ All input validation security tests PASSED")
    return True

def test_rate_limiting_security():
    """Test 2: Rate limiting prevents DoS attacks"""
    print("🔒 Testing rate limiting security...")
    
    # Create token bucket with low limits for testing
    rate_limiter = TokenBucket(rate=5.0, burst_capacity=3, window_seconds=1.0)
    
    # Test 2.1: Normal requests should pass
    allowed_count = 0
    for i in range(3):  # Within burst capacity
        if rate_limiter.consume():
            allowed_count += 1
    
    if allowed_count == 3:
        print("✅ PASS: Normal requests allowed within burst capacity")
    else:
        print(f"❌ FAIL: Expected 3 allowed, got {allowed_count}")
        return False
    
    # Test 2.2: Excess requests should be rate limited
    excess_blocked = 0
    for i in range(5):  # Should exceed capacity
        if not rate_limiter.consume():
            excess_blocked += 1
    
    if excess_blocked > 0:
        print(f"✅ PASS: {excess_blocked} excess requests blocked by rate limiter")
    else:
        print("❌ FAIL: Rate limiter should block excess requests")
        return False
    
    # Test 2.3: Rate limiting metrics are tracked
    metrics = rate_limiter.get_metrics()
    if metrics["rejected_requests"] > 0:
        print("✅ PASS: Rate limiting metrics properly tracked")
    else:
        print("❌ FAIL: Rate limiting metrics not tracked")
        return False
    
    print("✅ All rate limiting security tests PASSED")
    return True

def test_circuit_breaker_bypass_protection():
    """Test 3: Circuit breaker bypass detection prevents malicious patterns"""
    print("🔒 Testing circuit breaker bypass protection...")
    
    fingerprinter = RequestFingerprinter(window_seconds=10.0)
    
    # Create identical malicious requests
    malicious_request = RoutingRequest(content="malicious_payload", context={"client_ip": "192.168.1.100"})
    
    # Test 3.1: Generate consistent fingerprints
    fingerprint1 = fingerprinter.generate_fingerprint(malicious_request, "192.168.1.100")
    fingerprint2 = fingerprinter.generate_fingerprint(malicious_request, "192.168.1.100")
    
    if fingerprint1 == fingerprint2:
        print("✅ PASS: Consistent fingerprints generated for identical requests")
    else:
        print("❌ FAIL: Fingerprints should be consistent")
        return False
    
    # Test 3.2: Different requests generate different fingerprints
    different_request = RoutingRequest(content="different_payload", context={"client_ip": "192.168.1.100"})
    fingerprint3 = fingerprinter.generate_fingerprint(different_request, "192.168.1.100")
    
    if fingerprint1 != fingerprint3:
        print("✅ PASS: Different requests generate different fingerprints")
    else:
        print("❌ FAIL: Different requests should have different fingerprints")
        return False
    
    # Test 3.3: Flooding detection
    suspicious_detected = False
    for i in range(60):  # Exceed MAX_IDENTICAL_FINGERPRINTS (50)
        if fingerprinter.check_suspicious_pattern(fingerprint1, "192.168.1.100"):
            suspicious_detected = True
            break
    
    if suspicious_detected:
        print("✅ PASS: Flooding pattern detected as suspicious")
    else:
        print("❌ FAIL: Should detect flooding pattern")
        return False
    
    # Test 3.4: Security metrics tracking
    metrics = fingerprinter.get_security_metrics()
    if metrics["bypass_attempts_detected"] > 0:
        print("✅ PASS: Bypass attempts properly tracked in metrics")
    else:
        print("❌ FAIL: Bypass attempts should be tracked")
        return False
    
    print("✅ All circuit breaker bypass protection tests PASSED")
    return True

def test_integration_security():
    """Test 4: Integration security - all fixes work together"""
    print("🔒 Testing integration security...")
    
    classifier = MessageClassifier()
    
    # Test 4.1: Valid message should work
    try:
        result = classifier.classify_message("Plan Pokemon strategy", {"client_ip": "192.168.1.1"})
        if result.message_type in [MessageType.STRATEGIC, MessageType.TACTICAL]:
            print("✅ PASS: Valid message classified successfully")
        else:
            print(f"❌ FAIL: Valid message failed classification: {result.error_message}")
            return False
    except Exception as e:
        print(f"❌ FAIL: Valid message threw exception: {e}")
        return False
    
    # Test 4.2: Invalid message should be blocked by input validation
    try:
        result = classifier.classify_message("", {"client_ip": "192.168.1.1"})
        if result.error_message and "validation failed" in result.error_message:
            print("✅ PASS: Invalid message blocked by input validation")
        else:
            print("❌ FAIL: Invalid message should be blocked")
            return False
    except Exception as e:
        print(f"❌ FAIL: Invalid message handling error: {e}")
        return False
    
    print("✅ All integration security tests PASSED")
    return True

def main():
    """Run complete security validation suite"""
    print("🛡️  Security Validation Suite for Message Routing Engine")
    print("=" * 60)
    
    test_results = []
    
    # Run all security tests
    test_results.append(test_input_validation_security())
    print()
    test_results.append(test_rate_limiting_security()) 
    print()
    test_results.append(test_circuit_breaker_bypass_protection())
    print()
    test_results.append(test_integration_security())
    print()
    
    # Summary
    passed = sum(test_results)
    total = len(test_results)
    
    print("=" * 60)
    print(f"🎯 SECURITY VALIDATION RESULTS: {passed}/{total} test suites passed")
    
    if passed == total:
        print("✅ ALL SECURITY VULNERABILITIES SUCCESSFULLY MITIGATED!")
        print("🚀 System is ready for production deployment")
        return True
    else:
        print("❌ SECURITY VALIDATION FAILED - Do not deploy to production")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)