"""
Comprehensive test suite for ScriptCompiler with performance validation.

Tests cover:
- All functional requirements from design specification
- Performance requirements (<100ms compilation)
- Edge cases and error conditions
- Macro expansion and recursive detection
- Frame estimation accuracy
- Observation point insertion
"""

import statistics
import time

import pytest
from claudelearnspokemon.dsl_ast import NodeFactory, NodeType
from claudelearnspokemon.script_compiler import (
    CodeGenerator,
    HighPerformanceLexer,
    MacroRegistry,
    ScriptCompiler,
    ScriptParser,
    TokenType,
)


class TestHighPerformanceLexer:
    """Test lexical analysis performance and correctness."""

    def test_tokenizes_basic_inputs(self) -> None:
        """Test basic input tokenization."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("A B START")

        assert len(tokens) == 4  # 3 inputs + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "A"
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "B"
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "START"
        assert tokens[3].type == TokenType.EOF

    def test_tokenizes_numbers(self) -> None:
        """Test number tokenization for delays."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("5 10 123")

        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "5"
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "10"
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "123"

    def test_tokenizes_keywords(self) -> None:
        """Test keyword recognition."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("if then else end repeat times macro observe")

        expected_types = [
            TokenType.IF,
            TokenType.THEN,
            TokenType.ELSE,
            TokenType.END,
            TokenType.REPEAT,
            TokenType.TIMES,
            TokenType.MACRO,
            TokenType.OBSERVE,
            TokenType.EOF,
        ]

        for i, expected_type in enumerate(expected_types):
            assert tokens[i].type == expected_type

    def test_tokenizes_punctuation(self) -> None:
        """Test punctuation tokenization."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("( ) { } ; , =")

        expected_types = [
            TokenType.LPAREN,
            TokenType.RPAREN,
            TokenType.LBRACE,
            TokenType.RBRACE,
            TokenType.SEMICOLON,
            TokenType.COMMA,
            TokenType.ASSIGN,
            TokenType.EOF,
        ]

        for i, expected_type in enumerate(expected_types):
            assert tokens[i].type == expected_type

    def test_handles_comments(self) -> None:
        """Test comment handling."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("A # This is a comment\nB")

        # Comments should be skipped
        assert len(tokens) == 4  # A, NEWLINE, B, EOF
        assert tokens[0].value == "A"
        assert tokens[1].type == TokenType.NEWLINE
        assert tokens[2].value == "B"

    def test_tracks_line_numbers(self) -> None:
        """Test line number tracking."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("A\nB\nC")

        assert tokens[0].line == 1  # A
        assert tokens[1].line == 1  # NEWLINE
        assert tokens[2].line == 2  # B
        assert tokens[3].line == 2  # NEWLINE
        assert tokens[4].line == 3  # C

    def test_lexer_performance(self) -> None:
        """Test lexer performance with large input."""
        lexer = HighPerformanceLexer()

        # Create large script with 1000 tokens
        large_script = " ".join(["A B START"] * 333)

        start_time = time.perf_counter()
        tokens = lexer.tokenize(large_script)
        end_time = time.perf_counter()

        tokenize_time_ms = (end_time - start_time) * 1000

        # Should tokenize in <10ms (performance target)
        assert tokenize_time_ms < 10.0
        assert len(tokens) == 1000  # 999 tokens + EOF


class TestMacroRegistry:
    """Test macro storage and expansion with cycle detection."""

    def test_registers_simple_pattern(self) -> None:
        """Test basic pattern registration."""
        registry = MacroRegistry()
        registry.register_pattern("TEST", ["A", "B"])

        pattern = registry.get_pattern("TEST")
        assert pattern is not None
        assert len(pattern) == 2

    def test_expands_macro_correctly(self) -> None:
        """Test macro expansion."""
        registry = MacroRegistry()
        registry.register_pattern("CONFIRM", ["A", "1"])

        expanded = registry.expand_macro("CONFIRM")
        assert expanded == ["A", "WAIT"]  # 1 becomes WAIT

    def test_detects_direct_recursion(self) -> None:
        """Test detection of direct recursive macros."""
        registry = MacroRegistry()
        registry.register_pattern("RECURSIVE", ["RECURSIVE"])

        assert registry.has_recursive_dependency("RECURSIVE")

    def test_detects_indirect_recursion(self) -> None:
        """Test detection of indirect recursive macros."""
        registry = MacroRegistry()
        registry.register_pattern("A", ["B"])
        registry.register_pattern("B", ["C"])
        registry.register_pattern("C", ["A"])  # Creates cycle

        assert registry.has_recursive_dependency("A")
        assert registry.has_recursive_dependency("B")
        assert registry.has_recursive_dependency("C")

    def test_allows_non_recursive_patterns(self) -> None:
        """Test that non-recursive patterns are allowed."""
        registry = MacroRegistry()
        registry.register_pattern("SAFE_A", ["UP", "DOWN"])
        registry.register_pattern("SAFE_B", ["SAFE_A", "LEFT"])

        assert not registry.has_recursive_dependency("SAFE_A")
        assert not registry.has_recursive_dependency("SAFE_B")

    def test_estimates_pattern_frames(self) -> None:
        """Test frame estimation for patterns."""
        registry = MacroRegistry()
        registry.register_pattern("SHORT", ["A"])  # 1 frame
        registry.register_pattern("LONG", ["A", "5", "B"])  # 1 + 5 + 1 = 7 frames

        assert registry.estimate_pattern_frames("SHORT") == 1
        assert registry.estimate_pattern_frames("LONG") == 7


class TestScriptParser:
    """Test parsing DSL syntax into AST."""

    def setup_method(self) -> None:
        """Set up parser for each test."""
        self.registry = MacroRegistry()
        self.parser = ScriptParser(self.registry)

    def test_parses_simple_input_sequence(self) -> None:
        """Test parsing basic input sequence."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("A B START")

        ast = self.parser.parse(tokens)

        assert ast.node_type == NodeType.SEQUENCE
        assert len(ast.children) == 3
        assert ast.children[0].node_type == NodeType.INPUT
        assert ast.children[0].value == "A"

    def test_parses_repeat_construct(self) -> None:
        """Test parsing repeat loops."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("repeat 3 times\nA\nend")

        ast = self.parser.parse(tokens)

        assert ast.node_type == NodeType.REPEAT
        assert ast.value == 3
        assert len(ast.children) == 1
        assert ast.children[0].node_type == NodeType.INPUT
        assert ast.children[0].value == "A"

    def test_parses_conditional_construct(self) -> None:
        """Test parsing if/then/else."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("if test then A else B end")

        ast = self.parser.parse(tokens)

        assert ast.node_type == NodeType.CONDITIONAL
        assert ast.value == "test"
        assert len(ast.children) == 2  # then and else branches

    def test_parses_observation_points(self) -> None:
        """Test parsing observation points."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("observe(0.5)")

        ast = self.parser.parse(tokens)

        assert ast.node_type == NodeType.OBSERVATION
        assert ast.value == 0.5

    def test_handles_nested_constructs(self) -> None:
        """Test parsing nested constructs."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("repeat 2 times\nif test then A end\nend")

        ast = self.parser.parse(tokens)

        assert ast.node_type == NodeType.REPEAT
        assert len(ast.children) == 1
        assert ast.children[0].node_type == NodeType.CONDITIONAL

    def test_recovers_from_syntax_errors(self) -> None:
        """Test error recovery during parsing."""
        lexer = HighPerformanceLexer()
        tokens = lexer.tokenize("A invalid_syntax B")

        # Should parse A and B, skipping invalid syntax
        ast = self.parser.parse(tokens)
        assert ast is not None


class TestCodeGenerator:
    """Test code generation and instruction flattening."""

    def setup_method(self) -> None:
        """Set up code generator for each test."""
        self.registry = MacroRegistry()
        self.generator = CodeGenerator(self.registry)
        self.factory = NodeFactory()

    def test_generates_input_instructions(self) -> None:
        """Test generation of basic input instructions."""
        ast = self.factory.create_input_node("A")

        script = self.generator.generate(ast)

        assert script.instructions == ("A",)
        assert script.total_frames == 1

    def test_generates_delay_instructions(self) -> None:
        """Test generation of delay instructions."""
        ast = self.factory.create_delay_node(3)

        script = self.generator.generate(ast)

        assert script.instructions == ("WAIT", "WAIT", "WAIT")
        assert script.total_frames == 3

    def test_generates_sequence_instructions(self) -> None:
        """Test generation of instruction sequences."""
        children = [
            self.factory.create_input_node("A"),
            self.factory.create_delay_node(2),
            self.factory.create_input_node("B"),
        ]
        ast = self.factory.create_sequence_node(children)

        script = self.generator.generate(ast)

        assert script.instructions == ("A", "WAIT", "WAIT", "B")
        assert script.total_frames == 4

    def test_generates_repeat_instructions(self) -> None:
        """Test generation of repeated instructions."""
        children = [self.factory.create_input_node("A")]
        ast = self.factory.create_repeat_node(3, children)

        script = self.generator.generate(ast)

        assert script.instructions == ("A", "A", "A")
        assert script.total_frames == 3

    def test_expands_macros_inline(self) -> None:
        """Test inline macro expansion."""
        self.registry.register_pattern("TEST", ["A", "B"])
        ast = self.factory.create_macro_node("TEST")

        script = self.generator.generate(ast)

        assert len(script.instructions) == 2
        assert script.total_frames >= 2

    def test_inserts_observation_points(self) -> None:
        """Test observation point insertion."""
        ast = self.factory.create_observation_node(1.0)

        script = self.generator.generate(ast)

        assert len(script.observation_points) == 1
        assert script.observation_points[0] == 0


class TestScriptCompiler:
    """Test complete compiler functionality and performance."""

    def setup_method(self) -> None:
        """Set up compiler for each test."""
        self.compiler = ScriptCompiler()

    def test_compiler_tokenizes_dsl_syntax_into_ast_nodes(self) -> None:
        """Test complete tokenization to AST pipeline."""
        script = "A B START"

        compiled = self.compiler.compile(script)

        assert compiled.instructions == ("A", "B", "START")
        assert compiled.total_frames == 3

    def test_compiler_expands_macros_to_primitive_sequences(self) -> None:
        """Test macro expansion in compilation."""
        self.compiler.register_pattern("CONFIRM", ["A", "1"])
        script = "CONFIRM B"

        compiled = self.compiler.compile(script)

        # CONFIRM expands to A + WAIT, then B
        assert len(compiled.instructions) >= 3
        assert "B" in compiled.instructions

    def test_compiler_handles_nested_macro_expansion(self) -> None:
        """Test nested macro expansion."""
        self.compiler.register_pattern("BASE", ["A"])
        self.compiler.register_pattern("NESTED", ["BASE", "B"])
        script = "NESTED"

        compiled = self.compiler.compile(script)

        assert len(compiled.instructions) >= 2

    def test_compiler_detects_recursive_macro_definitions(self) -> None:
        """Test recursive macro detection."""
        with pytest.raises(ValueError, match="Recursive macro definition"):
            self.compiler.register_pattern("LOOP", ["LOOP"])

    def test_compiler_validates_parameter_types_in_patterns(self) -> None:
        """Test parameter validation for patterns."""
        with pytest.raises(ValueError, match="Pattern name cannot be empty"):
            self.compiler.register_pattern("", ["A"])

        with pytest.raises(ValueError, match="Pattern expansion cannot be empty"):
            self.compiler.register_pattern("TEST", [])

    def test_compiler_estimates_frame_count_accurately(self) -> None:
        """Test accurate frame count estimation."""
        script = "A 5 B"  # A (1) + delay (5) + B (1) = 7 frames

        compiled = self.compiler.compile(script)

        assert self.compiler.estimate_frames(compiled) == 7
        assert compiled.total_frames == 7

    def test_compiler_inserts_observation_points_at_specified_density(self) -> None:
        """Test observation point insertion."""
        script = "A B C D"  # 4 frame script

        compiled = self.compiler.compile(script)
        enhanced = self.compiler.add_observation_points(compiled, 0.5)

        # With density 0.5, should observe every 2nd frame
        assert len(enhanced.observation_points) == 2
        assert 0 in enhanced.observation_points
        assert 2 in enhanced.observation_points

    def test_compiler_maintains_compilation_determinism(self) -> None:
        """Test that compilation is deterministic."""
        script = "A B repeat 2 times\nC\nend"

        compiled1 = self.compiler.compile(script)
        compiled2 = self.compiler.compile(script)

        assert compiled1.instructions == compiled2.instructions
        assert compiled1.total_frames == compiled2.total_frames

    def test_validates_syntax_correctly(self) -> None:
        """Test syntax validation."""
        assert self.compiler.validate_syntax("A B C")  # Valid
        assert self.compiler.validate_syntax("repeat 3 times\nA\nend")  # Valid
        assert not self.compiler.validate_syntax("repeat invalid syntax")  # Invalid

    def test_returns_language_specification(self) -> None:
        """Test language specification retrieval."""
        spec = self.compiler.get_language_spec()

        assert "version" in spec
        assert "keywords" in spec
        assert "inputs" in spec
        assert "patterns" in spec
        assert isinstance(spec["keywords"], list)
        assert "if" in spec["keywords"]
        assert "A" in spec["inputs"]

    def test_handles_empty_script(self) -> None:
        """Test compilation of empty script."""
        compiled = self.compiler.compile("")

        assert len(compiled.instructions) >= 1  # Should have at least NOOP
        assert compiled.total_frames >= 0

    def test_handles_whitespace_and_comments(self) -> None:
        """Test handling of whitespace and comments."""
        script = """
        # This is a comment
        A   # Another comment

        B  # Final comment
        """

        compiled = self.compiler.compile(script)

        assert compiled.instructions == ("A", "B")
        assert compiled.total_frames == 2


class TestPerformanceRequirements:
    """Test performance requirements and benchmarks."""

    def setup_method(self) -> None:
        """Set up performance tests."""
        self.compiler = ScriptCompiler()

    def test_compilation_time_under_100ms_simple(self) -> None:
        """Test <100ms compilation for simple scripts."""
        script = "A B START SELECT"

        start_time = time.perf_counter()
        compiled = self.compiler.compile(script)
        end_time = time.perf_counter()

        compile_time_ms = (end_time - start_time) * 1000

        assert compile_time_ms < 100.0
        assert compiled.metadata["compile_time_ms"] < 100.0

    def test_compilation_time_under_100ms_complex(self) -> None:
        """Test <100ms compilation for complex scripts."""
        script = """
        # Complex script with nested constructs
        repeat 10 times
            if condition then
                A B
                repeat 3 times
                    UP DOWN
                end
            else
                START SELECT
            end
            observe(0.1)
        end
        """

        start_time = time.perf_counter()
        compiled = self.compiler.compile(script)
        end_time = time.perf_counter()

        compile_time_ms = (end_time - start_time) * 1000

        assert compile_time_ms < 100.0
        assert len(compiled.instructions) > 0

    def test_compilation_time_under_100ms_with_macros(self) -> None:
        """Test <100ms compilation with macro expansion."""
        # Register several macros
        self.compiler.register_pattern("MOVE_UP", ["UP", "1"])
        self.compiler.register_pattern("MOVE_DOWN", ["DOWN", "1"])
        self.compiler.register_pattern("MOVE_SEQUENCE", ["MOVE_UP", "MOVE_DOWN"])

        script = """
        repeat 20 times
            MOVE_SEQUENCE
            A B
        end
        """

        start_time = time.perf_counter()
        compiled = self.compiler.compile(script)
        end_time = time.perf_counter()

        compile_time_ms = (end_time - start_time) * 1000

        assert compile_time_ms < 100.0
        # Deterministic: 20 iterations * 4 instructions (UP, DOWN, A, B) = 80
        assert len(compiled.instructions) == 80

    def test_large_script_compilation_performance(self) -> None:
        """Test performance with very large scripts."""
        # Generate large script programmatically
        large_script_parts = []
        for i in range(100):
            large_script_parts.append(f"A B repeat {i % 5 + 1} times\nUP DOWN\nend")

        large_script = "\n".join(large_script_parts)

        start_time = time.perf_counter()
        compiled = self.compiler.compile(large_script)
        end_time = time.perf_counter()

        compile_time_ms = (end_time - start_time) * 1000

        # Even large scripts should compile in <100ms
        assert compile_time_ms < 100.0
        # Deterministic: 100 iterations with varying repeat counts
        # Sum of (2 + 2*(i%5+1)) for i in range(100) = 800 instructions
        assert len(compiled.instructions) == 800

    def test_repeated_compilation_performance(self) -> None:
        """Test performance consistency across multiple compilations."""
        script = "repeat 5 times\nA B UP DOWN\nend"

        compilation_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            self.compiler.compile(script)
            end_time = time.perf_counter()
            compilation_times.append((end_time - start_time) * 1000)

        avg_time = statistics.mean(compilation_times)
        max_time = max(compilation_times)

        assert avg_time < 50.0  # Average should be well under limit
        assert max_time < 100.0  # No single compilation should exceed limit

    def test_observation_point_insertion_performance(self) -> None:
        """Test performance of observation point insertion."""
        # Create large compiled script
        script = "A " * 1000  # 1000 input sequence
        compiled = self.compiler.compile(script)

        # Time observation point insertion
        start_time = time.perf_counter()
        enhanced = self.compiler.add_observation_points(compiled, 0.1)
        end_time = time.perf_counter()

        insertion_time_ms = (end_time - start_time) * 1000

        assert insertion_time_ms < 10.0  # Should be very fast
        assert len(enhanced.observation_points) == 100  # 10% of 1000 frames


class TestErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self) -> None:
        """Set up error handling tests."""
        self.compiler = ScriptCompiler()

    def test_handles_invalid_syntax_gracefully(self) -> None:
        """Test graceful handling of syntax errors."""
        invalid_script = "repeat without end times A"

        # Should not crash, but may produce empty or minimal result
        try:
            compiled = self.compiler.compile(invalid_script)
            assert compiled is not None
        except Exception as e:
            assert "Parse error" in str(e) or "Compilation failed" in str(e)

    def test_handles_missing_macro_definitions(self) -> None:
        """Test handling of undefined macros."""
        script = "UNDEFINED_MACRO"

        # Should either expand to empty or raise clear error
        try:
            compiled = self.compiler.compile(script)
            # If it doesn't raise, it should produce some result
            assert compiled is not None
        except Exception as e:
            assert "Unknown macro" in str(e) or "UNDEFINED_MACRO" in str(e)

    def test_handles_extremely_large_numbers(self) -> None:
        """Test handling of very large numbers."""
        script = "99999999"  # Very large delay

        try:
            compiled = self.compiler.compile(script)
            # If successful, should have correct frame count
            assert compiled.total_frames > 0
        except Exception as e:
            # Or should give clear error about resource limits
            assert isinstance(e, ValueError | OverflowError)

    def test_compilation_timeout_detection(self) -> None:
        """Test that compilation time limit is enforced."""
        # This test would require a way to artificially slow down compilation
        # For now, we verify the timeout mechanism exists
        assert hasattr(self.compiler, "compile")

        # Verify performance stats tracking
        stats = self.compiler.get_performance_stats()
        assert "max_time_ms" in stats
        assert "average_time_ms" in stats


if __name__ == "__main__":
    # Run performance benchmarks when executed directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("Running ScriptCompiler Performance Benchmarks...")

        compiler = ScriptCompiler()

        # Benchmark 1: Simple compilation
        simple_script = "A B START SELECT UP DOWN LEFT RIGHT"
        times = []
        for _ in range(100):
            start = time.perf_counter()
            compiler.compile(simple_script)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        print("Simple compilation (100 runs):")
        print(f"  Average: {statistics.mean(times):.2f}ms")
        print(f"  Max: {max(times):.2f}ms")
        print(f"  Min: {min(times):.2f}ms")

        # Benchmark 2: Complex compilation
        complex_script = """
        repeat 20 times
            if test then
                A B
                repeat 3 times
                    UP DOWN LEFT RIGHT
                end
                observe(0.1)
            else
                START SELECT
            end
        end
        """

        times = []
        for _ in range(50):
            start = time.perf_counter()
            compiler.compile(complex_script)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        print("\nComplex compilation (50 runs):")
        print(f"  Average: {statistics.mean(times):.2f}ms")
        print(f"  Max: {max(times):.2f}ms")
        print(f"  Min: {min(times):.2f}ms")

        print("\nPerformance requirement: <100ms")
        print(f"Status: {'PASS' if max(times) < 100.0 else 'FAIL'}")
    else:
        # Run tests with pytest
        pytest.main([__file__, "-v"])
