# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import unittest
from unittest.mock import MagicMock, patch

from are.simulation.validation.constants import SoftCheckerType
from are.simulation.validation.prompts import (
    CONTENT_CHECKER_PROMPT_TEMPLATES,
    EVENT_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
    SANITY_CHECKER_PROMPT_TEMPLATES,
    SIGNATURE_CHECKER_TEMPLATES,
    SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
    USER_MESSAGE_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
    LLMFunctionTemplates,
)
from are.simulation.validation.utils.llm_utils import (
    LLMChecker,
    LLMFunction,
    build_llm_checkers,
    build_subtask_extractor,
)


class TestLLMFunction(unittest.TestCase):
    def setUp(self):
        # Mock engine
        self.mock_engine = MagicMock()
        self.mock_engine.return_value = ("Test response", None)

        # Mock prompt templates
        self.mock_prompt_templates = MagicMock(spec=LLMFunctionTemplates)
        template = "System: {{system_arg}}"
        self.mock_prompt_templates.system_prompt_template = template
        self.mock_prompt_templates.system_prompt_args = {"system_arg": "test"}
        self.mock_prompt_templates.user_prompt_template = "User: {{user_arg}}"
        self.mock_prompt_templates.assistant_prompt_template = None
        self.mock_prompt_templates.examples = None

    @patch("are.simulation.validation.utils.llm_utils.jinja_format")
    def test_init_without_examples(self, mock_jinja):
        mock_jinja.return_value = "Formatted prompt"

        llm_function = LLMFunction(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
        )

        self.assertEqual(llm_function.engine, self.mock_engine)
        self.assertEqual(llm_function.prompt_templates, self.mock_prompt_templates)
        self.assertEqual(llm_function.system_prompt, "Formatted prompt")
        self.assertEqual(llm_function.user_prompt_template, "User: {{user_arg}}")
        self.assertEqual(len(llm_function.examples), 0)

        # Check system prompt formatting call
        mock_jinja.assert_called_with(
            template="System: {{system_arg}}",
            skip_validation=False,
            system_arg="test",
        )

    @patch("are.simulation.validation.utils.llm_utils.jinja_format")
    def test_init_with_examples(self, mock_jinja):
        # Setup examples
        assistant_template = "Assistant: {{output_arg}}"
        self.mock_prompt_templates.assistant_prompt_template = assistant_template
        self.mock_prompt_templates.examples = [
            {
                "input": {"user_arg": "example_input"},
                "output": {"output_arg": "example_output"},
            }
        ]

        mock_jinja.side_effect = [
            "System formatted",  # System prompt
            "User example",  # User example
            "Assistant example",  # Assistant example
        ]

        llm_function = LLMFunction(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
        )

        self.assertEqual(len(llm_function.examples), 2)  # User + assistant
        self.assertEqual(llm_function.examples[0]["role"], "user")
        self.assertEqual(llm_function.examples[0]["content"], "User example")
        self.assertEqual(llm_function.examples[1]["role"], "assistant")
        self.assertEqual(llm_function.examples[1]["content"], "Assistant example")

    @patch("are.simulation.validation.utils.llm_utils.jinja_format")
    def test_call(self, mock_jinja):
        # Setup
        mock_jinja.side_effect = ["System prompt", "User prompt"]
        llm_function = LLMFunction(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
        )

        # Call
        result = llm_function({"user_arg": "test_value"})

        # Assertions
        self.assertEqual(result, "Test response")
        self.mock_engine.assert_called_once()

        # Check the messages passed to engine
        call_args = self.mock_engine.call_args
        messages = call_args[0][0]

        expected_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
        ]
        self.assertEqual(messages, expected_messages)

        # Check additional trace tags
        kwargs = call_args[1]
        self.assertEqual(kwargs["additional_trace_tags"], ["judge_llm"])

    @patch("are.simulation.validation.utils.llm_utils.jinja_format")
    def test_call_with_examples(self, mock_jinja):
        # Setup with examples
        assistant_template = "Assistant: {{output_arg}}"
        self.mock_prompt_templates.assistant_prompt_template = assistant_template
        self.mock_prompt_templates.examples = [
            {
                "input": {"user_arg": "example_input"},
                "output": {"output_arg": "example_output"},
            }
        ]

        mock_jinja.side_effect = [
            "System prompt",  # System init
            "Example user",  # Example user init
            "Example assistant",  # Example assistant init
            "Actual user prompt",  # Actual call
        ]

        llm_function = LLMFunction(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
        )

        _ = llm_function({"user_arg": "test_value"})

        # Check messages structure
        call_args = self.mock_engine.call_args
        messages = call_args[0][0]

        expected_messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Example user"},
            {"role": "assistant", "content": "Example assistant"},
            {"role": "user", "content": "Actual user prompt"},
        ]
        self.assertEqual(messages, expected_messages)


class TestLLMChecker(unittest.TestCase):
    def setUp(self):
        # Mock engine
        self.mock_engine = MagicMock()

        # Mock prompt templates
        self.mock_prompt_templates = MagicMock(spec=LLMFunctionTemplates)
        self.mock_prompt_templates.system_prompt_template = "System prompt"
        self.mock_prompt_templates.user_prompt_template = "User: {{user_arg}}"
        self.mock_prompt_templates.system_prompt_args = None
        self.mock_prompt_templates.assistant_prompt_template = None
        self.mock_prompt_templates.examples = None

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_init_default_params(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
        )

        self.assertEqual(checker.engine, self.mock_engine)
        self.assertEqual(checker.prompt_templates, self.mock_prompt_templates)
        self.assertEqual(checker.judge, mock_judge)
        self.assertEqual(checker.num_votes, 1)
        self.assertEqual(checker.success_str, "[[Success]]")
        self.assertEqual(checker.failure_str, "[[Failure]]")

        # Verify LLMFunction was created correctly
        mock_llm_function_class.assert_called_once_with(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
        )

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_init_custom_params(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=3,
            success_str="[[True]]",
            failure_str="[[False]]",
        )

        self.assertEqual(checker.num_votes, 3)
        self.assertEqual(checker.success_str, "[[True]]")
        self.assertEqual(checker.failure_str, "[[False]]")

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_single_vote_success(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_judge.return_value = "Some reasoning [[Success]] more text"
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=1,
        )

        result = checker({"user_arg": "test"})

        self.assertTrue(result)
        mock_judge.assert_called_once_with({"user_arg": "test"})

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_single_vote_failure(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_judge.return_value = "Some reasoning [[Failure]] more text"
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=1,
        )

        result = checker({"user_arg": "test"})

        self.assertFalse(result)

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_multiple_votes_majority_success(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_judge.side_effect = [
            "[[Success]]",  # Vote 1
            "[[Failure]]",  # Vote 2
            "[[Success]]",  # Vote 3
        ]
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=3,
        )

        result = checker({"user_arg": "test"})

        self.assertTrue(result)  # 2 successes out of 3
        self.assertEqual(mock_judge.call_count, 3)

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_multiple_votes_majority_failure(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_judge.side_effect = [
            "[[Success]]",  # Vote 1
            "[[Failure]]",  # Vote 2
            "[[Failure]]",  # Vote 3
        ]
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=3,
        )

        result = checker({"user_arg": "test"})

        self.assertFalse(result)  # 1 success out of 3

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_no_valid_responses(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_judge.side_effect = [
            None,  # No response
            "Invalid response without markers",  # Invalid response
            None,  # No response
        ]
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=3,
        )

        result = checker({"user_arg": "test"})

        self.assertIsNone(result)  # No valid votes
        self.assertEqual(mock_judge.call_count, 3)

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_custom_success_failure_strings(self, mock_llm_function_class):
        mock_judge = MagicMock()
        mock_judge.return_value = "Response with [[True]] marker"
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=1,
            success_str="[[True]]",
            failure_str="[[False]]",
        )

        result = checker({"user_arg": "test"})

        self.assertTrue(result)

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_tie_vote_rounds_down(self, mock_llm_function_class):
        """Test that a tie vote (2 out of 4) returns True since 2 >= 4/2"""
        mock_judge = MagicMock()
        mock_judge.side_effect = [
            "[[Success]]",  # Vote 1
            "[[Success]]",  # Vote 2
            "[[Failure]]",  # Vote 3
            "[[Failure]]",  # Vote 4
        ]
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=4,
        )

        result = checker({"user_arg": "test"})

        self.assertTrue(result)  # 2 successes out of 4, not >= 2

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_call_verdict_markers_are_case_insensitive(self, mock_llm_function_class):
        """Lowercased verdict markers must still be parsed.

        Both LLM engines in this repo rewrite "True"/"False" to lowercase in
        every response they return, so a judge emitting "[[False]]" reaches the
        checker as "[[false]]".
        """
        mock_judge = MagicMock()
        mock_llm_function_class.return_value = mock_judge

        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=self.mock_prompt_templates,
            num_votes=1,
            success_str="[[True]]",
            failure_str="[[False]]",
        )

        mock_judge.return_value = "Reasoning, final verdict: [[false]]"
        self.assertIs(checker({"user_arg": "test"}), False)

        mock_judge.return_value = "Reasoning, final verdict: [[true]]"
        self.assertIs(checker({"user_arg": "test"}), True)


class TestLLMCheckerWithRealEngine(unittest.TestCase):
    """End-to-end: the engine's own response rewriting must not blind the checker."""

    def _engine(self, judge_response: str):
        from are.simulation.agents.llm.litellm.litellm_engine import (
            LiteLLMEngine,
            LiteLLMModelConfig,
        )

        engine = LiteLLMEngine(LiteLLMModelConfig(model_name="gpt-4o", provider="mock"))
        engine.mock_response = judge_response
        return engine

    def test_engine_lowercases_verdict_marker(self):
        engine = self._engine("Final verdict: [[False]]")
        response, _ = engine([{"role": "user", "content": "x"}])
        # Documents the root cause this fix has to tolerate.
        self.assertIn("[[false]]", response)

    def test_checker_parses_verdict_returned_by_engine(self):
        for emitted, expected in (
            ("The response is not coherent.\nFinal verdict: [[False]]", False),
            ("The response is coherent.\nFinal verdict: [[True]]", True),
        ):
            with self.subTest(emitted=emitted):
                checker = LLMChecker(
                    engine=self._engine(emitted),
                    prompt_templates=SANITY_CHECKER_PROMPT_TEMPLATES,
                    num_votes=1,
                    success_str="[[True]]",
                    failure_str="[[False]]",
                )
                verdict = checker(
                    {"task": "Send a message to the user", "agent_action_call": "..."}
                )
                self.assertIsNotNone(
                    verdict, "checker abstained on a well-formed judge verdict"
                )
                self.assertIs(verdict, expected)


class TestBuildLLMCheckers(unittest.TestCase):
    def setUp(self):
        self.mock_engine = MagicMock()

    @patch("are.simulation.validation.utils.llm_utils.LLMChecker")
    def test_build_llm_checkers_default_params(self, mock_llm_checker_class):
        mock_checker_instances = []
        for _ in range(9):  # 9 checkers created
            mock_checker = MagicMock()
            mock_checker_instances.append(mock_checker)

        mock_llm_checker_class.side_effect = mock_checker_instances

        result = build_llm_checkers(self.mock_engine)

        # Check all expected checkers are in the result
        expected_keys = [
            SoftCheckerType.signature_checker.value,
            SoftCheckerType.sanity_checker.value,
            SoftCheckerType.cab_checker.value,
            SoftCheckerType.content_checker.value,
            SoftCheckerType.email_checker.value,
            SoftCheckerType.message_checker.value,
            SoftCheckerType.user_message_checker.value,
            SoftCheckerType.event_checker.value,
            SoftCheckerType.tone_checker.value,
        ]

        for key in expected_keys:
            self.assertIn(key, result)

        # Check that 9 LLMChecker instances were created
        self.assertEqual(mock_llm_checker_class.call_count, 9)

        # Verify specific checker configurations
        calls = mock_llm_checker_class.call_args_list

        # Check signature checker (first call)
        signature_call = calls[0]
        self.assertEqual(signature_call[1]["success_str"], "[[True]]")
        self.assertEqual(signature_call[1]["failure_str"], "[[False]]")
        self.assertEqual(signature_call[1]["num_votes"], 1)

        # Check content checker (third call)
        content_call = calls[2]
        self.assertEqual(content_call[1]["success_str"], "[[Success]]")
        self.assertEqual(content_call[1]["failure_str"], "[[Failure]]")
        self.assertEqual(content_call[1]["num_votes"], 1)

    @patch("are.simulation.validation.utils.llm_utils.LLMChecker")
    def test_build_llm_checkers_custom_num_votes(self, mock_llm_checker_class):
        mock_checker_instances = []
        for _ in range(9):
            mock_checker = MagicMock()
            mock_checker_instances.append(mock_checker)

        mock_llm_checker_class.side_effect = mock_checker_instances

        _ = build_llm_checkers(self.mock_engine, num_votes=3)

        # Check calls for checkers that should use custom num_votes
        calls = mock_llm_checker_class.call_args_list

        # Content checker should use custom num_votes
        content_call = calls[2]
        self.assertEqual(content_call[1]["num_votes"], 3)

        # Email checker should use custom num_votes
        email_call = calls[4]
        self.assertEqual(email_call[1]["num_votes"], 3)

        # Signature checker should still use 1 (hardcoded)
        signature_call = calls[0]
        self.assertEqual(signature_call[1]["num_votes"], 1)

    @patch("are.simulation.validation.utils.llm_utils.LLMChecker")
    def test_build_llm_checkers_custom_templates(self, mock_llm_checker_class):
        mock_checker = MagicMock()
        mock_llm_checker_class.return_value = mock_checker

        custom_content_templates = MagicMock()

        _ = build_llm_checkers(
            self.mock_engine,
            content_checker_prompt_templates=custom_content_templates,
        )

        # Find the content checker call
        calls = mock_llm_checker_class.call_args_list
        content_call = calls[2]  # Content checker is the third one created

        self.assertEqual(content_call[1]["prompt_templates"], custom_content_templates)


class TestBuildSubtaskExtractor(unittest.TestCase):
    def setUp(self):
        self.mock_engine = MagicMock()

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_build_subtask_extractor_default(self, mock_llm_function_class):
        mock_extractor = MagicMock()
        mock_llm_function_class.return_value = mock_extractor

        result = build_subtask_extractor(self.mock_engine, "generic_tool")

        self.assertEqual(result, mock_extractor)
        mock_llm_function_class.assert_called_once_with(
            engine=self.mock_engine,
            prompt_templates=SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
        )

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_build_subtask_extractor_calendar_event(self, mock_llm_function_class):
        mock_extractor = MagicMock()
        mock_llm_function_class.return_value = mock_extractor

        result = build_subtask_extractor(self.mock_engine, "add_calendar_event")

        self.assertEqual(result, mock_extractor)
        mock_llm_function_class.assert_called_once_with(
            engine=self.mock_engine,
            prompt_templates=EVENT_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
        )

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_build_subtask_extractor_user_message(self, mock_llm_function_class):
        mock_extractor = MagicMock()
        mock_llm_function_class.return_value = mock_extractor

        result = build_subtask_extractor(
            self.mock_engine, "AgentUserInterface__send_message_to_user"
        )

        self.assertEqual(result, mock_extractor)
        mock_llm_function_class.assert_called_once_with(
            engine=self.mock_engine,
            prompt_templates=USER_MESSAGE_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
        )

    @patch("are.simulation.validation.utils.llm_utils.LLMFunction")
    def test_build_subtask_extractor_calendar_event_partial_match(
        self, mock_llm_function_class
    ):
        """Test that 'add_calendar_event' substring matching works"""
        mock_extractor = MagicMock()
        mock_llm_function_class.return_value = mock_extractor

        result = build_subtask_extractor(
            self.mock_engine, "CalendarApp__add_calendar_event"
        )

        self.assertEqual(result, mock_extractor)
        mock_llm_function_class.assert_called_once_with(
            engine=self.mock_engine,
            prompt_templates=EVENT_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES,
        )


class TestLLMFunctionIntegration(unittest.TestCase):
    """Integration tests using real prompt templates"""

    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_engine.return_value = ("Test response", None)

    def test_llm_function_with_real_content_checker_templates(self):
        llm_function = LLMFunction(
            engine=self.mock_engine,
            prompt_templates=CONTENT_CHECKER_PROMPT_TEMPLATES,
        )

        # Test that it initializes without errors
        self.assertIsNotNone(llm_function.system_prompt)
        self.assertIsNotNone(llm_function.user_prompt_template)

        # Test calling with expected arguments
        user_args = {
            "agent_action_call": "test agent call",
            "oracle_action_call": "test oracle call",
            "tool_name": "test_tool",
            "task": "test task",
            "today_date": "2023-01-01",
            "user_address": "123 Test St",
        }

        result = llm_function(user_args)
        self.assertEqual(result, "Test response")
        self.mock_engine.assert_called_once()

    def test_llm_checker_with_real_sanity_checker_templates(self):
        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=SANITY_CHECKER_PROMPT_TEMPLATES,
            success_str="[[True]]",
            failure_str="[[False]]",
        )

        # Test that it initializes without errors
        self.assertIsNotNone(checker.judge)

        # Mock a successful response
        self.mock_engine.return_value = ("[[True]]", None)

        user_args = {
            "task": "Test task",
            "agent_action_call": "Test response",
        }

        result = checker(user_args)
        self.assertTrue(result)

    def test_llm_checker_with_real_signature_checker_templates(self):
        checker = LLMChecker(
            engine=self.mock_engine,
            prompt_templates=SIGNATURE_CHECKER_TEMPLATES,
            success_str="[[True]]",
            failure_str="[[False]]",
        )

        # Test that it initializes with examples
        self.assertIsNotNone(checker.judge)
        self.assertGreater(len(checker.judge.examples), 0)

        # Mock a failure response
        self.mock_engine.return_value = ("[[False]]", None)

        user_args = {
            "user_name": "John Doe",
            "agent_action_call": "subject: Test\ncontent: Best regards,\nAssistant",
        }

        result = checker(user_args)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
