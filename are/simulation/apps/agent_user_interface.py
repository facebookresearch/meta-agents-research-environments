# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from are.simulation.agents.user_proxy import UserProxy
from are.simulation.apps.app import App
from are.simulation.tool_utils import app_tool, data_tool, OperationType, user_tool
from are.simulation.types import event_registered, EventType
from are.simulation.utils import from_dict, get_state_dict, type_check

logger = logging.getLogger(__name__)


NO_RESPONSE_FROM_USER_DEFAULT_MESSAGE = "No response received from the User."


class Sender(Enum):
    USER = "User"
    AGENT = "Agent"


@dataclass
class AUIMessage:
    sender: Sender
    content: str
    attachments: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: time.time())
    time_read: float | None = None
    id: str | None = None

    @property
    def already_read(self) -> bool:
        return self.time_read is not None

    def __str__(self):
        timestamp_recv_str = datetime.fromtimestamp(
            self.timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        s = f"Received at: {timestamp_recv_str}\nSender: {self.sender.value}\nMessage: {self.content}"
        s += f"\nAlready read: {self.already_read}"
        if self.time_read is not None:
            timestamp_read_str = datetime.fromtimestamp(
                self.time_read, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
            s += f"\nRead at: {timestamp_read_str}"
        return s


class AgentUserInterface(App):
    """
    The Agent User Interface (AUI) is a tool for facilitating interactions between a user and an agent.
    This interface provides methods to exchange messages, retrieve conversation history, and manage
    the flow of communication, including support for simulated user responses through a proxy.

    The AgentUserInterface manages message exchanges in a list structure, storing messages from both the user and
    the agent, which can be accessed and filtered by various criteria. It allows for message persistence, tracking
    unread messages, and retrieving conversation history, along with environment control for CLI or GUI-based
    interactions.

    Key Features:
        - Message Exchange: Send and retrieve messages from user and agent
        - User Proxy Support: Simulate user responses using an optional proxy for testing or automation
        - Unread Message Tracking: Retrieve unread messages from conversation history
        - Environment Control: Pause and resume environment during user-agent interactions
        - State Management: Access conversation state with methods to fetch and filter messages

    Notes:
        - Messages are stored in a list and ordered chronologically
        - Supports CLI or GUI-based environments for real-time user input handling
        - The user proxy allows simulated responses for automated testing without real user input
        - Methods provide options to retrieve last messages or all messages by sender (user or agent)
        - State can be retrieved as a dictionary, preserving message history and interaction details
    """

    def __init__(
        self,
        user_proxy: UserProxy | None = None,
    ):
        """
        Initializes the Agent User Interface.
        :param user_proxy: The user proxy to use for simulating user responses.
        """
        super().__init__()
        self.messages: list[AUIMessage] = []
        self.user_proxy = user_proxy
        self.is_cli = True
        # When wait_for_user_response=False, all the user messages are actually passed to the Agent directly in the history
        # So no need to return a response from the user
        # When the Agent sends a message, the Agent will stop ending the task, and when response is received, Agent is going to start again
        self.wait_for_user_response = True

    def set_wait_for_user_response(self, wait_for_user_response: bool):
        self.wait_for_user_response = wait_for_user_response

    def get_state(self) -> dict[str, Any]:
        return get_state_dict(self, ["messages"])

    def reset(self):
        super().reset()
        self.messages.clear()

    def load_state(self, state_dict: dict[str, Any]):
        self.messages = []
        timestamp_offset = float("inf")
        for message_state in state_dict["messages"]:
            message: AUIMessage = from_dict(
                AUIMessage, message_state
            )  # pyright: ignore[reportAssignmentType]/
            timestamp_offset = min(timestamp_offset, message.timestamp)
            if message.time_read:
                timestamp_offset = min(timestamp_offset, message.time_read)
            self.messages.append(message)

        for message in self.messages:
            message.timestamp -= timestamp_offset
            if message.time_read:
                message.time_read -= timestamp_offset

    def set_cli(self, is_cli: bool):
        self.is_cli = is_cli

    def add_message_to_user(self, content: str) -> None:
        """
        Adds a message to be displayed to the user. This function does not
        generate an event, it just modifies the app state. This function is
        intended to be invoked from other app functions to display results from
        those events as user messages.
        :param content: The message content to send to the user.
        """
        self._send_message(Sender.AGENT, content)

    @type_check
    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def send_message_to_user(self, content: str = "") -> str | None:
        """
        Sends a message to the User. This will end the Agent's turn.
        :param content: the content to send to the user
        :returns: The response from the user
        """
        self._send_message(Sender.AGENT, content)
        previous_nb_user_messages = len(
            [msg for msg in self.messages if msg.sender == Sender.USER]
        )

        if self.user_proxy is not None:
            # If a user proxy is set, we need to make sure the user proxy is called
            # And it's message generated and sent to the agent in all cases
            user_response = self._get_user_proxy_reply(content)
            self.send_message_to_agent(user_response)
        else:
            user_response = None

        if not self.wait_for_user_response:
            # If we don't wait for user response, we don't need to return a response
            # But even in this case, we still need to make sure the user proxy is called
            return None

        if self.is_cli:
            # Inside the CLI we cannot wait for user input, so we just return the message of the UserProxy
            # Or if not UserProxy, we just return None
            return user_response
        else:
            # Wait for the user to respond in case of the UI
            # By checking the number of messages, we can wait until it changed
            # This also handles the case where the UserProxy sent a message to the Agent
            while (
                len([msg for msg in self.messages if msg.sender == Sender.USER])
                == previous_nb_user_messages
            ):
                logger.debug(
                    f"Waiting for user response - still {previous_nb_user_messages} ..."
                )
                time.sleep(1.0)

            # Get the last messages from the user
            new_messages = self.messages[previous_nb_user_messages:]
            new_user_messages = [
                msg for msg in new_messages if msg.sender == Sender.USER
            ]
            logger.info(f"Received {len(new_user_messages)} messages from the user")
            for message in new_user_messages:
                message.time_read = self.time_manager.time()
            return "\n".join([str(msg) for msg in new_user_messages])

    @type_check
    @user_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.USER)
    def send_message_to_agent(
        self,
        content: str = "",
        attachments: list[str] | None = None,
        base64_utf8_encoded_attachment_contents: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Sends a message to the Agent.
        :param content: the content to send to the agent
        :param attachments: optional; a list of attachments to send to the agent, each of them is a URL to a file
        :param base64_utf8_encoded_attachment_contents: optional; attachment contents corresponding to the attachments parameter. An entry may be an empty dict if an empty file was passed in.
        :return: The message ID that was generated for this message, can be used for tracking
        """
        if attachments is None:
            attachments = []
        if base64_utf8_encoded_attachment_contents is None:
            base64_utf8_encoded_attachment_contents = []
        message_id = self._send_message(Sender.USER, content, attachments)
        return message_id

    def _send_message(
        self, sender: Sender, content: str = "", attachments: list[str] | None = None
    ) -> str:
        """
        Sends a message from sender.
        :param sender: the sender of the message
        :param content: the content of the message
        :param attachments: optional; a list of attachments
        :return: The message ID that was generated for this message
        """
        if attachments is None:
            attachments = []
        timestamp = self.time_manager.time()
        # Always generate a new UUID on the backend for consistency
        message_id = str(uuid.uuid4())
        message = AUIMessage(
            sender=sender,
            content=content,
            attachments=attachments,
            timestamp=timestamp,
            id=message_id,
        )
        if sender == Sender.AGENT:
            message.time_read = timestamp
        self.messages.append(message)
        return message_id

    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_last_message_from_user(self) -> AUIMessage | None:
        """
        Retrieves the last message that was sent from the User to the Agent.
        :returns: the last message from the user or None if no messages have been sent
        """
        user_messages = [msg for msg in self.messages if msg.sender == Sender.USER]
        if user_messages:
            user_message = user_messages[-1]
            if not user_message.already_read:
                user_message.time_read = self.time_manager.time()
            return user_message
        return None

    @app_tool(
        llm_formatter=lambda msg_list: "\n\n".join([str(msg) for msg in msg_list])
    )
    @data_tool(
        llm_formatter=lambda msg_list: "\n\n".join([str(msg) for msg in msg_list])
    )
    @event_registered(operation_type=OperationType.READ)
    def get_last_unread_messages(self) -> list[AUIMessage]:
        """
        Retrieves all unread messages from the User and Agent conversation.
        :returns: a list of all unread messages or an empty list if all messages have been read
        """
        unread_user_messages = [
            msg
            for msg in self.messages
            if msg.sender == Sender.USER and not msg.already_read
        ]
        for message in unread_user_messages:
            message.time_read = self.time_manager.time()

        return unread_user_messages

    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_last_message_from_agent(self) -> AUIMessage | None:
        """
        Retrieves the last message that was sent from the Agent to the User.
        :returns: the last message from the agent or None if no messages have been sent
        """
        agent_messages = [msg for msg in self.messages if msg.sender == Sender.AGENT]
        if agent_messages:
            return agent_messages[-1]
        return None

    @app_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_all_messages(self) -> list[AUIMessage]:
        """
        Retrieves all messages from the User and Agent conversation.
        :returns: a list of all messages
        """
        # Order by timestamp most recent first
        for message in self.messages:
            message.time_read = self.time_manager.time()
        return [
            message
            for message in sorted(
                self.messages, key=lambda x: x.timestamp, reverse=True
            )
        ]

    def send_user_message_to_agent(
        self,
        content: str,
        pause_func: Callable,
        wait_for_user_input_timeout: float | None = None,
    ):
        """
        Send a user response to the agent.
        :param content: the message content to be sent from the user to the agent
        :param environment_type: the environment type
        :param pause_func: the function to pause the environment
        :param wait_for_user_input_timeout: optional; the maximum time in seconds to wait for user input. Specify `0` to not wait, or `None` to wait indefinitely
        :returns: None
        """
        wait_for_user_input = (
            wait_for_user_input_timeout is not None and wait_for_user_input_timeout > 0
        )
        if self.is_cli and wait_for_user_input:
            # Wait for user console input.
            user_response = self._get_user_cli_reply(
                content,
                wait_for_user_input_timeout,
            )
            self.send_message_to_agent(user_response)

        elif not self.is_cli and wait_for_user_input:
            # Pause the environment and wait for user input through API.
            pause_func()

        elif self.user_proxy is not None:
            # Simulate user response through user proxy.
            user_proxy_response = self._get_user_proxy_reply(content)
            self.send_message_to_agent(user_proxy_response)

    def get_all_messages_from_user(self) -> list[AUIMessage]:
        """
        Retrieves all messages from the User.
        :returns: a list of all messages as strings
        """
        # order by timestamp most recent first
        return [
            message
            for message in sorted(
                self.messages, key=lambda x: x.timestamp, reverse=True
            )
            if message.sender == Sender.USER
        ]

    def get_all_messages_from_agent(self) -> list[AUIMessage]:
        """
        Retrieves all messages from Agent.
        :returns: a list of all messages as strings
        """
        # order by timestamp most recent first
        return [
            message
            for message in sorted(
                self.messages, key=lambda x: x.timestamp, reverse=True
            )
            if message.sender == Sender.AGENT
        ]

    def get_last_message_content(self) -> str:
        """
        Retrieves the last message content.
        :returns: the content of the last message
        """
        return self.messages[-1].content if self.messages else ""

    def get_last_message(self) -> AUIMessage | None:
        """
        Retrieves the last message.
        :returns: the last message
        """
        return self.messages[-1] if self.messages else None

    def pause_env(self):
        """
        Pauses the environment, this is used to stop the environment from running while the user is interacting with the Agent.
        """

    def resume_env(self):
        """
        Resumes the environment, this is used to resume the environment after the user is done interacting with the Agent.
        """

    def _get_user_cli_reply(self, message: str, timeout: float | None) -> str:
        """
        Get the reply of the user using the console input
        """
        user_message = textwrap.dedent(
            f"""
        🤖 Agent has sent you a message. Please read it and respond.
        Message: {message}
        """
        )
        self.pause_env()
        user_input = countdown_input(user_message, timeout)
        self.resume_env()
        return user_input

    def _get_user_proxy_reply(self, message: str) -> str:
        """
        Get the reply of the user using the user proxy
        """
        if self.user_proxy is not None:
            if message is not None:
                return self.user_proxy.reply(message)
            else:
                return self.user_proxy.init_conversation()
        else:
            return ""


def countdown_input(prompt: str, timeout: float | None):
    if timeout is None:
        message = colored(f"\n\n{prompt}>>>", "red")
    else:
        message = colored(
            f"\n\n{prompt} (You have {timeout} seconds to answer):\n>>>", "red"
        )
    print(message)
    try:
        if timeout is None:
            user_input = input()
        else:
            user_input = inputimeout(timeout=timeout)
        return user_input
    except TimeoutOccurred:
        return NO_RESPONSE_FROM_USER_DEFAULT_MESSAGE
