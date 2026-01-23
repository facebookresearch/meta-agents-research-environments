# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import base64
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from are.simulation.apps.app import App, Protocol
from are.simulation.apps.sandbox_file_system import SandboxLocalFileSystem
from are.simulation.apps.virtual_file_system import VirtualFileSystem
from are.simulation.tool_utils import OperationType, app_tool, data_tool, env_tool
from are.simulation.types import EventType, disable_events, event_registered
from are.simulation.utils import get_state_dict, type_check, uuid_hex

logger = logging.getLogger(__name__)


class EmailFolderName(Enum):
    INBOX = "INBOX"
    SENT = "SENT"
    DRAFT = "DRAFT"
    TRASH = "TRASH"


@dataclass
class Email:
    sender: str = "user@meta.com"
    recipients: list[str] = field(default_factory=list)
    subject: str = ""
    content: str = ""
    email_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    cc: list[str] = field(default_factory=list)
    attachments: dict[str, bytes] | None = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
    is_read: bool = field(default=False)

    def __str__(self):
        return f"From: {self.sender}\nTo: {', '.join(self.recipients)}\nCC: {', '.join(self.cc)}\nSubject: {self.subject}\nContent: {self.content}"

    @property
    def summary(self):
        return f"From: {self.sender}\nSubject: {self.subject}"

    def __post_init__(self):
        if self.email_id is None or len(self.email_id) == 0:
            self.email_id = uuid.uuid4().hex

        if "@" not in self.sender:
            raise ValueError(f"Invalid email address: {self.sender}")

        for recipient in self.recipients:
            if "@" not in recipient:
                raise ValueError(f"Invalid email address: {recipient}")

        if self.cc is None:
            self.cc = []

        for cc in self.cc:
            if "@" not in cc:
                raise ValueError(f"Invalid email address: {cc}")

    def add_attachment(self, path: str):
        assert isinstance(path, str), "Path must be a string."
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        with open(path, "rb") as f:
            file_content = base64.b64encode(f.read())
            file_name = os.path.basename(path)
            if not self.attachments:
                self.attachments = {}
            self.attachments[file_name] = file_content


@dataclass
class ReturnedEmails:
    emails: list[Email]
    emails_range: tuple[int, int]
    total_returned_emails: int
    total_emails: int


class EmailFolder:
    def __init__(self, folder_name: EmailFolderName):
        assert isinstance(folder_name, EmailFolderName), (
            "Folder name must be an instance of EmailFolderName"
        )
        self.folder_name: EmailFolderName = folder_name
        self.emails: list[Email] = []

    def add_email(self, email: Email):
        assert isinstance(email, Email), "Email must be an instance of Email"
        self.emails.append(email)
        self.emails = sorted(self.emails, key=lambda x: x.timestamp, reverse=True)

    def get_emails(self, offset: int = 0, limit: int = 5) -> ReturnedEmails:
        assert isinstance(offset, int), "Offset must be an integer."
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        if offset > len(self.emails):
            raise ValueError("Offset must be less than the number of emails")

        start_idx = offset
        end_idx = min(offset + limit, len(self.emails))
        return ReturnedEmails(
            emails=self.emails[start_idx:end_idx],
            emails_range=(start_idx, end_idx),
            total_returned_emails=end_idx - start_idx,
            total_emails=len(self.emails),
        )

    def get_email(self, idx: int = 0) -> Email:
        assert isinstance(idx, int), "Index must be an integer."
        if int(idx) > len(self.emails):
            raise ValueError(f"Email with index {idx} does not exist")
        return self.emails[int(idx)]

    def get_email_by_id(self, email_id: str) -> Email:
        assert isinstance(email_id, str), "Email ID must be a string."
        for email in self.emails:
            if email.email_id == email_id:
                return email
        raise ValueError(f"Email with id {email_id} does not exist")

    def get_state(self) -> dict[str, Any]:
        return get_state_dict(self, ["folder_name", "emails"])

    def load_state(self, state_dict: dict[str, Any]):
        temp_folder_name = state_dict["folder_name"]
        self.folder_name = getattr(EmailFolderName, temp_folder_name)
        self.emails = [Email(**email) for email in state_dict["emails"]]


@dataclass
class EmailClientApp(App):
    """
    An email client application that manages email operations and folder organization. This class provides comprehensive
    functionality for handling emails including sending, receiving, organizing, and managing email attachments.

    The app maintains emails in different folders (INBOX, SENT, DRAFT, TRASH) and provides various operations
    for email manipulation and searching.

    Key Features:
    - Email Management: Send, reply, move, and delete emails
    - Folder Organization: Maintains separate folders for different email categories
    - Attachment Support: Handle email attachments (upload and download)
    - Search Functionality: Search emails across folders with text-based queries
    - State Management: Save and load application state

    Key Components:
    - Folders: Each EmailFolder instance (INBOX, SENT, DRAFT, TRASH) maintains its own collection of emails
    - Email Validation: Validates email addresses using regex pattern matching
    - View Limits: Configurable limit for email viewing and pagination
    - Event Registration: All operations are tracked through event registration

    Notes:
    - Email IDs are automatically generated when creating new emails
    - Attachments are handled using base64 encoding
    - Search operations are case-insensitive
    - All email operations maintain folder integrity
    - Supports CC recipients and multiple attachments
    """

    name: str | None = None
    view_limit: int = 5
    folders: dict[EmailFolderName, EmailFolder] = field(default_factory=dict)
    user_email: str = "user@meta.com"

    def __post_init__(self):
        super().__init__(self.name, user_email=self.user_email)
        for folder_name in EmailFolderName:
            if folder_name not in self.folders:
                self.folders[folder_name] = EmailFolder(folder_name)
        email_regex = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+$")
        if not email_regex.match(self.user_email):
            raise ValueError(f"Invalid email address: {self.user_email}")

    def get_state(self) -> dict[str, Any] | None:
        result = {
            "user_email": self.user_email,
            "view_limit": self.view_limit,
            "folders": {k.value: v.get_state() for k, v in self.folders.items()},
        }
        return result

    def load_state(self, state_dict: dict[str, Any]):
        self.user_email = state_dict["user_email"]
        self.view_limit = state_dict["view_limit"]
        for folder in state_dict["folders"]:
            email_folder = EmailFolder(EmailFolderName.INBOX)
            email_folder.load_state(state_dict["folders"][folder])
            self.folders[email_folder.folder_name] = email_folder

    def reset(self):
        super().reset()
        for folder in self.folders:
            self.folders[folder].emails.clear()

    @event_registered(operation_type=OperationType.WRITE)
    def add_email(
        self, email: Email, folder_name: EmailFolderName = EmailFolderName.INBOX
    ) -> str:
        """
        Adds an email to the specified folder.
        :param email: The email to add.
        :param folder_name: The folder to add the email to.
        """
        assert isinstance(email, Email), "Email must be an instance of Email"
        assert isinstance(folder_name, EmailFolderName), (
            "Folder name must be an instance of EmailFolderName"
        )
        self.folders[folder_name].add_email(email)
        return email.email_id

    @event_registered(operation_type=OperationType.WRITE)
    def send_email_to_user(self, email: Email) -> str:
        assert isinstance(email, Email), "Email must be an instance of Email"
        self.folders[EmailFolderName.INBOX].add_email(email)
        return email.email_id

    @env_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.ENV)
    def send_email_to_user_only(
        self,
        sender: str,
        subject: str = "",
        content: str = "",
    ) -> str:
        """
        Creates an email with the user as sole recipient and adds it to the user's INBOX folder.
        :param sender: The sender of the email.
        :param subject: The subject of the email.
        :param content: The content of the email.
        :returns: The id of the email just created.
        """
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=sender,
            recipients=[self.user_email],
            subject=subject,
            content=content,
            timestamp=self.time_manager.time(),
            is_read=False,
        )
        self.folders[EmailFolderName.INBOX].add_email(email)
        return email.email_id

    @type_check
    @env_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.ENV)
    def reply_to_email_from_user(
        self,
        sender: str,
        email_id: str,
        content: str = "",
        attachment_paths: list[str] | None = None,
    ) -> str:
        """
        Creates a reply to an email from the user and adds it to the user's INBOX folder.
        :param sender: The sender of the email.
        :param email_id: The id of the email to reply to, the Id is specified in the email details.
        :param content: The content of the reply.
        :param attachment_paths: The paths of the attachments to add to the reply.
        :returns: The id of the email just created and sent.

        :example:
            reply_to_email_from_user("other_user@meta.com", "1234567890abcdef", "Hello, this is a reply to your email.")
        """
        if attachment_paths is None:
            attachment_paths = []

        replying_to_email = self.folders[EmailFolderName.SENT].get_email_by_id(email_id)
        if sender not in replying_to_email.recipients:
            raise ValueError(
                f"Sender {sender} is not a recipient of the email, the recipients are {replying_to_email.recipients}"
            )
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=sender,
            recipients=[self.user_email],
            subject="Re: " + replying_to_email.subject,
            content=content,
            timestamp=self.time_manager.time(),
            cc=[],
            parent_id=replying_to_email.email_id,
        )
        for path in attachment_paths:
            email.add_attachment(path)

        with disable_events():
            self.add_email(email=email, folder_name=EmailFolderName.INBOX)
        return email.email_id

    @env_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.ENV)
    def create_and_add_email(
        self,
        sender: str,
        recipients: list[str] | None = None,
        subject: str = "",
        content: str = "",
        folder_name: str = "INBOX",
    ) -> str:
        """
        Create and add an email to the specified folder.
        :param sender: The sender of the email.
        :param recipients: The recipients of the email.
        :param subject: The subject of the email.
        :param content: The content of the email.
        :param folder_name: The folder to add the email to. Can only be one of INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :returns: The id of the email just created.
        """
        if recipients is None:
            recipients = []
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=sender,
            recipients=recipients,
            subject=subject,
            content=content,
            timestamp=self.time_manager.time(),
            is_read=False,
        )
        folder_name_enum = EmailFolderName(folder_name)
        self.folders[folder_name_enum].add_email(email)
        return email.email_id

    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def create_and_add_email_with_time(
        self,
        sender: str,
        recipients: list[str] | None = None,
        subject: str = "",
        content: str = "",
        email_time: str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        folder_name: str = "INBOX",
    ) -> str:
        """
        Create and add an email to the specified folder.
        :param sender: The sender of the email.
        :param recipients: The recipients of the email.
        :param subject: The subject of the email.
        :param content: The content of the email.
        :param folder_name: The folder to add the email to. Can only be one of INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :param email_time: The time of the email. Defaults to the current time.
        :returns: The id of the email just created.
        """

        if recipients is None:
            recipients = []
        try:
            timestamp = (
                datetime.strptime(email_time, "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        except ValueError:
            raise ValueError(
                "Invalid datetime format for the email time. Please use YYYY-MM-DD HH:MM:SS"
            )

        email = Email(
            email_id=uuid_hex(self.rng),
            sender=sender,
            recipients=recipients,
            subject=subject,
            content=content,
            timestamp=timestamp,
            is_read=True,
        )
        folder_name_enum = EmailFolderName(folder_name)
        self.folders[folder_name_enum].add_email(email)
        return email.email_id

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def list_emails(
        self,
        folder_name: str = EmailFolderName.INBOX.value,
        offset: int = 0,
        limit: int = 10,
    ) -> ReturnedEmails:
        """
        Lists emails in the specified folder with a specified offset.
        :param folder_name: The folder to list emails from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :param offset: The offset of the first email to return.
        :param limit: The maximum number of emails to return.
        :returns: Emails with additional metadata about the range of emails retrieved and total number of emails

        :example:
            list_emails("INBOX", 0, 10)
        """
        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")

        return self.folders[folder].get_emails(offset, limit)

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_email_by_id(
        self,
        email_id: str,
        folder_name: str = EmailFolderName.INBOX.value,
    ) -> Email:
        """
        Reads an email from the specified folder, marking it as read.
        :param email_id: The id of the email to read, the Id is specified in the email.
        :param folder_name: The folder to read the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :returns: Email details if successful, otherwise raise IndexError.

        :example:
            get_email_by_id("1234567890abcdef")
        """
        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")

        if email_id is not None:
            email = self.folders[folder].get_email_by_id(email_id)
        else:
            raise ValueError("Must provide either email_id")

        if type(email) is Email:
            email.is_read = True
        return email

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def get_email_by_index(
        self,
        idx: int,
        folder_name: str = EmailFolderName.INBOX.value,
    ) -> Email:
        """
        Reads an email from the specified folder, marking it as read.
        :param idx: The index of the email to read among the list of emails in the folder.
        :param folder_name: The folder to read the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :returns: Email details if successful, otherwise raise IndexError.

        :example:
            get_email_by_index(0, "INBOX")
        """
        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")

        if idx is not None:
            email = self.folders[folder].get_email(idx)
        else:
            raise ValueError("Must provide either email_id or idx")

        if type(email) is Email:
            email.is_read = True
        return email

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def send_email(
        self,
        recipients: list[str] | None = None,
        subject: str = "",
        content: str = "",
        cc: list[str] | None = None,
        attachment_paths: list[str] | None = None,
    ) -> str:
        """
        Sends an email to the specified recipients.
        :param recipients: The recipients of the email.
        :param subject: The subject of the email.
        :param content: The content of the email.
        :param cc: The cc of the email.
        :param attachment_paths: The paths of the attachments to add to the email.
        :returns: The id of the email just created.

        :example:
            send_email(["user1@meta.com", "user2@meta.com"], "Hello", "Hi there", ["user3@meta.com"], ["tmp123/file1.txt", "tmp234/file2.txt"])
        """
        if recipients is None:
            recipients = []
        if cc is None:
            cc = []
        if attachment_paths is None:
            attachment_paths = []
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=self.user_email,
            recipients=recipients,
            subject=subject,
            content=content,
            timestamp=self.time_manager.time(),
            cc=cc,
        )
        for path in attachment_paths:
            email.add_attachment(path)

        # This is to avoid getting an event for add_email method
        with disable_events():
            self.add_email(email=email, folder_name=EmailFolderName.SENT)
        return email.email_id

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def forward_email(
        self,
        email_id: str,
        recipients: list[str] | None = None,
        folder_name: str = EmailFolderName.INBOX.value,
    ) -> str:
        """
        Forwards an email to the specified recipients.
        :param email_id: The id of the email to forward.
        :param recipients: The recipients of the email.
        :param folder_name: The folder to forward the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :returns: The id of the email just created and sent.

        :example:
            forward_email("1234567890abcdef", ["user1@meta.com", "user2@meta.com"], "INBOX")
        """
        if recipients is None:
            recipients = []
        email = self.get_email_by_id(email_id, folder_name=folder_name)

        new_email = Email(
            email_id=uuid_hex(self.rng),
            sender=self.user_email,
            recipients=recipients,
            subject="FWD: " + email.subject,
            content="> " + email.content,
            timestamp=self.time_manager.time(),
            parent_id=email.email_id,
            attachments=email.attachments,
            cc=[],
        )
        with disable_events():
            self.add_email(email=new_email, folder_name=EmailFolderName.SENT)
        return new_email.email_id

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def download_attachments(
        self,
        email_id: str,
        folder_name: str = EmailFolderName.INBOX.value,
        path_to_save: str = "Downloads/",
    ) -> list[str]:
        """
        Downloads attachments from an email.
        :param email_id: The id of the email to download.
        :param folder_name: The folder to download the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :param path_to_save: The path to save the attachments to.
        :returns: The list of  full path to the downloaded attachments.

        :example:
            download_attachments("1234567890abcdef", "INBOX", "Downloads/")
        """
        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")

        email = self.folders[folder].get_email_by_id(email_id)
        filenames = []
        if not email.attachments:
            return filenames
        for file_name, file_content in email.attachments.items():
            with open(os.path.join(path_to_save, file_name), "wb") as f:
                file_data = base64.b64decode(file_content)
                f.write(file_data)
                filenames.append(os.path.join(path_to_save, file_name))
        return filenames

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def reply_to_email(
        self,
        email_id: str,
        folder_name: str = EmailFolderName.INBOX.value,
        content: str = "",
        attachment_paths: list[str] | None = None,
    ) -> str:
        """
        Replies to an email.
        :param email_id: The id of the email to reply to, the Id is specified in the email details.
        :param folder_name: The folder to reply to the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :param content: The content of the reply.
        :param attachment_paths: The paths of the attachments to add to the reply.
        :returns: The id of the email just created and sent.

        :example:
            reply_to_email("1234567890abcdef", "INBOX", "Hi there", ["tmp123/file1.txt", "tmp234/file2.txt"])
        """

        if attachment_paths is None:
            attachment_paths = []

        def get_recipient(email: Email) -> str:
            while email.sender == self.user_email and email.parent_id:
                email_found = False
                for folder in self.folders:
                    try:
                        email = self.folders[folder].get_email_by_id(email.parent_id)
                        email_found = True
                        break
                    except Exception:
                        logger.error("Email not found in ", folder)
                if not email_found:
                    raise ValueError(f"Email with id {email.parent_id} not found")
            return email.sender

        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")

        replying_to_email = self.folders[folder].get_email_by_id(email_id)
        recipient = get_recipient(replying_to_email)
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=self.user_email,
            recipients=[recipient],
            subject="Re: " + replying_to_email.subject,
            content=content,
            timestamp=self.time_manager.time(),
            cc=[],
            parent_id=replying_to_email.email_id,
        )
        for path in attachment_paths:
            email.add_attachment(path)

        with disable_events():
            self.add_email(email=email, folder_name=EmailFolderName.SENT)
        return email.email_id

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def move_email(
        self,
        email_id: str,
        source_folder_name: str = EmailFolderName.INBOX.value,
        dest_folder_name: str = EmailFolderName.DRAFT.value,
    ) -> str:
        """
        Moves an email from one folder to another.
        :param email_id: The id of the email to move.
        :param source_folder_name: The folder to move the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :param dest_folder_name: The folder to move the email to: INBOX, SENT, DRAFT, TRASH. Defaults to DRAFT.
        :returns: The id of the email just moved, if successful, otherwise raise ValueError.

        :example:
            move_email("1234567890abcdef", "INBOX", "DRAFT")
        """
        source_folder = EmailFolderName[source_folder_name.upper()]
        if source_folder not in self.folders:
            raise ValueError(f"Folder {source_folder} not found")
        dest_folder = EmailFolderName[dest_folder_name.upper()]
        if dest_folder not in self.folders:
            raise ValueError(f"Folder {dest_folder} not found")
        email = self.folders[source_folder].get_email_by_id(email_id)
        self.folders[dest_folder].add_email(email)
        self.folders[source_folder].emails.remove(email)
        return email.email_id

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def delete_email(
        self,
        email_id: str,
        folder_name: str = EmailFolderName.INBOX.value,
    ) -> str:
        """
        Deletes an email from the specified folder.
        :param email_id: The id of the email to delete.
        :param folder_name: The folder to delete the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :returns: The id of the email just deleted, if successful, otherwise raise ValueError.

        :example:
            delete_email("1234567890abcdef", "INBOX")
        """
        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")
        email = self.folders[folder].get_email_by_id(email_id)
        self.folders[folder].emails.remove(email)
        self.folders[EmailFolderName.TRASH].add_email(email)
        return email.email_id

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def search_emails(
        self,
        query: str,
        folder_name: str = EmailFolderName.INBOX.value,
    ) -> list[Email]:
        """
        Searches for emails across all folders based on a query string.
        The search looks for partial matches in sender, recipients, subject, and content.
        :param query: The search query string
        :param folder_name: The folder to search in: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :returns: A list of emails that match the query.

        :example:
            search_emails("hello", "INBOX")
        """
        query = query.lower()
        results = []
        folder = EmailFolderName[folder_name.upper()]

        for email in self.folders[folder].emails:
            if (
                query in email.sender.lower()
                or any(query in recipient.lower() for recipient in email.recipients)
                or query in email.subject.lower()
                or query in email.content.lower()
            ):
                results.append(email)

        return results

    def delete_future_data(self, timestamp: float):
        num_emails = sum(len(self.folders[folder].emails) for folder in self.folders)
        for folder in self.folders:
            self.folders[folder].emails = [
                email
                for email in self.folders[folder].emails
                if email.timestamp <= timestamp
            ]
        num_emails_after = sum(
            len(self.folders[folder].emails) for folder in self.folders
        )
        logger.debug(
            f"Deleted emails {num_emails - num_emails_after} from {num_emails}"
        )


@dataclass
class Mail(EmailClientApp):
    __doc__ = EmailClientApp.__doc__
    name: str | None = "Mail"


@dataclass
class EmailClientV2(EmailClientApp):
    internal_fs: SandboxLocalFileSystem | VirtualFileSystem | None = None

    def connect_to_protocols(self, protocols: dict[Protocol, Any]) -> None:
        file_system = protocols.get(Protocol.FILE_SYSTEM)
        if isinstance(file_system, (SandboxLocalFileSystem, VirtualFileSystem)):
            self.internal_fs = file_system

    def add_attachment(
        self,
        email: Email,
        attachment_path: str,
    ) -> None:
        """
        Adds an attachment to an email.
        :param email: The email to add the attachment to.
        :param attachment_path: The path to the attachment to add.
        :param folder_name: The folder to add the attachment to: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :returns: None
        """
        if self.internal_fs is not None:
            if not self.internal_fs.exists(attachment_path):
                raise ValueError(f"File does not exist: {attachment_path}")
            with disable_events():
                with self.internal_fs.open(attachment_path, "rb") as f:
                    file_content = base64.b64encode(f.read())
                    file_name = os.path.basename(attachment_path)
                    if not email.attachments:
                        email.attachments = {}
                    email.attachments[file_name] = file_content
        else:
            email.add_attachment(attachment_path)

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def send_email(
        self,
        recipients: list[str] | None = None,
        subject: str = "",
        content: str = "",
        cc: list[str] | None = None,
        attachment_paths: list[str] | None = None,
    ) -> str:
        """
        Sends an email to the specified recipients.
        :param recipients: The recipients of the email.
        :param subject: The subject of the email.
        :param content: The content of the email.
        :param cc: The cc of the email.
        :param attachment_paths: The paths of the attachments to add to the email.
        :returns: The id of the email just created.

        :example:
            send_email(["user1@meta.com", "user2@meta.com"], "Hello", "Hi there", ["user3@meta.com"], ["tmp123/file1.txt", "tmp234/file2.txt"])
        """
        if recipients is None:
            recipients = []
        if cc is None:
            cc = []
        if attachment_paths is None:
            attachment_paths = []
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=self.user_email,
            recipients=recipients,
            subject=subject,
            content=content,
            timestamp=self.time_manager.time(),
            cc=cc,
        )
        for path in attachment_paths:
            self.add_attachment(email, path)

        with disable_events():
            self.add_email(email=email, folder_name=EmailFolderName.SENT)
        return email.email_id

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def download_attachments(
        self,
        email_id: str,
        folder_name: str = EmailFolderName.INBOX.value,
        path_to_save: str = "Downloads/",
    ) -> list[str]:
        """
        Downloads attachments from an email.
        :param email_id: The id of the email to download.
        :param folder_name: The folder to download the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :param path_to_save: The path to save the attachments to.
        :returns: The list of  full path to the downloaded attachments.

        :example:
            download_attachments("1234567890abcdef", "INBOX", "Downloads/")
        """
        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")

        email = self.folders[folder].get_email_by_id(email_id)
        filenames = []
        if not email.attachments:
            return filenames
        for file_name, file_content in email.attachments.items():
            if self.internal_fs is not None:
                with disable_events():
                    with self.internal_fs.open(
                        os.path.join(path_to_save, file_name), "wb"
                    ) as f:
                        file_data = base64.b64decode(file_content)
                        f.write(file_data)
                        filenames.append(os.path.join(path_to_save, file_name))
            else:
                with open(os.path.join(path_to_save, file_name), "wb") as f:
                    file_data = base64.b64decode(file_content)
                    f.write(file_data)
                    filenames.append(os.path.join(path_to_save, file_name))
        return filenames

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def reply_to_email(
        self,
        email_id: str,
        folder_name: str = EmailFolderName.INBOX.value,
        content: str = "",
        attachment_paths: list[str] | None = None,
    ) -> str:
        """
        Replies to an email.
        :param email_id: The id of the email to reply to, the Id is specified in the email details.
        :param folder_name: The folder to reply to the email from: INBOX, SENT, DRAFT, TRASH. Defaults to INBOX.
        :param content: The content of the reply.
        :param attachment_paths: The paths of the attachments to add to the reply.
        :returns: The id of the email just created and sent.

        :example:
            reply_to_email("1234567890abcdef", "INBOX", "Hi there", ["tmp123/file1.txt", "tmp234/file2.txt"])
        """
        if attachment_paths is None:
            attachment_paths = []

        def get_recipient(email: Email) -> str:
            while email.sender == self.user_email and email.parent_id:
                email_found = False
                for folder in self.folders:
                    try:
                        email = self.folders[folder].get_email_by_id(email.parent_id)
                        email_found = True
                        break
                    except Exception:
                        logger.error("Email not found in ", folder)
                if not email_found:
                    raise ValueError(f"Email with id {email.parent_id} not found")
            return email.sender

        folder = EmailFolderName[folder_name.upper()]
        if folder not in self.folders:
            raise ValueError(f"Folder {folder_name} not found")

        replying_to_email = self.folders[folder].get_email_by_id(email_id)
        recipient = get_recipient(replying_to_email)
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=self.user_email,
            recipients=[recipient],
            subject="Re: " + replying_to_email.subject,
            content=content,
            timestamp=self.time_manager.time(),
            cc=[],
            parent_id=replying_to_email.email_id,
        )
        for path in attachment_paths:
            self.add_attachment(email, path)

        with disable_events():
            self.add_email(email=email, folder_name=EmailFolderName.SENT)
        return email.email_id

    @type_check
    @env_tool()
    @event_registered(operation_type=OperationType.WRITE, event_type=EventType.ENV)
    def reply_to_email_from_user(
        self,
        sender: str,
        email_id: str,
        content: str = "",
        attachment_paths: list[str] | None = None,
    ) -> str:
        """
        Creates a reply to an email from the user and adds it to the user's INBOX folder.
        :param sender: The sender of the email.
        :param email_id: The id of the email to reply to, the Id is specified in the email details.
        :param content: The content of the reply.
        :param attachment_paths: The paths of the attachments to add to the reply.
        :returns: The id of the email just created and sent.

        :example:
            reply_to_email_from_user("other_user@meta.com", "1234567890abcdef", "Hello, this is a reply to your email.")
        """
        if attachment_paths is None:
            attachment_paths = []

        folder = EmailFolderName[EmailFolderName.SENT.value]

        replying_to_email = self.folders[folder].get_email_by_id(email_id)

        if sender not in replying_to_email.recipients:
            raise ValueError(
                f"Sender {sender} is not a recipient of the email, the recipients are {replying_to_email.recipients}"
            )
        email = Email(
            email_id=uuid_hex(self.rng),
            sender=sender,
            recipients=[self.user_email],
            subject="Re: " + replying_to_email.subject,
            content=content,
            timestamp=self.time_manager.time(),
            cc=[],
            parent_id=replying_to_email.email_id,
        )
        for path in attachment_paths:
            self.add_attachment(email, path)

        with disable_events():
            self.add_email(email=email, folder_name=EmailFolderName.INBOX)
        return email.email_id
