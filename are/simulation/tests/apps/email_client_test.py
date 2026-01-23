# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import os

import pytest

from are.simulation.apps.email_client import (
    Email,
    EmailClientApp,
    EmailFolder,
    EmailFolderName,
)
from are.simulation.apps.sandbox_file_system import SandboxLocalFileSystem
from are.simulation.environment import Environment


def dummy_fs_population(fs):
    fs.mkdir("source_dir")
    fs.mkdir("download_dir")
    with fs.open("source_dir/test1.txt", "w") as file:
        file.write("test with emojis 💵")
    with fs.open("source_dir/test2.txt", "w") as file:
        file.write("test with emojis 💵💵")
    fs.write_bytes("source_dir/llama.jpg", b"I am a binary image !")


class TestEmail:
    def test_instantiate_email_default(self):
        email = Email(
            sender="test@gmail.com",
            recipients=["test@gmail.com"],
            subject="test",
            content="test",
        )
        assert email.sender == "test@gmail.com"
        assert email.recipients == ["test@gmail.com"]
        assert email.subject == "test"
        assert email.content == "test"

    def test_instantiate_email_with_cc(self):
        email = Email(
            sender="test@gmail.com",
            recipients=["test@gmail.com"],
            subject="test",
            content="test",
            cc=["cc1@example.com", "cc2@example.com"],
        )
        assert email.sender == "test@gmail.com"
        assert email.recipients == ["test@gmail.com"]
        assert email.subject == "test"
        assert email.content == "test"
        assert email.cc == ["cc1@example.com", "cc2@example.com"]

    def test_instantiate_email_with_attachments(self):
        attachment1 = bytes("attachment1", "utf-8")
        attachment2 = bytes("attachment2", "utf-8")
        email = Email(
            sender="test@gmail.com",
            recipients=["test@gmail.com"],
            subject="test",
            content="test",
            attachments={
                "attachment1.txt": attachment1,
                "attachment2.txt": attachment2,
            },
        )
        assert email.sender == "test@gmail.com"
        assert email.recipients == ["test@gmail.com"]
        assert email.subject == "test"
        assert email.content == "test"
        assert email.attachments is not None
        assert email.attachments["attachment1.txt"] == attachment1
        assert email.attachments["attachment2.txt"] == attachment2

    def test_instantiate_email_with_invalid_sender(self):
        with pytest.raises(ValueError):
            Email(
                sender="invalid",
                recipients=["test@gmail.com"],
                subject="test",
                content="test",
            )

    def test_instantiate_email_with_invalid_recipient(self):
        with pytest.raises(ValueError):
            Email(
                sender="test@gmail.com",
                recipients=["invalid"],
                subject="test",
                content="test",
            )

    def test_instantiate_email_with_invalid_cc(self):
        with pytest.raises(ValueError):
            Email(
                sender="test@gmail.com",
                recipients=["test@gmail.com"],
                subject="test",
                content="test",
                cc=["invalid"],
            )


def create_email():
    return Email(
        sender="sender@example.com",
        recipients=["recipient@example.com"],
        subject="Test Email",
        content="This is a test email.",
    )


class TestEmailFolder:
    def test_add_email(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        email = create_email()
        folder.add_email(email)
        assert email in folder.emails
        assert folder.emails[0] == email  # Since it's sorted by timestamp

    def test_get_emails_within_limit(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        for _ in range(3):
            email = create_email()
            folder.add_email(email)

        res = folder.get_emails(0)
        emails = res.emails
        assert len(emails) == 3
        assert emails[0].timestamp >= emails[1].timestamp >= emails[2].timestamp

    def test_get_emails_with_offset(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        for _ in range(10):
            email = create_email()
            folder.add_email(email)

        res = folder.get_emails(5)
        emails = res.emails
        assert len(emails) == 5  # view_limit is 5 by default
        assert emails[0].timestamp >= emails[-1].timestamp

    def test_get_emails_with_large_offset(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        for _ in range(10):
            email = create_email()
            folder.add_email(email)

        res = folder.get_emails(8)
        emails = res.emails
        assert len(emails) == 2
        assert emails[0].timestamp >= emails[-1].timestamp

    def test_get_emails_negative_offset(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        with pytest.raises(ValueError):
            folder.get_emails(-1)

    def test_get_emails_excessive_offset(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        for _ in range(3):
            email = create_email()
            folder.add_email(email)

        with pytest.raises(ValueError):
            folder.get_emails(5)

    def test_get_email_by_index(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        email = create_email()
        folder.add_email(email)
        retrieved_email = folder.get_email(0)
        assert retrieved_email == email

    def test_get_email_by_index_out_of_range(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        with pytest.raises(IndexError):
            folder.get_email(0)  # No emails added yet

    def test_get_email_by_id(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        email = create_email()
        folder.add_email(email)
        retrieved_email = folder.get_email_by_id(email.email_id)
        assert retrieved_email == email

    def test_get_email_by_id_not_found(self):
        folder = EmailFolder(EmailFolderName.INBOX)
        with pytest.raises(Exception):
            folder.get_email_by_id("nonexistent_id")

    def test_total_returned_emails_matches_actual_count(self):
        """Verify total_returned_emails equals the actual number of emails returned."""
        folder = EmailFolder(EmailFolderName.INBOX)
        for _ in range(10):
            folder.add_email(create_email())

        # Test various offset/limit combinations
        test_cases = [
            (0, 5),  # First 5 emails
            (0, 10),  # All 10 emails
            (5, 5),  # Last 5 emails
            (8, 5),  # Only 2 emails available (offset 8, limit 5, but only 10 total)
            (0, 3),  # First 3 emails
        ]
        for offset, limit in test_cases:
            res = folder.get_emails(offset, limit)
            assert res.total_returned_emails == len(res.emails), (
                f"total_returned_emails ({res.total_returned_emails}) does not match "
                f"len(emails) ({len(res.emails)}) for offset={offset}, limit={limit}"
            )


class TestEmailClientApp:
    def test_instantiate_email_client_app(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        assert len(app.folders) == 4
        assert app.folders[EmailFolderName.INBOX].folder_name == EmailFolderName.INBOX
        assert app.folders[EmailFolderName.SENT].folder_name == EmailFolderName.SENT
        assert app.folders[EmailFolderName.DRAFT].folder_name == EmailFolderName.DRAFT
        assert app.folders[EmailFolderName.TRASH].folder_name == EmailFolderName.TRASH

    def test_add_email_to_folder(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        email = create_email()
        app.add_email(email, EmailFolderName.INBOX)
        assert email in app.folders[EmailFolderName.INBOX].emails

    def test_move_email(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        email = create_email()
        app.add_email(email, EmailFolderName.INBOX)
        assert email in app.folders[EmailFolderName.INBOX].emails
        assert email not in app.folders[EmailFolderName.TRASH].emails
        app.move_email(email.email_id, "INBOX", "TRASH")
        assert email not in app.folders[EmailFolderName.INBOX].emails
        assert email in app.folders[EmailFolderName.TRASH].emails

    def test_delete_email(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        email = create_email()
        app.add_email(email, EmailFolderName.INBOX)
        assert email in app.folders[EmailFolderName.INBOX].emails
        app.delete_email(email_id=email.email_id, folder_name="INBOX")
        assert email not in app.folders[EmailFolderName.INBOX].emails

    def test_read_email(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        email = create_email()
        app.add_email(email, EmailFolderName.INBOX)
        read_email = app.get_email_by_id(email_id=email.email_id, folder_name="inbox")
        assert read_email.is_read
        assert app.folders[EmailFolderName.INBOX].emails[0].is_read

    def test_list_emails(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        email = Email(
            sender="test@example.com",
            recipients=["recipient@example.com"],
            subject="Test Email",
            content="This is a test email.",
        )
        app.add_email(email, EmailFolderName.INBOX)
        result = app.list_emails("INBOX", 0)
        assert len(result.emails) == 1
        assert result.emails[0].subject == "Test Email"
        assert result.total_emails == 1
        assert result.emails_range == (0, 1)

    def test_list_emails_offset_out_of_range(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        email = Email(
            sender="test@example.com",
            recipients=["recipient@example.com"],
            subject="Test Email",
            content="This is a test email.",
        )
        app.add_email(email, EmailFolderName.INBOX)
        with pytest.raises(ValueError):
            app.list_emails("INBOX", 10)  # offset is greater than total emails

    def test_list_emails_folder_not_found(self):
        app = EmailClientApp()
        env = Environment()
        env.register_apps([app])
        with pytest.raises(Exception):
            app.list_emails("INVALID_FOLDER", 0)  # folder does not exist

    def test_send_email(self):
        app = EmailClientApp()
        fs = SandboxLocalFileSystem()
        environment = Environment()
        environment.register_apps([app, fs])
        dummy_fs_population(fs)

        app.send_email(
            recipients=["test1@meta.com", "test2@meta.com"],
            subject="Test Email",
            content="This is a test email.",
            attachment_paths=[os.path.join(fs.tmpdir, "source_dir/test1.txt")],
        )

        assert len(app.list_emails("SENT").emails) == 1
        em = app.list_emails("SENT").emails[0]
        assert em.subject == "Test Email"
        assert em.content == "This is a test email."
        assert em.sender == app.user_email
        assert em.recipients == ["test1@meta.com", "test2@meta.com"]
        assert len(em.attachments) == 1

    def test_send_email_complex(self):
        app = EmailClientApp()
        fs = SandboxLocalFileSystem()
        environment = Environment()
        environment.register_apps([app, fs])
        # FS population
        fs.mkdir("source_dir")
        fs.write_bytes("source_dir/llama.jpg", b"I am a binary image !")
        with fs.open("source_dir/test1.txt", "w") as file:
            file.write("test with emojis 💵")
        fs.mkdir("downloads")

        app.send_email(
            recipients=["test1@meta.com", "test2@meta.com"],
            subject="Test Email",
            content="This is a test email.",
            attachment_paths=[
                os.path.join(fs.tmpdir, "source_dir/test1.txt"),
                os.path.join(fs.tmpdir, "source_dir/llama.jpg"),
            ],
        )

        assert len(app.list_emails("SENT").emails) == 1
        em = app.list_emails("SENT").emails[0]
        assert em.subject == "Test Email"
        assert em.content == "This is a test email."
        assert em.sender == app.user_email
        assert em.recipients == ["test1@meta.com", "test2@meta.com"]
        assert len(em.attachments) == 2

        app.download_attachments(
            email_id=em.email_id,
            folder_name="sent",
            path_to_save=os.path.join(fs.tmpdir, "downloads"),
        )
        assert len(os.listdir(os.path.join(fs.tmpdir, "downloads"))) == 2
        assert fs.read_bytes("downloads/test1.txt") == fs.read_bytes(
            "source_dir/test1.txt"
        )
        assert fs.read_bytes("downloads/llama.jpg") == fs.read_bytes(
            "source_dir/llama.jpg"
        )

    def test_reply_to_email(self):
        app = EmailClientApp()
        environment = Environment()
        environment.register_apps([app])
        email = Email(
            sender="test@example.com",
            recipients=[app.user_email],
            subject="Test Email",
            content="This is a test email.",
        )
        app.add_email(email, EmailFolderName.INBOX)
        reply_content = "This is a reply to the test email."
        app.reply_to_email(
            email_id=email.email_id, folder_name="inbox", content=reply_content
        )
        replied_email = app.list_emails("SENT").emails[0]
        assert replied_email.subject == "Re: " + email.subject
        assert replied_email.content == reply_content
        assert replied_email.sender == app.user_email
        assert replied_email.recipients == [email.sender]
        assert replied_email.parent_id == email.email_id

    def test_double_reply_to_email(self):
        app = EmailClientApp()
        environment = Environment()
        environment.register_apps([app])
        email = Email(
            sender="test@example.com",
            recipients=[app.user_email],
            subject="Test Email",
            content="This is a test email.",
        )
        app.add_email(email, EmailFolderName.INBOX)
        reply_email_id = app.reply_to_email(
            email_id=email.email_id, folder_name="inbox", content="First reply"
        )
        app.reply_to_email(
            email_id=reply_email_id, folder_name="sent", content="Second reply"
        )
        replied_email = app.list_emails("SENT").emails[1]
        assert replied_email.subject == "Re: " + email.subject
        assert replied_email.content == "First reply"
        assert replied_email.sender == app.user_email
        assert replied_email.recipients == [email.sender]
        assert replied_email.parent_id == email.email_id

        replied_email = app.list_emails("SENT").emails[0]
        assert replied_email.subject == "Re: " + "Re: " + email.subject
        assert replied_email.content == "Second reply"
        assert replied_email.sender == app.user_email
        assert replied_email.recipients == [email.sender]
        assert replied_email.parent_id == reply_email_id

    def test_reply_to_chain_of_emails(self):
        # Reply to the latest email in the chain with a sender different from the user.
        app = EmailClientApp()
        environment = Environment()
        environment.register_apps([app])
        email = Email(
            sender="test@example.com",
            recipients=[app.user_email],
            subject="Test Email",
            content="This is a test email.",
        )
        app.add_email(email, EmailFolderName.INBOX)
        reply_email_id = app.reply_to_email(
            email_id=email.email_id, folder_name="inbox", content="First reply"
        )
        reply_to_user_email_id = app.reply_to_email_from_user(
            sender="test@example.com",
            email_id=reply_email_id,
            content="Reply to user",
        )
        app.reply_to_email(
            email_id=reply_to_user_email_id, folder_name="inbox", content="Second reply"
        )
        replied_email = app.list_emails("SENT").emails[1]
        assert replied_email.subject == "Re: " + email.subject
        assert replied_email.content == "First reply"
        assert replied_email.sender == app.user_email
        assert replied_email.recipients == ["test@example.com"]
        assert replied_email.parent_id == email.email_id

        replied_email = app.list_emails("SENT").emails[0]
        assert replied_email.subject == f"Re: Re: Re: {email.subject}"
        assert replied_email.content == "Second reply"
        assert replied_email.sender == app.user_email
        assert replied_email.recipients == ["test@example.com"]
        assert replied_email.parent_id == reply_to_user_email_id


def test_forward_email():
    app = EmailClientApp()
    # Create original email
    original_email = Email(
        sender="sender@meta.com",
        recipients=["user@meta.com"],
        subject="Original Email",
        content="Original content",
    )
    original_id = app.send_email_to_user(original_email)

    # Forward the email
    forwarded_id = app.forward_email(
        email_id=original_id,
        recipients=["forward@meta.com"],
        folder_name=EmailFolderName.INBOX.value,
    )

    # Verify forwarded email is in SENT folder in both app and tree_app
    assert len(app.folders[EmailFolderName.SENT].emails) == 1
    assert app.folders[EmailFolderName.SENT].emails[0].email_id == forwarded_id
