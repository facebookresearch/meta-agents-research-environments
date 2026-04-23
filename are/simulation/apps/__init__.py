# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from .agent_user_interface import AgentUserInterface
from .apartment_listing import ApartmentListingApp, RentAFlat
from .app import App
from .cab import CabApp
from .calendar import Calendar, CalendarApp
from .city import CityApp
from .contacts import Contacts, ContactsApp, InternalContacts
from .email_client import EmailClientV2, Mail
from .flight_customer_service import (
    FlightClient,
    FlightClientApp,
    FlightCustomerService,
    FlightCustomerServiceApp,
)
from .messaging_v2 import MessagingAppV2
from .reminder import ReminderApp
from .sandbox_file_system import Files, SandboxLocalFileSystem
from .shopping import Shopping, ShoppingApp
from .system import SystemApp
from .virtual_file_system import VirtualFileSystem

__all__ = [
    "AgentUserInterface",
    "ApartmentListingApp",
    "App",
    "CabApp",
    "Calendar",
    "CalendarApp",
    "CityApp",
    "Contacts",
    "ContactsApp",
    "EmailClientV2",
    "Files",
    "FlightClient",
    "FlightClientApp",
    "FlightCustomerService",
    "FlightCustomerServiceApp",
    "InternalContacts",
    "Mail",
    "MessagingAppV2",
    "ReminderApp",
    "RentAFlat",
    "SandboxLocalFileSystem",
    "Shopping",
    "ShoppingApp",
    "SystemApp",
    "VirtualFileSystem",
]

ALL_APPS = [
    AgentUserInterface,
    ApartmentListingApp,
    App,
    CabApp,
    Calendar,
    CalendarApp,
    CityApp,
    Contacts,
    ContactsApp,
    EmailClientV2,
    Files,
    FlightClient,
    FlightClientApp,
    FlightCustomerService,
    FlightCustomerServiceApp,
    InternalContacts,
    Mail,
    MessagingAppV2,
    RentAFlat,
    SandboxLocalFileSystem,
    Shopping,
    ShoppingApp,
    SystemApp,
    VirtualFileSystem,
]

INTERNAL_APPS = [InternalContacts]
