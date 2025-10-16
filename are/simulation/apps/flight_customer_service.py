import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from os import name
from typing import Any, Optional, TypedDict

from are.simulation.apps.app import App
from are.simulation.tool_utils import app_tool, data_tool, env_tool, OperationType
from are.simulation.types import event_registered, EventType
from are.simulation.utils import get_state_dict, type_check, uuid_hex
from are.simulation.utils.type_utils import check_type

logger = logging.getLogger(__name__)


class TicketType(Enum):
    ECONOMY = "ECONOMY"
    ECONOMY_PLUS = "ECONOMY_PLUS"
    BUSINESS = "BUSINESS"
    FIRST_CLASS = "FIRST_CLASS"


class LoyaltyTier(Enum):
    NORMAL = "NORMAL"
    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"


class BaggageType(Enum):
    NORMAL_BAG = "normal_bag"
    MEDICAL_BAG = "medical_bag"
    SPORTS_EQUIPMENT = "sports_equipment"
    WHEELCHAIR = "wheelchair"
    STROLLER = "stroller"
    BABY_CARRIER = "baby_carrier"


class BaggageCheckinType(Enum):
    CARRY_ON = "carry_on"
    CHECKED_LUGGAGE = "checked_luggage"


@dataclass(kw_only=True)
class Email:
    email_address: str
    from_address: str
    to_address: list[str] = field(default_factory=list)
    title: str = ""
    content: str = ""


@dataclass(kw_only=True)
class MedicalDocFile:
    diagnosis: str
    carry_on_items: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class MedicalDocFolder:
    medical_docs: list[MedicalDocFile] = field(default_factory=list)


@dataclass(kw_only=True)
class FlightInformation:
    flight_number: str
    airline: str = "Unity Air"
    origin: str = ""
    destination: str = ""
    departure_time: str = ""
    arrival_time: str = ""
    departure_gate: str = ""
    arrival_terminal: str = ""


@dataclass(kw_only=True)
class BaggageInfo:
    bagage_type: Optional[BaggageType] = BaggageType.NORMAL_BAG
    baggage_checkin_type: BaggageCheckinType = BaggageCheckinType.CARRY_ON
    weight: Optional[str] = None
    size: Optional[str] = None
    check_type: Optional[str] = None


@dataclass(kw_only=True)
class BookingInformation:
    confirmation_number: str
    flight_information: FlightInformation
    passenger_name: str
    loyalty_tier: Optional[LoyaltyTier] = None
    ticket_type: TicketType = TicketType.ECONOMY
    baggage_info: list[BaggageInfo] = field(default_factory=list)


@dataclass(kw_only=True)
class UserMemberInformation:
    name: Optional[str] = None
    email: Optional[str] = None
    air_line: str = "Unity Air"
    loyalty_member_id: Optional[str] = None
    loyalty_tier: Optional[LoyaltyTier] = None
    points_balance: int = 0
    lifetime_miles: int = 0
    member_since: str = ""
    phone_number: str = ""


@dataclass(kw_only=True)
class UserInformation:
    name: Optional[str] = None
    age: Optional[int] = None
    emails: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class RegularBaggagePolicy:
    ticket_type: TicketType = TicketType.ECONOMY
    loyalty_tier: LoyaltyTier = LoyaltyTier.NORMAL
    free_checked_bag_number: int = 0
    checked_bag_weight_allowance: str = "50 lbs"
    additional_fee_for_checked_bag: str = "50 USD"
    carry_on_bag_size: str = "22 x 14 x 9 inches"


@dataclass(kw_only=True)
class SpecialBaggagePolicy:
    bagage_type: Optional[BaggageType] = None
    allow_to_check_in: bool = False
    allow_to_carry_on: bool = False
    require_additional_fee: bool = True
    weight_allowance: Optional[str] = None
    size_allowance: Optional[str] = None


@dataclass(kw_only=True)
class NumberGeneratorResult:
    generated_number: int = 0


@dataclass(kw_only=True)
class NumberValidationResult:
    target_number: int = 80
    received_number: int = 0


@dataclass
class FlightClientApp(App):
    """
    Applications for clients to find their personal information.

    This app manages user information, including the user personal information, flight
    booking information and their medical conditions. It's designed to handle the specific
    scenario where customers find their flight details, membership details from emails and provide
    other information such as the baggage information and medical info to the customer service.

    Key Features:
    - Flight Booking Information Management: Get the booking information of the user.
    - Membership  Tracking: Idenfity the membership information of the user.
    - Baggage  Management: Way to manage the baggage for the user.
    - Medical condition: For clients to provide their medical related information.
    """

    name: str | None = None

    def __post_init__(self):
        super().__init__(self.name)
        self.air_line: str = "Unity Air"
        self.bookings: Optional[BookingInformation] = None
        self.user_membership_info: Optional[UserMemberInformation] = None
        self.user_info: UserInformation = UserInformation()
        self.medical_folder: MedicalDocFolder = MedicalDocFolder()
        self.emails: list[Email] = []
        self.baggage_info: list[BaggageInfo] = []

    def reset(self):
        super().reset()
        self.air_line: str = "Unity Air"
        self.bookings: Optional[BookingInformation] = None
        self.user_membership_info: Optional[UserMemberInformation] = None
        self.user_info: UserInformation = UserInformation()
        self.medical_folder: MedicalDocFolder = MedicalDocFolder()
        self.emails: list[Email] = []
        self.baggage_info: list[BaggageInfo] = []

    def get_state(self) -> dict[str, Any]:
        """Return the current state of the app."""
        return {
            "bookings": self.bookings,
            "user_membership_info": self.user_membership_info,
            "user_info": self.user_info,
            "medical_folder": self.medical_folder,
            "emails": self.emails,
            "air_line": self.air_line,
            "baggage_info": self.baggage_info,
        }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Load state into the app."""
        self.bookings = state_dict.get("bookings")
        self.user_membership_info = state_dict.get("user_membership_info")
        self.user_info = state_dict.get("user_info", {})
        self.medical_folder = state_dict.get("medical_folder", {})
        self.air_line = state_dict.get("air_line", "Unity Air")
        self.baggage_info = state_dict.get("baggage_info", [])

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_membership_info(
        self,
    ) -> Optional[UserMemberInformation]:
        """
        Get the membership information of the user.
        """
        if self.user_membership_info:
            return self.user_membership_info
        return self.get_membership_info_from_email()

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def say_hi_to_customer_service(
        self,
    ) -> str:
        """
        Say hi to customer service.
        """
        return "Hi, my name is Alex Sharma, and I need your help?"

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_booking_info(
        self,
    ) -> Optional[BookingInformation]:
        """
        Get the booking information of the user.
        """
        if self.bookings:
            return self.bookings
        return self.get_booking_info_from_email()

    @type_check
    @app_tool()
    @env_tool()
    @data_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def check_medical_info(
        self,
    ):
        """
        Check medical information of the user.
        If medical note requires a medical bag, add it to the baggage info.
        """
        if len(self.medical_folder.medical_docs) > 0:
            self.baggage_info.append(
                BaggageInfo(
                    bagage_type=BaggageType.MEDICAL_BAG,
                    baggage_checkin_type=BaggageCheckinType.CARRY_ON,
                )
            )

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_baggage_info(
        self,
    ) -> list[BaggageInfo]:
        """
        Get the baggage information user.
        """
        return self.baggage_info

    def get_membership_info_from_email(
        self,
    ) -> Optional[UserMemberInformation]:
        """
        Get the membership information of the user.
        """
        for email in self.emails:
            """
            Expect email content like
                Thank you for becoming a Unity Air member!   your loyalty_member_id is UA_member_123, and your loyalty_tier is GOLD                   Welcome to the Unity Airline, your loyalty_member_id is UA_member_123, and your loyalty_tier is GOLD
            """
            pattern = r"loyalty_member_id is (\w+).*loyalty_tier is (\w+)"
            match = re.search(pattern, email.content)
            if match:
                loyalty_member_id = match.group(1)
                loyalty_tier = match.group(2)
                self.user_membership_info = UserMemberInformation(
                    name=self.name,
                    email=email.email_address,
                    air_line=self.air_line,
                    loyalty_member_id=loyalty_member_id,
                    loyalty_tier=LoyaltyTier[loyalty_tier],
                )
        return self.user_membership_info

    def get_booking_info_from_email(
        self,
    ) -> Optional[BookingInformation]:
        """
        Get the flight booking information of the user.
        """
        for email in self.emails:
            """
            Expect email content like
                Thanks for choosing Unity Airline, your flight is UA-2209, and your confirmation number is UA_1234567890
            """
            pattern = r"flight is is (\w+).*confirmation number is (\w+)"
            match = re.search(pattern, email.content)
            if match:
                flight_number = match.group(1)
                confirmation_number = match.group(2)
                self.bookings = BookingInformation(
                    confirmation_number=confirmation_number,
                    flight_information=FlightInformation(flight_number=flight_number),
                    passenger_name=(self.user_info.name or ""),
                )
        return self.bookings


@dataclass
class FlightClient(FlightClientApp):
    __doc__ = FlightClientApp.__doc__
    name: str | None = "FlightClient"


@dataclass
class FlightCustomerServiceApp(App):
    """
    Applications for flight customer service to handle the client inqueries.
    """

    name: str | None = None

    def __post_init__(self):
        super().__init__(self.name)
        self.air_line: str = "Unity Air"
        self.user_info: UserInformation = UserInformation()
        self.client_booking_info: list[BookingInformation] = []
        self.client_membership_info: list[UserMemberInformation] = []
        self.client_user_info: list[UserInformation] = []
        self.regular_bag_policies: list[RegularBaggagePolicy] = []
        self.special_bag_policies: list[SpecialBaggagePolicy] = []

    def reset(self):
        super().reset()
        self.air_line: str = "Unity Air"
        self.user_info: UserInformation = UserInformation()
        self.client_user_info: list[UserInformation] = []
        self.client_booking_info: list[BookingInformation] = []
        self.client_membership_info: list[UserMemberInformation] = []
        self.user_info: UserInformation = UserInformation()
        self.regular_bag_policies: list[RegularBaggagePolicy] = []
        self.special_bag_policies: list[SpecialBaggagePolicy] = []

    def get_state(self) -> dict[str, Any]:
        """Return the current state of the app."""
        return {
            "bookings": self.bookings,
            "user_info": self.user_info,
            "client_bookings": self.client_booking_info,
            "client_membership_info": self.client_membership_info,
            "regular_bag_policies": self.regular_bag_policies,
            "special_bag_policies": self.special_bag_policies,
        }

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Load state into the app."""
        self.air_line = state_dict.get("air_line", "Unity Air")
        self.user_info = state_dict.get("user_info", UserInformation())
        self.client_user_info = state_dict.get("client_user_info", UserInformation())
        self.client_booking_info = state_dict.get("client_user_info", [])
        self.client_membership_info = state_dict.get("client_membership_info", [])
        self.user_info: UserInformation = state_dict.get("user_info", UserInformation())
        self.regular_bag_policies = state_dict.get("regular_bag_policies", [])
        self.special_bag_policies = state_dict.get("special_bag_policies", [])

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_greeting_message_to_client(
        self,
    ) -> str:
        """
        Generate a greeting message to the client, and ask their confirmation number and loyalty_member_id.
        """
        return """Hello, my name is Sarah Chen, and I am happy to help you.
         Can you please provide me with your confirmation number and your loytalty_member_id?"""

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_client_membership_info(
        self,
        client_loyalty_member_id: str,
    ) -> Optional[UserMemberInformation]:
        """
        Get the membership information from the client based on their loyalty_member_id, return null if not found.
        """
        for client_membership_info in self.client_membership_info:
            if client_membership_info.loyalty_member_id == client_loyalty_member_id:
                return client_membership_info
        return None

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_client_flight_booking_info(
        self,
        client_confirmation_number: str,
    ) -> Optional[BookingInformation]:
        """
        Get the flight booking information from the client based on their confirmation_number, return null if not found.
        """
        for client_booking_info in self.client_booking_info:
            if client_booking_info.confirmation_number == client_confirmation_number:
                return client_booking_info
        return None

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_client_baggage_policy_info(
        self,
        client_loyalty_member_id: str,
        client_confirmation_number: str,
    ) -> RegularBaggagePolicy:
        """
        Get the normal baggage policy information from the client.
        The baggage policy information differs for client LoyaltyTier
        and their ticket_type. Where the LoyaltyTier can be determined
        by their loyalty_member_id and ticket_type can be obtained from
        their client_confirmation_number.
        """
        client_loyalty_tier_info = self.get_client_membership_info(
            client_loyalty_member_id
        )
        if client_loyalty_tier_info:
            client_loyalty_tier = client_loyalty_tier_info.loyalty_tier
        else:
            client_loyalty_tier = LoyaltyTier.NORMAL
        client_ticket_info = self.get_client_flight_booking_info(
            client_confirmation_number
        )
        if client_ticket_info:
            client_ticket_type = client_ticket_info.ticket_type
        else:
            client_ticket_type = TicketType.ECONOMY
        for regular_baggage_policy in self.regular_bag_policies:
            if (
                regular_baggage_policy.loyalty_tier == client_loyalty_tier
                and regular_baggage_policy.ticket_type == client_ticket_type
            ):
                return regular_baggage_policy
        return RegularBaggagePolicy()

    @type_check
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def get_special_baggage_policy_info(
        self,
        bagage_type: BaggageType,
    ) -> SpecialBaggagePolicy:
        """
        Get the normal baggage policy information from the client.
        The special baggage policy information depends on the bagage_type.
        """
        for special_baggage_policy in self.special_bag_policies:
            if special_baggage_policy.bagage_type == bagage_type:
                return special_baggage_policy
        return SpecialBaggagePolicy()

    @type_check
    @app_tool()
    @data_tool()
    @env_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def update_user_booking(
        self,
        client_confirmation_number: str,
        add_speical_bags: list[BaggageType] = [],
    ):
        """
        Update the user booking information based on the client_confirmation_number.
        Will add the special baggage info to the user's booking information.
        """
        new_baggage_info = []
        for speical_bag_type in add_speical_bags:
            new_baggage_info.append(
                BaggageInfo(
                    bagage_type=speical_bag_type,
                    baggage_checkin_type=BaggageCheckinType.CARRY_ON,
                )
            )
        for client_booking_info in self.client_booking_info:
            if client_booking_info.confirmation_number == client_confirmation_number:
                client_booking_info.baggage_info.extend(new_baggage_info)


@dataclass
class FlightCustomerService(FlightCustomerServiceApp):
    __doc__ = FlightCustomerServiceApp.__doc__
    name: str | None = "FlightCustomerService"
