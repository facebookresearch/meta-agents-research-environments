import textwrap
import uuid
from dataclasses import dataclass, field
from enum import Enum
from os import system
from re import A
from typing import Any

from are.simulation.agents.user_proxy import UserProxyLLM

from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.apps.app import App
from are.simulation.apps.contacts import Contact, ContactsApp, Gender, Status
from are.simulation.apps.flight_customer_service import (
    BaggageCheckinType,
    BaggageInfo,
    BaggageType,
    BookingInformation,
    Email,
    FlightClient,
    FlightClientApp,
    FlightCustomerService,
    FlightCustomerServiceApp,
    FlightInformation,
    LoyaltyTier,
    RegularBaggagePolicy,
    SpecialBaggagePolicy,
    TicketType,
    UserMemberInformation,
)
from are.simulation.environment import EnvironmentConfig
from are.simulation.scenarios.scenario import Scenario, ScenarioValidationResult
from are.simulation.scenarios.utils.registry import register_scenario
from are.simulation.tool_utils import app_tool, data_tool, OperationType
from are.simulation.types import event_registered, EventRegisterer
from are.simulation.utils.llm_utils import build_llm

from fsspec import spec


FLIGHT_CUSTOMER_SERVICE_PROMPT = textwrap.dedent(
    """\
        From now on role play as a customer service agent interacting with a customer. You are Sarah Chen,
        a Unity Air Customer Service Agent. Your core function is to provide accurate, efficient,
        and empathetic assistance to  passengers, resolve inquiries, and ensure
        a positive customer experience by leveraging comprehensive flight and policy information.

        You are a seasoned customer service agent with a calm demeanor and a deep
        understanding of airline policies and operations. You pride yourself on your ability
        to quickly access information and provide clear, concise answers, especially when dealing
        with potentially stressful customer situations. You are trained to handle a wide range of
        inquiries, from simple flight status checks to complex baggage and special assistance requests.

        # Tools available to you:
         - get_client_membership_info: Get the membership information of the client based on their loyalty_member_id.
         - get_client_flight_booking_info: Get the flight booking information of the client based on their confirmation_number.
         - get_client_baggage_policy_info: Get the baggage policy information of the client based on their loyalty_member_id and confirmation_number.
         - get_special_baggage_policy_info: Get the special baggage policy information of the client based on their bag_type.
         - update_user_booking: Update the booking information of the user based on their special bags.

        With those tools, you should have general knowledge about Global Air flights and services, including:
            - Client membership information
            - Client flight booking information
            - Client baggage policy information
            - Client special baggage policy information

        # Process:
        Please conduct a conversation with the customer to resolve their issue with following step in strict order:
        1. Greet the customer and ask them to provide their loyalty_member_id and confirmation_number.
        2. Use the tools to get the client membership information, flight booking information and get confirmed with the customer.
        3. After the confirmation, ask the customer what needs to be done.
        4. In case customer ask for baggage info, you should use the tools to get the client baggage policy information, and
            if customer's baggage is special, such as sports, medical and so on, you should use the tools to get the
            client special baggage policy information.

        Your communication style is:
        - Empathetic & Reassuring: Understands customer concerns, especially regarding health and travel.
        - Clear & Concise: Provides direct answers without jargon.
        - Proactive: Offers additional relevant information (e.g., recommending documentation for medical bags) even if not explicitly asked.
        - Problem-Solver: Aims to resolve the customer's issue efficiently and effectively on the first contact.
        - Professional: Maintains a polite and helpful tone.

        Your goals are:
        - To accurately answer all customer questions.
        - To ensure the customer feels confident and prepared for their flight.
        - To minimize customer anxiety, especially concerning critical items like medical supplies.
        - To uphold Unity Air's commitment to customer service and safety.
    """
)

CLIENT_PROMPT = textwrap.dedent(
    """\
        From now on role play as a client who is traveling with Unity Air. You are Alex Sharma,
        a frequent flyer with Unity Air.

        Your want to understand baggage allowance, especially for a crucial medical bag, and
        ensure a smooth travel experience despite an underlying health condition.
        Your primary concern revolves around their medical condition: Type 1 Diabetes.
        This necessitates carrying a medical bag with insulin, syringes, a glucose monitor, and other essential supplies.
        They are acutely aware that these items are delicate and vital, making carry-on the preferred and safest option.

        Specific Baggage Questions/Concerns:
        - Standard Baggage Allowance: Alex needs to know the exact allowance for checked bags (weight and dimensions) and carry-on bags (dimensions) for his flight.
        - Medical Bag Allowance: This is the most critical point. Alex needs explicit confirmation that
            their medical bag will not count towards their standard carry-on limit and can be brought
            on board in addition to their personal item and standard carry-on.

         # Tools available to you:
         - get_membership_info: Get the membership information for Alex
         - get_booking_info: Get the booking information for Alex
         - get_medical_info: Check if Alex need to bring a medical bag
         - get_baggage_info: Get baggage information for Alex, Alex want to
            know the exact allowance for them.

        With those tools, Alex should have general knowledge his flight information, including:
            - If Alex is a member of Unity Air
            - Alex's flight details:
            - If Alex need to bring a medical bag
            - Alex's baggage

        # Process:
        Please conduct a conversation with the customer service agent to
        obtain the baggage allowance information, please expect it to be a multple turn
        conversation and each turn we focus on one problem.
        1. Say "Hello, I'm Alex Sharma, a frequent flyer with Unity Air"
        2. Work with the agent to confirm the membership and flight information for Alex.
        3. After the confirmation, check if Alex need to bring a medical bag.
        4. Ask the agent for the overall baggage allowance information.
        5. In case Alex needs to bring a medical bag, ask the agent for the
         medical bag allowance information.


        Make sure you follow these instructions:
        - You have the power to command the agent.
        - IF agent doesn't ask you any question, JUST SAY "Read my previous messages.".
        - When you are done, say "Thank you. I have everything I need. I'm ready to go."
        """
)


@register_scenario("scenario_flight_customer_service_with_medical_bag")
@dataclass
class ScenarioFlightCustomerServiceWithMedicalBag(Scenario):
    """
    Two number generators Apps talking to each other.
    """

    start_time: float | None = 0
    duration: float | None = 2
    nb_turns: int | None = 10
    system_prompt: str | None = field(default=CLIENT_PROMPT)
    additional_system_prompt: str | None = field(default=FLIGHT_CUSTOMER_SERVICE_PROMPT)

    def init_and_populate_apps(self, *args, **kwargs) -> None:
        """Initialize apps and populate with sample data"""

        # Create Agent User Interface with the flight app
        agui = AgentUserInterface()

        email_instance_1 = Email(
            email_address="alex.sharma@gmail.com",
            from_address="customer_service@unityair.com",
            title="Your Unity Air Membership",
            content="""
            Thank you for becoming a Unity Air member! your loyalty_member_id is UA_member_123, and your loyalty_tier is GOLD
            """,
        )
        email_instance_2 = Email(
            email_address="alex.sharma@gmail.com",
            from_address="customer_service@unityair.com",
            title="Your Unity Air Flight Booking",
            content="""
                Thanks for choosing Unity Airline, your flight is UA-2209, and your confirmation number is UA_confirm_123
            """,
        )
        flight_information_1 = FlightInformation(
            flight_number="UA123",
            origin="SFO",
            destination="LAX",
        )
        flight_information_2 = FlightInformation(
            flight_number="UA456",
            origin="SJC",
            destination="NWC",
        )

        booking_instance_1 = BookingInformation(
            confirmation_number="UA_confirm_123",
            passenger_name="Alex Sharma",
            flight_number="UA123",
            loyalty_tier=LoyaltyTier.GOLD.value,
            ticket_type=TicketType.BUSINESS.value,
            baggage_info=[],
        )
        booking_instance_2 = BookingInformation(
            confirmation_number="UA_confirm_456",
            passenger_name="Bob Wang",
            flight_number="UA456",
            loyalty_tier=LoyaltyTier.BRONZE.value,
            ticket_type=TicketType.FIRST_CLASS.value,
            baggage_info=[],
        )

        baggage_info_1 = BaggageInfo(
            bagage_type=BaggageType.NORMAL_BAG.value,
            baggage_checkin_type=BaggageCheckinType.CHECKED_LUGGAGE.value,
        )

        user_membership_instance_1 = UserMemberInformation(
            name="Alex Sharma",
            email=email_instance_1.email_address,
            loyalty_member_id="UA_member_123",
            loyalty_tier=LoyaltyTier.GOLD.value,
        )
        user_membership_instance_2 = UserMemberInformation(
            name="Bob Wang",
            email="bob.wang@gmail.com",
            loyalty_member_id="UA_member_456",
            loyalty_tier=LoyaltyTier.NORMAL.value,
        )
        regular_baggage_policy_1 = RegularBaggagePolicy(
            ticket_type=TicketType.BUSINESS.value,
            loyalty_tier=LoyaltyTier.GOLD.value,
            free_checked_bag_number=1,
        )
        regular_baggage_policy_2 = RegularBaggagePolicy(
            ticket_type=TicketType.ECONOMY.value,
            loyalty_tier=LoyaltyTier.GOLD.value,
            free_checked_bag_number=0,
        )
        regular_baggage_policy_3 = RegularBaggagePolicy(
            ticket_type=TicketType.ECONOMY.value,
            loyalty_tier=LoyaltyTier.NORMAL.value,
            free_checked_bag_number=0,
        )
        special_baggage_policy_1 = SpecialBaggagePolicy(
            bagage_type=BaggageType.MEDICAL_BAG.value,
            allow_to_carry_on=True,
            allow_to_check_in=True,
            require_additional_fee=False,
        )
        special_baggage_policy_2 = SpecialBaggagePolicy(
            bagage_type=BaggageType.SPORTS_EQUIPMENT.value,
            allow_to_carry_on=False,
            allow_to_check_in=True,
            require_additional_fee=True,
        )

        # Create Flight Client App and populate with data
        client_app = FlightClientApp()
        # Populate with Alex Sharma's flight data
        client_app.name = "Alex Sharma"
        client_app.emails = [email_instance_1, email_instance_2]
        client_app.baggage_info = [BaggageType.NORMAL_BAG]

        # Create Flight Customer Service App and populate with data
        customer_service_app = FlightCustomerServiceApp()
        # Populate with Alex Sharma's flight data
        customer_service_app.name = "Sarah Chen"
        customer_service_app.client_booking_info = [
            booking_instance_1,
            booking_instance_2,
        ]
        customer_service_app.client_membership_info = [
            user_membership_instance_1,
            user_membership_instance_2,
        ]
        customer_service_app.regular_bag_policies = [
            regular_baggage_policy_1,
            regular_baggage_policy_2,
            regular_baggage_policy_3,
        ]
        customer_service_app.special_bag_policies = [
            special_baggage_policy_1,
            special_baggage_policy_2,
        ]

        # Store apps for scenario use
        self.apps = [agui, client_app, customer_service_app]

    def build_events_flow(self) -> None:
        """Define the sequence of events that will occur during the scenario"""

        agui = self.get_typed_app(AgentUserInterface)
        client_app = self.get_typed_app(FlightClientApp)
        customer_service_app = self.get_typed_app(FlightCustomerServiceApp)

        with EventRegisterer.capture_mode():
            # User event: User requests task creation
            event1 = agui.send_message_to_agent(
                content="Please play the role as Alex Sharma",
            ).depends_on(None, delay_seconds=1)
            # Oracle event 1: Client app say hi to the customer service agent
            oracle1 = (
                client_app.say_hi_to_customer_service()
                .oracle()
                .depends_on(event1, delay_seconds=1)
            )

            # Oracle event 2: Customer service agent have greeting message
            oracle2 = (
                customer_service_app.get_greeting_message_to_client()
                .oracle()
                .depends_on(oracle1, delay_seconds=1)
            )

        self.events = [event1, oracle1, oracle2]

    def validate(self, env) -> ScenarioValidationResult:
        """
        Validate that the scenario completed successfully.

        Check that the agent properly interacted with our custom app.
        """
        try:
            customer_service_app = env.get_app("FlightCustomerServiceApp")
            # check if customer service has update Alex's booking information
            # with medical bag
            for booking in customer_service_app.client_booking_info:
                if booking.confirmation_number == "UA_confirm_123":
                    for bag_info in booking.baggage_info:
                        if bag_info.bagage_type == BaggageType.MEDICAL_BAG:
                            return ScenarioValidationResult(success=True)
            return ScenarioValidationResult(success=False)

        except Exception as e:
            return ScenarioValidationResult(success=False, exception=e)


if __name__ == "__main__":
    from are.simulation.scenarios.utils.cli_utils import run_and_validate

    # Run the scenario in oracle mode and validate the agent actions
    run_and_validate(ScenarioFlightCustomerServiceWithMedicalBag())
