# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""LLM prompt templates and few-shot examples for the judge.

Ported verbatim from ``gaia2.validation.prompts`` and ``gaia2.validation.examples``.
All static text — no external dependencies.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# LLMFunctionTemplates dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMFunctionTemplates:
    system_prompt_template: str = ""
    user_prompt_template: str = ""
    system_prompt_args: dict[str, str] | None = None
    assistant_prompt_template: str | None = None
    examples: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Few-shot examples  (from gaia2.validation.examples)
# ---------------------------------------------------------------------------

USER_MESSAGE_SUBTASK_EXTRACTOR_EXAMPLES = [
    {
        "input": {
            "tool_name": "AgentUserInterface__send_message_to_user",
            "oracle_action_call": textwrap.dedent("""\
                content: Augustín Moreno-Sánchez
                """),
            "task": textwrap.dedent("""\
                Tell me only the name of the youngest person I have emailed this month who is not in any of my Chats conversations but is in at least one Messages thread. If there is a tie, choose the one based on first name in alphabetical order.
                """),
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The tool call indicates that the task is to send a message to the user containing a specific name, which is "Augustín Moreno-Sánchez." This implies that the task involves identifying this individual's name based on specific criteria and communicating that information to the user. The criteria involve the individual's age, email interaction, absence from Chats conversations, and presence in Messages threads. Among the individuals fitting these criteria, the name is selected based on alphabetical order in case of a tie.
                """),
            "subtask": textwrap.dedent("""\
                Communicate to the user the name of the youngest person emailed this month, not in Chats but in Messages, chosen based on alphabetical order if needed.
                """),
        },
    },
    {
        "input": {
            "tool_name": "AgentUserInterface__send_message_to_user",
            "oracle_action_call": textwrap.dedent("""\
                content: 6
                """),
            "task": textwrap.dedent("""\
                Identify the average attendee age from the August Networking Events that took place on the same days that I had scheduled Free Time. Do not include me in the calculation. Round that value to the nearest whole number and tell me how many contacts I have who are within 3 years of the number.
                """),
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The task involves several steps including identifying average attendee age, rounding it to the nearest whole number, and determining the number of contacts within 3 years of that number. The tool call is for "AgentUserInterface__send_message_to_user" with the content "6". This implies that the subtask is related to sending a message to the user. Given the task context, the message to be sent is likely the final result.
                """),
            "subtask": textwrap.dedent("""\
                Tell me how many contacts I have who are within 3 years of the calculated average age.
                """),
        },
    },
]


EVENT_SUBTASK_EXTRACTOR_EXAMPLES = [
    {
        "input": {
            "tool_name": "CalendarApp__add_calendar_event",
            "oracle_action_call": textwrap.dedent("""\
                title: Research Time
                description: Time allocated for research and study.
                """),
            "task": textwrap.dedent("""\
                I am unable to perform any physical activities for the rest of this week as I have a back spasm. Please cancel all yoga events that are scheduled for the rest of this week. Also, I would like to spend the freed time on research, so please add "Research Time" events in the place of the cancelled yoga events. Send a group email to the attendees of yesterday's Research Meeting and ask them what topics they are researching.
                """),
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The tool call is for the Calendar__add_calendar_event tool, with the title "Research Time" and description "Time allocated for research and study." This suggests that the subtask is related to adding a new event to the calendar. Looking at the task, it mentions adding "Research Time" events in place of the cancelled yoga events. This indicates that the subtask is to add these specific events to the calendar, which aligns with the tool call's details.
                """),
            "subtask": textwrap.dedent("""\
                Add "Research Time" event to the calendar in place of cancelled yoga event.
                """),
        },
    },
    {
        "input": {
            "tool_name": "CalendarApp__add_calendar_event",
            "oracle_action_call": textwrap.dedent("""\
                title: Inspection Day
                """),
            "task": textwrap.dedent("""\
                Save all unsaved properties that are listed as 800 sqft and add an inspection day event to my calendar from 9 AM to 4 PM on the next Sunday with no events scheduled on that day.
                """),
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The tool call is for the Calendar__add_calendar_event tool with the title "Inspection Day". This suggests that the subtask we are looking for is related to adding an event to the calendar. The task mentions adding an "inspection day event" to the calendar, which aligns with the title in the tool call. Therefore, we can infer that the subtask is about adding this specific event to the calendar.
                """),
            "subtask": textwrap.dedent("""\
                Add an inspection day event to my calendar.
                """),
        },
    },
]


CAB_CHECKER_EXAMPLES = [
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                start_location: Mittlere Strasse 12, 4058 Basel, Switzerland
                end_location: Rue de Lausanne 45, Geneva, Switzerland
                """),
            "oracle_action_call": textwrap.dedent("""\
                start_location: Mittlere Strasse 12, 4058 Basel, Switzerland
                end_location: Rue de Lausanne 45, Geneva
                """),
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The start_locations match exactly. The end_locations differ slightly — the agent includes "Switzerland" while the oracle does not — but the street, number, and city match. This is a minor formatting variation.
                """),
            "evaluation": "[[True]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                start_location: Kungsgatan 37, 411 19 Göteborg
                end_location: Linnégatan 1, 413 04 Göteborg
                """),
            "oracle_action_call": textwrap.dedent("""\
                start_location: Kungsgatan 37, 411 19 Göteborg
                end_location: Linnégatan 12, 413 04 Göteborg
                """),
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The start_locations match exactly. However, the end_locations differ. The agent's end_location is "Linnégatan 1, 413 04 Göteborg", while the oracle's end_location is "Linnégatan 12, 413 04 Göteborg". Although the street names and postal codes match, the house numbers are different (1 vs 12). Since the task allows for small variations in address formatting, this discrepancy in house numbers indicates the locations do not match exactly as required.
                """),
            "evaluation": "[[False]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                start_location:  Praias Brancas, 4470-558 Vila Nova da Telha
                end_location: Rua do Almada, 147, 4050-036 Porto, Portugal
                """),
            "oracle_action_call": textwrap.dedent("""\
                start_location: Pedras Rubras, 4470-558 Vila Nova da Telha
                end_location: Rua do Almada, 147, 4050-036 Porto, Portugal
                """),
            "user_address": "Avenida da Liberdade 144, 1250-146 Lisboa",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The agent action call and the oracle action call have different start_locations. Although they share the same postal code, indicating they are in the same general area, the specific locations ("Praias Brancas" vs "Pedras Rubras") do not match.
                """),
            "evaluation": "[[False]]",
        },
    },
]


SIGNATURE_CHECKER_EXAMPLES = [
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                subject: UPDATED: Property Viewing and Cab Booking Confirmation
                content: Dear Astrid,
                Thank you for your reply. I have updated the arrangements as per your request:
                Changed the property to view: Herning City Living - The unfurnished 1-bedroom condo located at Torvegade 20, Herning (7400), priced at 800 per month.
                Rescheduled the premium cab to pick you up tomorrow (Wednesday, October 16th) at 10:00 AM from your address (Østerbrogade 27, København Ø) to take you to the unfurnished property. The updated ride details are:
                Ride ID: 9559c4a0c3ba4e0da3e2c6a74bca8c24
                Service Type: Premium
                Pickup time: 10:00 AM, October 16th, 2024
                Estimated fare: $38.96
                As requested, I am sending this confirmation via email and will also send a message in our 1-on-1 Chats conversation.
                Please let me know if you need any further adjustments.
                Best regards,
                Assistant
                """),
            "user_name": "Søren Kjær",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The email content contains a signature with the name "Assistant". According to the task, if the 'Your assistant', 'Assistant' or 'User' is in the email signature, the evaluation should return False. Therefore, the presence of "Assistant" in the signature is sufficient to conclude the evaluation.
                """),
            "evaluation": "[[False]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                subject: Kaveri Apartments Viewing Appointment - Tuesday, Oct 22 at 10:30 AM
                content: Dear Ravindra,

                I hope this email finds you well. I'm writing to confirm our viewing appointment at Kaveri Apartments (Chennai, Tamil Nadu) scheduled for next Tuesday, October 22, 2024, from 10:30 AM to 11:30 AM.

                The property is a 2-bedroom, 1-bathroom apartment with 900 sq ft, unfurnished on the upper floors. It includes amenities like parking and laundry facilities.

                Please let me know if you need any additional information or if there are any changes to the schedule.

                Looking forward to meeting you at the property.

                Best regards,
                Jasmine
                """),
            "user_name": "Rohan Kumar",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The email content contains a signature with the name "Assistant". According to the task, if the 'Your assistant', 'Assistant' or 'User' is in the email signature, the evaluation should return False. Therefore, the presence of "Assistant" in the signature is sufficient to conclude the evaluation.
                """),
            "evaluation": "[[False]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                content: Hi Takashi, I'd be happy to collaborate on a project with you. My free days over the next two weeks are Monday, Tuesday, Thursday, and Sunday. Let me know which day works best for you to meet up and brainstorm some ideas.
                """),
            "user_name": "Ali Hassan",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The email content does not contain a signature. There is no name or any indication of a signature at the end of the email.
                """),
            "evaluation": "[[True]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                subject: Travel Enquiry
                content: Dear Klaasje,

                I hope this email finds you well. I am planning a trip to Melbourne and was wondering if you might still have connections with your old employer that could help me get some cheap deals.

                I've set aside some time this Saturday, October 19, from 5:00 PM to 6:00 PM for a 'Travel Enquiring' session, and I would be keen to have a call with you during this time if you're available.

                Looking forward to hearing from you.

                Best regards,
                Rohini
                """),
            "user_name": "Rohini Deshpande",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The email content contains a signature with the name "Rohini", which matches the first name of the provided username "Rohini Deshpande". Since the names match, the evaluation is True.
                """),
            "evaluation": "[[True]]",
        },
    },
]


EVENT_CHECKER_EXAMPLES = [
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                title: Wine & Crackers Gathering!
                location: 1234 NW 23rd Ave, Portland, OR 97210
                """),
            "oracle_action_call": textwrap.dedent("""\
                title: Wine & Crackers Gathering!
                location: Home
                """),
            "task": "Schedule the 'Wine & Crackers Gathering!' event on the calendar",
            "user_address": "1234 NW 23rd Ave, Portland, OR 97210",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The agent's event title matches the reference event title, which is "Wine & Crackers Gathering!". The location in the reference event is "Home", which is a placeholder for the user's home address. The user's address is provided as "1234 NW 23rd Ave, Portland, OR 97210", which is the same location used by the agent. According to the evaluation rules, the agent can use the user's address if the reference event's location includes a placeholder for the home address. Since the agent's event location matches the user's address and the title matches the reference event title, the agent's event includes all the key semantic information present in the reference event.
                """),
            "evaluation": "[[Success]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                title: Research Meeting
                description: Conduct research on ongoing projects and read relevant literature.
                """),
            "oracle_action_call": textwrap.dedent("""\
                title: Research Time
                description: Time allocated for research purposes.
                """),
            "task": "Add a 'Research Time' calendar event to replace a cancelled yoga event",
            "user_address": "4567 NE 45th St, Seattle, WA 98101",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The agent's event has a title "Research Meeting" which is different from the reference event's title "Research Time". However, according to the evaluation rules, the title can be different unless the task specifically asks for a certain title. In this case, the user's task does ask for a 'Research Time' calendar event, so the title should match. The description of the agent's event, "Conduct research on ongoing projects and read relevant literature", provides more specific details about the research activity compared to the reference event's description, "Time allocated for research purposes". Since the task specifically asks for a 'Research Time' event, the agent's event does not match the reference event due to the title mismatch.
                """),
            "evaluation": "[[Failure]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                title: Dinner with Family
                description: Spend quality time with family.
                location: Tiensestraat 39, 3000 Leuven
                """),
            "oracle_action_call": textwrap.dedent("""\
                title: Dinner with Family
                description: Spend quality time with family.
                location: Place Saint-Lambert, 4000 Liege, Belgium
                """),
            "tool_name": "Calendar__add_calendar_event",
            "task": "Reschedule the Dinner with Family event by changing its location to the destination of the cab ride taken on October 11th",
            "user_address": "Rue de l'Église 12, 1000 Bruxelles, Belgium",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The agent's event has the same title and description as the reference event, which matches the requirements. However, the location in the agent's event is "Tiensestraat 39, 3000 Leuven", whereas the reference event's location is "Place Saint-Lambert, 4000 Liege, Belgium". The user's task specifies that the location should be changed to the destination of the cab ride taken on October 11th, which must be the one in the reference event. Since the location in the agent's event does not match the location in the reference event we conclude that the agent's event does not matches the reference event.
                """),
            "evaluation": "[[Failure]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                title: Yoga Class
                description: 45-minute yoga class with a certified instructor.
                tag: Yoga
                """),
            "oracle_action_call": textwrap.dedent("""\
                title: Yoga Session
                description: yoga class.
                tag: Wellness
                """),
            "tool_name": "Calendar__add_calendar_event",
            "task": "Schedule a yoga class for 45 minutes with a certified instructor",
            "user_address": "67 Main St, Anytown, USA",
        },
        "output": {
            "reasoning": textwrap.dedent("""\
                The agent's event title is "Yoga Class" while the reference event title is "Yoga Session". Although the titles are not identical, they convey the same semantic meaning. The agent's event description is "45-minute yoga class with a certified instructor", which matches the user's task of scheduling a "45-minute yoga class with a certified instructor". The reference event description is simply "yoga class", which is less detailed but still conveys the main activity. The tags are different, with the agent's event tagged as "Yoga" and the reference event tagged as "Wellness". However, since the task does not specify a particular tag, this difference is acceptable according to the evaluation rules. The agent's event includes all the key semantic information present in the reference event and the user's task, which is the main criterion for evaluation.
                """),
            "evaluation": "[[Success]]",
        },
    },
]


USER_MESSAGE_CHECKER_EXAMPLES = [
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                content: You had 1 dinner appointment with Søren Mikkelsen last month.
                """),
            "oracle_action_call": textwrap.dedent("""\
                content: 2
                """),
            "task": "Inform the user of the number of dinner appointments last month with the contact matching the specified name and location criteria",
        },
        "output": {
            "content": textwrap.dedent("""\
                The number of dinner appointments last month is specified as 2.
                """),
            "reasoning": textwrap.dedent("""\
                The reference message indicates that there were 2 dinner appointments last month, which suggests that the number of appointments is the key piece of information. In contrast, the agent's message states that there was 1 dinner appointment with Søren Mikkelsen last month. This discrepancy suggests that the agent's message does not include all the semantic information found in the reference message, specifically the correct number of dinner appointments.
                """),
            "evaluation": "[[Failure]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                content: I have updated the job details for all 7 contacts living in France to 'Teacher'. I'm ready to order 7 guitar pinatas, but I need to know which variant you prefer. Here are the available options:

                1. Medium, Paper Mache, Brown, Candy-filled - $28.64
                2. Medium, Cardboard, Brown, Candy-filled - $25.49
                3. Medium, Paper Mache, Red, Confetti-filled - $30.99

                Please let me know which variant you would like to order.
                """),
            "oracle_action_call": textwrap.dedent("""\
                content: Which version of the Acoustic Guitar Pinata would you like to purchase?
                """),
            "task": "Clarify the version of the Acoustic Guitar Pinata to purchase",
        },
        "output": {
            "content": textwrap.dedent("""\
                The task is to clarify the version of the Acoustic Guitar Pinata to purchase.
                The reference message asks the user to choose a version of the Acoustic Guitar Pinata.
                """),
            "reasoning": textwrap.dedent("""\
                The agent's message  provide options for the user to choose from, which indirectly relates to choosing a version of the guitar pinata. The key information from the reference message is the request to choose a version of the Acoustic Guitar Pinata. The agent's message, while including this choice, specifies the options which are more detailed than the reference message's general inquiry. Despite the additional details, the core request to select a variant (or version) of the pinata is present.
                """),
            "evaluation": "[[Success]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                content: I found two available versions of the Rubie's Adult Star Wars Darth Vader Glovelettes in the color black. One is made of Latex and costs $10.7, and the other is made of Faux Leather and costs $12.99. Which one would you like to buy?
                """),
            "oracle_action_call": textwrap.dedent("""\
                content: Rohan, I found two versions of the Rubie's Adult Star Wars Darth Vader Glovelettes in black. Which one would you like to buy?
                """),
            "task": "Ask the user to choose between two versions of the Rubie's Adult Star Wars Darth Vader Glovelettes in black",
        },
        "output": {
            "content": textwrap.dedent("""\
                The semantic information present in the reference message includes:
                The user's name: Rohan
                The product name: Rubie's Adult Star Wars Darth Vader Glovelettes
                The color of the product: black
                The existence of two versions of the product
                A question asking the user to choose between the two versions
                """),
            "reasoning": textwrap.dedent("""\
                The agent's message includes the product name, color, and the task of choosing between two versions. It provides additional details not present in the reference message, such as the materials and prices of the two versions. The reference message does not mention these specifics but does include the user's name, "Rohan," which the agent's message omits. According to the guidelines, the agent can omit the user's name. The critical aspect is whether the agent's message contains all the semantic information from the reference message, which it does.
                """),
            "evaluation": "[[Success]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                content: Based on my search, I found three people who have used the word 'computer' in their emails. Their email addresses are: kaarina.makinen@gmail.com, kaarlo.nieminen@student.aalto.fi, and jianzhi.wen@shsci.ac.cn. However, I couldn't find any matching Chats conversations for the given user IDs. If you have any further information or clarification about the person you are looking for, I may be able to help you narrow down the search.
                """),
            "oracle_action_call": textwrap.dedent("""\
                content: jianzhi.wen@shsci.ac.cn
                """),
            "task": "Identify and communicate the email address of the fellow computer lover who matches the specified criteria.",
        },
        "output": {
            "content": textwrap.dedent("""\
                The email address of the fellow computer lover who matches the specified criteria is jianzhi.wen@shsci.ac.cn.
                """),
            "reasoning": textwrap.dedent("""\
                The reference message provides a specific email address (jianzhi.wen@shsci.ac.cn) as the answer to identifying the fellow computer lover who matches the criteria. However, the agent's message provides three different email addresses (kaarina.makinen@gmail.com, kaarlo.nieminen@student.aalto.fi, and jianzhi.wen@shsci.ac.cn) without clearly identifying which one is the correct match for the specified criteria. While the correct email address is included in the agent's list, the agent fails to definitively identify it as the answer, instead presenting it as one of three options and asking for further clarification. This does not fulfill the task requirement to "identify and communicate the email address" of the specific person who matches the criteria.
                """),
            "evaluation": "[[Failure]]",
        },
    },
]


EMAIL_CHECKER_EXAMPLES = [
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                subject: Which house will you be?
                content: I've ordered you the Harry Potter Sorting Hat.
                """),
            "oracle_action_call": textwrap.dedent("""\
                subject: Which house will you be?
                content: Dear Kymberley,
                I've ordered the Harry Potter Sorting Hat for you.
                Best,
                Rohan
                """),
            "today_date": "2024-10-15 Tuesday",
        },
        "output": {
            "content": textwrap.dedent("""\
                ### Reference Email Body Content:
                1. The sender has ordered the Harry Potter Sorting Hat.
                2. The Harry Potter Sorting Hat was ordered for the recipient, Kymberley.
                """),
            "reasoning": textwrap.dedent("""\
                The agent's email states that the sender has ordered the Harry Potter Sorting Hat for the recipient, but it does not explicitly mention the recipient's name, "Kymberley". However, the key semantic information in the reference email is that the Harry Potter Sorting Hat was ordered for the recipient. Since the agent's email implies the hat was ordered for the recipient (though without specifying the name), it captures the essential action and intent of the reference email.
                """),
            "evaluation": "[[Success]]",
        },
    },
    {
        "input": {
            "agent_action_call": textwrap.dedent("""\
                subject: Dallas Apartment Recommendation
                content: Hi Matthew, I found an apartment in Dallas that might interest you. It's called Uptown Lofts, located at 789 Uptown Dr, Dallas, TX 75201. It's a furnished 2-bedroom, 2-bathroom apartment on the penthouse floor with amenities like a gym, pool, and laundry. The rent is $2500 per month. Would you recommend this as a good place to live when I move to Dallas?
                """),
            "oracle_action_call": textwrap.dedent("""\
                subject: Dallas Apartment Recommendation
                content: Hi Matthew,
                Are the Uptown Lofts a good place to live when I move?
                Best,
                Aurélien
                """),
            "today_date": "2024-10-15 Tuesday",
        },
        "output": {
            "content": textwrap.dedent("""\
                Reference Email Body Content:
                    - The email inquires about whether the Uptown Lofts are a good place to live when the sender moves to Dallas.
                """),
            "reasoning": textwrap.dedent("""\
                The agent's email includes all the semantic information present in the reference email body, which is the inquiry about the suitability of the Uptown Lofts as a living place. Although the agent's email provides additional details about the apartment, it contains the core question posed in the reference email, thus covering all the semantic information present in the reference email body.
                """),
            "evaluation": "[[Success]]",
        },
    },
]


# ---------------------------------------------------------------------------
# Prompt templates  (from gaia2.validation.prompts)
# ---------------------------------------------------------------------------

IN_CONTEXT_JUDGE_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Evaluator Task
    You are an impartial evaluator responsible for assessing the success of an agent assisting a user within an environment in comparison to an oracle agent.
    In this environment, the user communicates with the agent via a UserAgentInterface, and the agent utilizes tools from various apps to fulfill user requests.
    You will receive two lists of actions (each action is a tool call): one representing actions taken by the agent and another representing actions performed by a skilled oracle agent that perfectly fulfilled the user's request.

    ###Instructions
    First, you will list the differences and similarities between the actions taken by the agent and those performed by the oracle agent.
    Then, based on the evaluation criteria below, you will decide if the agent's actions match the oracle agent's actions within acceptable tolerance limits.

    ### Evaluation Criteria
    The agent's actions should be executed in an order that does not violate the causal relationships between oracle actions provided by with the parent tool call ids.
    The number of calls to each tool should be the same for the agent and the oracle agent actions.
    The agent's action call parameters should be free of significant grammatical or spelling errors and maintain an appropriate tone.
    {{evaluation_criteria}}

    ### Input Format
    The input will be provided in the following format:

    Agent Actions:

    < List of agent actions in the format:
        Tool name: <name of the tool used in the action>
        Tool call time: <time of the action>
        Arguments:
        <tool arguments>
    >

    Oracle Actions:

    < List of oracle actions in the format:
        Tool call id: <id of the oracle tool call>
        Parent tool call ids: <ids of the parent tool calls>
        Tool name: <name of the tool used in the action>
        Tool call time: <time of the action>
        Arguments:
        <tool arguments>
    >

    Task: <user's task>

    Previous task: <previous task solved by the agent>

    User name: <name of the user>

    ### Output Format
    For the evaluation, first list the differences and similarities between the agent and oracle agent actions.
    Then give your reasoning as to why the agent's actions match or critically differ from the oracle agent actions.
    Finally, provide your final evaluation by strictly following this format: "[[success]]" if the agent actions match the oracle agent actions otherwise "[[failure]]".
    Report your evaluation in the following format:

    -Similarities and differences: <List the differences and similarities between the agent and oracle agent actions.>
    -Reasoning: <Detailed explanation of why the agent's actions match or not the oracle agent actions.>
    -Evaluation: <[[success]] if the agent actions match oracle agent actions [[failure]] otherwise.>

    ### Your Evaluation
    For the following input, provide your evaluation following the output format specified above.
    """)


TIME_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""
    All agent actions matching an oracle action with a delay exceeding {{check_time_threshold_seconds}} seconds relative to its parent should be executed within the following time window:
    [oracle action delay - {{pre_event_tolerance_seconds}} seconds, oracle action delay + {{post_event_tolerance_seconds}} seconds]
    """)

PER_TOOL_EVALUATION_CRITERIA = {
    "CalendarApp__add_calendar_event": textwrap.dedent("""\
        4. The title, description and tag of the calendar event CAN BE DIFFERENT to the oracle's calendar event, UNLESS the task, if specified, asks for a specific title, description or tag.
        5. Stylistic and tone differences between the agent and oracle title and descriptions are acceptable.
        """),
    "EmailClientApp__reply_to_email": textwrap.dedent("""\
        4. The content of the agent's email should include all information present in the oracle's email, though stylistic and tone differences are acceptable.
        5. The agent's email can provide additional context or details as long as it does not contradict the oracle's information.
        6. The agent can miss some information in content that is NOT RELEVANT for the task if specified.
        7. IMPORTANT: The agent's email should not include a signature placeholder, such as '[Your Name]', or '[User's Name]', nor placeholders for recipient names.
        8. IMPORTANT: The agent email's signature should not be 'User', 'Assistant' or 'Your assistant'.
        9. Greetings and salutations can differ between the agent and oracle emails, including the option for either party to omit them altogether.
        """),
    "EmailClientApp__send_email": textwrap.dedent("""\
        4. The subject and the content of the agent's email should include all information present in the oracle's email, though stylistic and tone differences are acceptable.
        5. The agent's email can provide additional context or details as long as it does not contradict the oracle's information.
        6.The agent can miss some information in content that are NOT RELEVANT for the task if specified.
        7. IMPORTANT: The email should not include a signature placeholder, such as '[Your Name]', '[My Name]', or '[User's Name]', nor placeholders for recipient names.
        8. IMPORTANT: The agent email's signature should not be 'User', 'Assistant' or 'Your assistant'.
        9. Greetings and salutations can differ between the agent and oracle emails, including the option for either party to omit them altogether.
        """),
    "MessagingApp__send_message": textwrap.dedent("""\
        4. The message sent by the agent can contain more information than oracle's message but should NOT miss information present in the oracle's message.
        5. The stylistic and tone differences between the agent and oracle agent messages are acceptable.
        6. Greetings and salutations can slightly differ between the agent and oracle messages.
        """),
    "CabApp__order_ride": textwrap.dedent("""\
        4. The agent should book a cab to the same `end_location`, with the same `start_location`, allowing for small variations in address formatting as in the oracle action.
        5. If the oracle uses 'Home' as location, the agent can replace it with any specific address.
        """),
    "AgentUserInterface__send_message_to_user": textwrap.dedent("""\
        4. All information conveyed by message from the oracle to the user should also be conveyed by the agent message, including:
        - Names
        - Places
        - Product details
        - Contact information (e.g., phone numbers)
        - Email addresses
        - Conversation history
        - Apartment details (e.g., address, amenities)
        - Event details (e.g., date, time, location)
        - Ride details (e.g., pickup/dropoff locations, times)
        - File system information (e.g., file names, locations)
        - Statistics on items (e.g., number of items in a list). No tolerance is allowed for statistics.
        5. The agent's message can provide more information than the oracle's message.
        6. Messages from the agent can differ slightly in style and tone (e.g., capitalization, spacing, punctuation) from those of the oracle agent.
        """),
    "MessagingApp__create_conversation": textwrap.dedent("""\
        4. The agent can provide extra information in the title.
        5. Title from the agent can differ slightly in style and tone (e.g., capitalization, spacing, punctuation) from those of the oracle agent UNLESS user explicitly asks for a specific title in the task.
        """),
}


# ---------------------------------------------------------------------------
# CONTENT_CHECKER
# ---------------------------------------------------------------------------

CONTENT_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Evaluator Task
    You are an impartial evaluator responsible for comparing the action of an agent assisting a user in comparison to an oracle action.
    You will receive two sets of parameters used to call the same tool: one set used for the action call made by the agent and another used in the action call by a skilled oracle that perfectly fulfills the user's request.
    Your task is to judge if the agent action call matches the oracle action call.

    ### Instructions
    Based on the evaluation criteria below, you will decide if the agent's action matches the oracle's action.
    You MUST strictly follow the evaluation criteria below.

    ### Evaluation rules
    1. ONLY compare the provided parameters and NOTHING ELSE.
    2. NEVER question the oracle action call.
    3. Disregard any differences in the parameters regarding tone, style, formatting, greetings and salutations, and other non-essential details.

    ### Input Format
    The input will be parameters of the action call made by the agent and the oracle as follows and some additional information formatted as follows:

    - Agent Action Call:
    <agent_action_call>

    - Oracle Action Call:
    <oracle_action_call>

    - Tool name: <tool_name>

    - Task: <user's task if specified, else empty>

    - Today's date ('YYYY-MM-DD' format): <today's date>

    - User's address: <user's address>

    ### Output Format
    Report your evaluation in the following format:

    - Reasoning: <Detailed explanation of why the agent's action matches or not the oracle action.>
    - Evaluation: <[[Success]] if the agent action matches the oracle action, [[Failure]] otherwise.>

    ### Your Evaluation
    For the following input, provide your evaluation following the output format specified above.
    """)


CONTENT_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Agent Action Call:
    {{agent_action_call}}

    - Oracle Action Call:
    {{oracle_action_call}}

    - Tool name: {{tool_name}}

    - Task: {{task}}

    - Today's date ('YYYY-MM-DD' format): {{today_date}}

    - User address: {{user_address}}
    """)


CONTENT_CHECKER_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Reasoning: {{reasoning}}
    - Evaluation: {{evaluation}}
    """)


CONTENT_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=CONTENT_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=CONTENT_CHECKER_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=CONTENT_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# SANITY_CHECKER
# ---------------------------------------------------------------------------

SANITY_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### System
    You are a sanity filter to catch reward hacking attempts.

    ### Input
    - Task: <the requested task>
    - Response: <the generated response>

    ### Instructions
    Return [[False]] if the response looks like reward hacking:
    - Contains placeholders like {name}, [INSERT_X]
    - Is garbled, corrupted, or nonsensical
    - Contains technical artifacts when plain text was expected (code blocks, markup, tokens)
    Else return [[True]].

    ### Output
    Reasoning: <brief explanation>
    Evaluation: <[[True]] or [[False]]>
    """)


SANITY_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    Task: {{task}}

    Response: {{agent_action_call}}
    """)

SANITY_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=SANITY_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=SANITY_CHECKER_USER_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# SIGNATURE_CHECKER
# ---------------------------------------------------------------------------

SIGNATURE_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### System
    You are a skilled email signature verification assistant.

    ### Input
    Input will be a pair of email subject and content, formatted as follows:

    - Username: <the username to check against the email signature>

    - Email:
    subject: <subject of the email>
    content: <content of the email>

    ### Task:
    Check if the email content contains a signature.
    If there is no signature or a signature without a name, return [[True]].
    If there is a signature, extract the name from the signature.
    Compare the extracted name with the provided username.
    If the names match (e.g., same first name and or last name), return [[True]].
    If the names do not match, return [[False]].
    IMPORTANT: If the email signature contains 'Your Name', 'Your assistant', 'Assistant' or 'User' return [[False]].

    ### Output:
    Report your evaluation in the following format:

    - Reasoning: <Detailed explanation of your decision.>
    - Evaluation: <[[True]] if no signature or matching, [[False]] otherwise.>

    ### Your Evaluation
    For the following input, provide your evaluation strictly following the output format specified above.
    """)

SIGNATURE_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Username: {{user_name}}

    - Email:
    {{agent_action_call}}
    """)

SIGNATURE_CHECKER_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Reasoning: {{reasoning}}

    - Evaluation: {{evaluation}}
    """)


SIGNATURE_CHECKER_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=SIGNATURE_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=SIGNATURE_CHECKER_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=SIGNATURE_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    examples=SIGNATURE_CHECKER_EXAMPLES,
)


# ---------------------------------------------------------------------------
# CAB_CHECKER
# ---------------------------------------------------------------------------

CAB_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### System
    You are a skilled Address Comparison Assistant.

    ### Input
    - Agent Action Call: start and end locations of the agent's cab booking
    - Oracle Action Call: start and end locations of the expected cab booking
    - User Address: the user's home address

    ### Task
    Verify if the agent booked a cab to the same locations as the oracle.

    ### CRITICAL Rules
    1. **"Home" = User Address**: If one side uses "Home" (or "My place", "My address", "home") and the other uses the User Address (or any part of it), this is a MATCH. This is the most important rule.
    2. **Address formatting tolerance**: Abbreviations, missing/extra zip codes, different component ordering, missing country — all acceptable if they refer to the same place.
    3. **Semantic matching**: Two addresses match if they refer to the same physical location, regardless of how they are written.
    4. Both start_location AND end_location must match.

    ### Output
    - Reasoning: <Detailed explanation. First check if either address is "Home" and should be compared to User Address.>
    - Evaluation: <[[True]] if both locations match, [[False]] otherwise.>

    ### Your Evaluation
    For the following input, provide your evaluation strictly following the output format specified above.

    """)

CAB_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Agent Action Call:
    {{agent_action_call}}

    - Oracle Action Call:
    {{oracle_action_call}}

    User Address: {{user_address}}
    """)

CAB_CHECKER_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Reasoning: {{reasoning}}
    - Evaluation: {{evaluation}}
    """)


CAB_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=CAB_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=CAB_CHECKER_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=CAB_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    examples=CAB_CHECKER_EXAMPLES,
)


# ---------------------------------------------------------------------------
# SUBTASK_EXTRACTOR
# ---------------------------------------------------------------------------

SUBTASK_EXTRACTOR_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### System
    You are a skilled subtask extractor.

    ### Input
    Input will be the name of tool used, the tool call and task, formatted as follows:

    - Tool name: <the name of tool called by the tool call>

    - Tool call: <arguments of the tool call>

    - Task: <task to extract the subtask from>

    ### Task:
    Extract from the task the subtask addressed by the tool call. Follow these rules:
    1. Focus only on the arguments present in the tool call; exclude any parts related to other arguments.
    2. Concentrate solely on the tool specified in the tool call; disregard any parts related to other tools.
    3. Ensure the subtask does not include specific information from the tool call that is absent in the task.

    ### Output:
    Report your answer in the following format:

    - Reasoning:
    <reasoning><Provide a detailed explanation of how the tool call relates to the task.></reasoning>

    - Subtask:
    <subtask><The extracted subtask></subtask>

    ### Your answer
    For the following input, provide the extracted subtask strictly following the output format specified above.
    Do not forget the <subtask> and </subtask> tags.
    """)


SUBTASK_EXTRACTOR_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Tool name: {{tool_name}}

    - Tool call:
    {{oracle_action_call}}

    - Task: {{task}}
    """)


SUBTASK_EXTRACTOR_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Reasoning:
    <reasoning>{{reasoning}}</reasoning>

    - Subtask:
    <subtask>{{subtask}}</subtask>
    """)


SUBTASK_EXTRACTOR_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=SUBTASK_EXTRACTOR_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=SUBTASK_EXTRACTOR_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=SUBTASK_EXTRACTOR_ASSISTANT_PROMPT_TEMPLATE,
)

USER_MESSAGE_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=SUBTASK_EXTRACTOR_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=SUBTASK_EXTRACTOR_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=SUBTASK_EXTRACTOR_ASSISTANT_PROMPT_TEMPLATE,
    examples=USER_MESSAGE_SUBTASK_EXTRACTOR_EXAMPLES,
)

EVENT_SUBTASK_EXTRACTOR_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=SUBTASK_EXTRACTOR_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=SUBTASK_EXTRACTOR_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=SUBTASK_EXTRACTOR_ASSISTANT_PROMPT_TEMPLATE,
    examples=EVENT_SUBTASK_EXTRACTOR_EXAMPLES,
)


# ---------------------------------------------------------------------------
# EMAIL_CHECKER
# ---------------------------------------------------------------------------

EMAIL_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Evaluator Task
    You are an impartial evaluator responsible for comparing an email sent by an agent on behalf of a user to a reference email.
    The reference email contains crucial information.
    Your goal is to determine if the agent's email conveys the same core information as the reference email body.

    ### Evaluation Rules
    1. **Content Focus**: Concentrate solely on the semantic content of the emails. Ignore differences in style, tone, verbosity, and formatting.
    2. **Exclude greetings and sign-offs**: Ignore all differences in greetings (e.g., "Dear" vs "Hi"), salutations, signatures, and sign-offs. The agent may use first name only, full name, or a different closing. Sign-off differences are NEVER a reason to fail.
    3. **Complete Information**: The agent's email must include the core semantic information present in the reference email body. This means the key facts, actions, requests, and data — not peripheral phrases.
    4. **Additional Information**: The agent's email may contain additional information beyond the reference email, but it must not contradict any core information.
    5. **Acceptable Variations** — none of these are reasons to fail:
        - Different wording that conveys the same meaning
        - Extra context, details, politeness, or explanations beyond the reference
        - Absolute dates (e.g., "October 17th") and relative dates (e.g., "tomorrow", "Thursday") are equivalent when referring to the same day. Omitting the year is acceptable.
        - "17:00" and "5 PM" are equivalent
        - First name only when the person is unambiguous
        - Minor courtesies or filler phrases from the reference being absent (e.g., "please make necessary adjustments", "as per our discussion") when the core action/info is conveyed
        - Implicit information counts as present (e.g., "I cancelled and reordered in Navy Blue" implies the product was purchased)
        - Embedded information: if the core facts appear anywhere in the agent's email, they count as present
        - Rephrased requests that ask for the same thing (e.g., "Would you be my chess coach?" ≈ "Could you coach me in chess?")
        - The agent adding more specific details than the reference (e.g., adding a specific date to "this Thursday", or adding a time to a date) is acceptable as long as the added detail is consistent with the reference
        - A detailed summary in place of a brief statement is acceptable
        - Different currency symbols or codes for the same amount (e.g., "$2,000" and "€2,000" and "2,000 EUR") are equivalent when the underlying numeric value matches. The data source does not specify currency, so the agent may reasonably choose any symbol.

    ### What IS a failure
    Only fail the agent if:
    - A key fact is wrong (wrong date, name, number, location)
    - Core information the recipient needs is completely absent and not inferable
    - The agent contradicts what the reference says

    ### Input Format
    - Agent Email: <agent's email>
    - Reference Email: <reference email>
    - Today's Date: <today's date in 'YYYY-MM-DD Weekday' format>

    ### Output Format
    - Reference Email Body Content: <List core semantic information in the reference email body, EXCLUDING greetings and sign-offs>.
    - Reasoning: <Does the agent's email convey the same core information?>
    - Evaluation: <[[Success]] or [[Failure]]>

    ### Your Evaluation
    Based on the input provided, deliver your evaluation following the output format specified above.
    """)

EMAIL_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Agent Email:
    {{agent_action_call}}

    - Reference Email:
    {{oracle_action_call}}

    - Today's date ('YYYY-MM-DD Weekday' format): {{today_date}}
    """)


EMAIL_CHECKER_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Reference Email Content: {{content}}
    - Reasoning: {{reasoning}}
    - Evaluation: {{evaluation}}
    """)


EMAIL_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=EMAIL_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=EMAIL_CHECKER_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=EMAIL_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    examples=EMAIL_CHECKER_EXAMPLES,
)


# ---------------------------------------------------------------------------
# EVENT_CHECKER
# ---------------------------------------------------------------------------

EVENT_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Evaluator Task
    You are an impartial evaluator responsible for comparing a calendar event created by an agent to a reference event.
    The reference event perfectly fulfills the user's request.
    Your task is to determine if the agent's event matches the reference event.

    ### Evaluation Rules
    1. **Reference as Standard**: Treat the reference event as the definitive standard.
    2. **Focus on Content**: Concentrate on the semantic content of the event details. Ignore differences in formatting.
    3. **Start and end times**: Must match the reference when BOTH the reference and the agent event include time information. Different time formats are equivalent ("5 PM" = "17:00"). Dates referring to the same day are equivalent. **If the agent's event does not include start/end time fields, do not treat this as a failure** — the times may have been set correctly but not displayed in the event output.
    4. **Title**: Titles match if they refer to the same activity. These are examples of acceptable variations (not an exhaustive list):
        - "Movie" = "Movie with Kaitlyn Nakamura" = "Movie Night"
        - "Walk" = "Walk with Alessia" = "Afternoon Walk"
        - "Dinner Time" = "Daily Dinner Reminder" = "Dinner"
        - "Call Placeholder" = "Placeholder - Available for Call" = "Available for calls"
        - "Film Club Revival Discussion" = "Tentative: Film Club Revival Discussion"
        - "Networking Prep" = "Networking Prep Call"
        - "One-on-one with Kaia Nakamura" = "Meeting with Kaia"
        - "Call with X" = "Meeting with X" (calls and meetings are interchangeable)
        - "Online Drone Learning Session" = "Drone Learning Session"
        - "KDOT Party Planning Committee Meeting" = "KDOT Party Planning Committee"
        Both adding AND removing attendee names, qualifiers (e.g., "Tentative:", "Online"), topic context, or session numbers is acceptable as long as the core activity is the same. Only fail on title if the agent's title refers to a completely different activity (e.g., "Gym Session" vs "Dinner Party").
    5. **Description, tag**: CAN be different. Empty/None values match anything.
    6. **Location**: CAN be different unless the task explicitly names a specific venue. "Home"/"My place"/"Home Office"/"Office" = user's actual address. Missing postal codes are acceptable — never fail for a missing postal code. Empty locations match anything.
    7. **Attendees**: Extra or missing attendees are acceptable.
    8. **Absent fields**: If any field is absent from the agent's event output (not shown at all), do not fail for that field. Only fail if a field is present in the agent's event but has a WRONG value.

    ### What IS a failure
    Only fail if:
    - Start or end time is explicitly present in the agent's event but wrong
    - The agent created an event for a clearly different activity
    - A core fact is present in the agent's event but contradicts the reference

    ### Input Format
    - Agent Event: <agent's event>
    - Reference Event: <reference event>
    - User's task: <user's task if specified, otherwise empty>
    - User's address: <user's address>

    ### Output Format
    - Reasoning: <Does the agent's event match the reference?>
    - Evaluation: <[[Success]] if match, [[Failure]] otherwise.>

    ### Your Evaluation
    For the following input, provide your evaluation following the output format specified above.

    """)

EVENT_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Agent Event:
    {{agent_action_call}}

    - Reference Event:
    {{oracle_action_call}}

    - User's task: {{task}}

    - User's address: {{user_address}}
    """)


EVENT_CHECKER_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
     - Reasoning: {{reasoning}}
     - Evaluation: {{evaluation}}
    """)


EVENT_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=EVENT_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=EVENT_CHECKER_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=EVENT_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    examples=EVENT_CHECKER_EXAMPLES,
)


# ---------------------------------------------------------------------------
# MESSAGE_CHECKER
# ---------------------------------------------------------------------------

MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Evaluator Task
    You are an impartial evaluator tasked with comparing a message sent by an agent on behalf of a user to a reference message.
    The reference message contains crucial information that the user needs.
    Your objective is to determine if the agent's message conveys the same core information as the reference message.

    ### Evaluation Rules
    1. **Content Focus**: Concentrate solely on the semantic content of the messages. Ignore differences in style, tone, verbosity, and formatting.
    2. **Greetings and sign-offs**: Ignore the greeting line at the start of the message and the sign-off at the end. These are verified separately and do not matter for this evaluation.
    3. **Complete Information**: The agent's message must include the core semantic information present in the reference message body — key facts, actions, requests, and data.
    4. **Additional Information**: The agent's message may contain additional information, but it must not contradict core information.
    5. **Acceptable Variations** — none of these are reasons to fail:
        - Different wording conveying the same meaning
        - Extra context, details, politeness, or explanations
        - Absolute dates and relative dates are equivalent when referring to the same day (e.g., "October 17th" = "Thursday" = "tomorrow"). Omitting the year is acceptable. "17:00" = "5 PM".
        - First name only when the person is unambiguous
        - Minor courtesies or filler phrases absent when core info is conveyed
        - Implicit information counts as present
        - Embedded information: core facts anywhere in the message count
        - Rephrased requests conveying the same meaning
        - Adding more specific details consistent with the reference
        - A detailed summary in place of a brief statement
        - Name in the greeting line being different (e.g., "Hi Kåre" vs "Hi Solveig") — this is verified separately
        - Different currency symbols or codes for the same amount (e.g., "$2,000" and "€2,000" and "2,000 EUR") are equivalent when the underlying numeric value matches. The data source does not specify currency, so the agent may reasonably choose any symbol.
        - Equivalent questions: "how old is he now?" ≈ "how old is he turning?" — both ask about the person's age and are interchangeable
        - "you" vs "you both" when context makes clear who is being addressed
        - Omitting a time range endpoint when the start time is correct (e.g., "at 2 PM" vs "from 2 PM to 4 PM") — the start time is the critical detail

    ### What IS a failure
    Only fail the agent if:
    - A key fact in the message body is wrong (wrong number, location, product, event details)
    - Core information the recipient needs is completely absent and not inferable
    - The agent contradicts what the reference says

    ### Input Format
    - Agent Message: <agent's message>
    - Reference Message: <reference message>
    - Today's Date: <today's date in 'YYYY-MM-DD Weekday' format>

    ### Output Format
    - Reference Message Content: <List core semantic information in the reference message body, EXCLUDING the greeting line and sign-off>.
    - Reasoning: <Does the agent's message convey the same core information?>
    - Evaluation: <[[Success]] or [[Failure]]>

    ### Your Evaluation
    For the input provided, deliver your evaluation following the output format specified above.
    """)

MESSAGE_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Agent Message:
    {{agent_action_call}}

    - Reference Message:
    {{oracle_action_call}}

    - Today's Date: {{today_date}}
    """)


MESSAGE_CHECKER_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Reference Message Content: {{content}}
    - Reasoning: {{reasoning}}
    - Evaluation: {{evaluation}}
    """)


MESSAGE_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=MESSAGE_CHECKER_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=MESSAGE_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# USER_MESSAGE_CHECKER
# ---------------------------------------------------------------------------

USER_MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Evaluator Task
    You are an impartial evaluator.
    Your job is to compare a message sent by an agent to a user with a reference message.
    The reference message is considered the perfect response to a provided user's task.
    Your task: Decide whether the agent's message conveys the same core information as the reference message, in the context of the user's task.

    ### Instructions
    1. **Reference as Standard**: Treat the reference message as the definitive standard. Do not question its content or intent.
    2. **Content Over Form**: Concentrate solely on the semantic information of the messages. Ignore differences in style, tone, verbosity, length, or formatting. Exact wording is NEVER required — only semantic equivalence matters.
    3. **Information Completeness**: The agent's message must include the core semantic information present in the reference message. This includes, but is not limited to:
        - Names, places, product details
        - Contact info (phone numbers, emails)
        - Conversation or event details (dates, times, locations)
        - File or system info (file names, locations)
        - Statistics or counts (e.g., number of items)
    4. **Acceptable Variations**: The following differences are acceptable and should NOT cause a failure:
        - **Names**: The agent can omit the user's name. Using first name only (e.g., "Asa" instead of "Asa Lindstrom") is acceptable when the person is unambiguous from context. Using a person's actual name instead of their role (e.g., "Kare Fjellstad" instead of "your best friend") or vice versa is also acceptable.
        - **Greeting line and sign-off**: Differences in the greeting (e.g., "Hi Søren" vs "Hello" vs no greeting) and sign-off are NEVER a reason to fail. The greeting name is verified separately and does not matter for this evaluation.
        - **Date formats**: Absolute dates (e.g., "October 18th") and relative dates (e.g., "this Friday", "tomorrow") are equivalent if they refer to the same day. Omitting the year is acceptable.
        - **Rephrased questions and statements**: The agent does NOT need to use the same wording as the reference. Any rephrasing that conveys the same meaning is acceptable. For example: "Did you mean a different position?" and "Could you clarify which position?" are equivalent. "Which one would you like to visit?" and "Which one should I book?" are equivalent when context makes them interchangeable. "Which one should I buy?" and "Would you like to purchase X or skip?" both ask the user to choose.
        - **Additional detail**: The agent may include extra information (e.g., listing specific items, providing context, summarizing completed steps) as long as the core information from the reference is present and not contradicted.
        - **Implicit vs explicit**: Information that is clearly implied by the agent's message counts as present. For example, asking "Which variant would you like?" after listing options implies the task was completed.
        - **Terse reference, detailed agent**: If the reference message is very brief (e.g., "All done", "Message sent", "None", or just a number like "5"), the agent may provide a detailed summary instead. As long as the agent's response covers the same intent and does not contradict the reference, this is acceptable. A detailed summary of completed actions is equivalent to a brief confirmation. In particular, if the reference is "None" or empty, the agent may still provide a status update or summary — this is acceptable.
        - **Units, currency, and formatting**: Omitting units (e.g., "2000" vs "2000 square feet"), omitting or changing currency symbols (e.g., "$7" vs "7", "$2,000" vs "€2,000"), or using different number formatting is acceptable when the numeric value is clear from context.
        - **Embedded information**: The agent's message may be longer than the reference and embed the key information within a broader status update or summary. As long as the reference's core information can be found somewhere in the agent's message — even if surrounded by additional context — this counts as present.
        - **Partial action summaries**: When the reference mentions multiple completed actions briefly, the agent does not need to list every single one in the exact same way. If the agent's message covers the key actions and the overall status is the same, minor omissions in the summary are acceptable — especially when the agent provides detail on other aspects.
        - **Equivalent clarification questions**: When the reference asks the user to clarify something (e.g., "Did you mean a different position?"), the agent may ask an equivalent but differently worded clarification (e.g., "Could you provide the contact's name?" or "Could you specify which one?"). Both are valid ways to request the missing information.
        - **Reporting actions taken**: When reporting what messages were sent or actions were taken, the agent does not need to reproduce the exact content. A summary or paraphrase of what was done is acceptable (e.g., "I sent a message to Kofi about the apartment search" is equivalent to quoting the exact message text).
        - **Event/item names**: When referring to events, items, or actions that were created or found, adding context to the name is acceptable (e.g., "Walk" = "Walk with Alessia", "the email to Elara" = "the verification email to Elara").
    5. **What IS a failure**: The agent's message must not omit or contradict any core factual information that would change the user's understanding. Focus on: wrong numbers, wrong names, missing a key piece of information the user needs to act on, or contradicting what the reference states. Stylistic differences, rewordings, and extra detail are never failures on their own.

    ### Input Format
    You will receive:
    - Agent Message: <agent's message>
    - Reference Message: <reference message>
    - Task: <user's task>

    ### Output Format
    Respond as follows:
    - Reference Message Content: <List the core semantic information present in the reference message.>
    - Reasoning: <Explain whether the agent's message conveys the same core information, keeping in mind the acceptable variations above.>
    - Evaluation: <[[Success]] if the core information is present, [[Failure]] if core information is missing or contradicted.>

    ### Your Evaluation
    For the following input, provide your evaluation using the output format above.

    """)

USER_MESSAGE_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Agent Message:
    {{agent_action_call}}

    - Reference Message:
    {{oracle_action_call}}

    - Task: {{task}}
    """)


USER_MESSAGE_CHECKER_ASSISTANT_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Reference Message Content: {{content}}
    - Reasoning: {{reasoning}}
    - Evaluation: {{evaluation}}
    """)

USER_MESSAGE_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=USER_MESSAGE_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=USER_MESSAGE_CHECKER_USER_PROMPT_TEMPLATE,
    assistant_prompt_template=USER_MESSAGE_CHECKER_ASSISTANT_PROMPT_TEMPLATE,
    examples=USER_MESSAGE_CHECKER_EXAMPLES,
)


# ---------------------------------------------------------------------------
# TONE_CHECKER
# ---------------------------------------------------------------------------

TONE_CHECKER_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### System
    You are a skilled style and tone checker assistant tasked with reviewing the text of messages and emails.
    ### Input
    Input will be formatted as follows:
    - Text:
    <email or message>
    ### Task:
    Your goal is to ensure that the text of the messages and emails meets the following criteria:
    1. The text is in human-readable plain English or a piece of information including: phone number, email address, numerical value, date, address, or name.
    2. The tone of the text is appropriate for the context and audience.
    3. The text is free of significant grammatical or spelling errors.
    If any of the above criteria are not met, return [[False]]. Otherwise, return [[True]].
    ### Output:
    Report your evaluation in the specified format:
    Reasoning: <Detailed explanation of your decision.>
    Evaluation: <[[True]] if all criteria are met, [[False]] otherwise.>
    ### Your Evaluation
    For the following input, provide your evaluation strictly following the output format specified above.
    """)

TONE_CHECKER_USER_PROMPT_TEMPLATE = textwrap.dedent("""\
    - Text:
    {{agent_action_call}}
    """)

TONE_CHECKER_PROMPT_TEMPLATES = LLMFunctionTemplates(
    system_prompt_template=TONE_CHECKER_SYSTEM_PROMPT_TEMPLATE,
    user_prompt_template=TONE_CHECKER_USER_PROMPT_TEMPLATE,
)
