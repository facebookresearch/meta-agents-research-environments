def register_scenarios(registry):
    """
    Register all scenarios in this package with the provided registry.

    Args:
        registry: The ScenarioRegistry instance to register with
    """
    # Simply import the modules containing the scenarios
    # The decorators will handle the registration
    import scenario_flight_customer_service.scenario
