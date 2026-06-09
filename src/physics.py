def calculate_reynolds_number(density: float, velocity: float, diameter: float, viscosity: float) -> float:
    """
    Calculates the Reynolds Number for fluid flow in a pipe.
    """
    if viscosity <= 0:
        raise ValueError("Viscosity must be greater than zero.")
        
    return (density * velocity * diameter) / viscosity

def classify_flow_regime(reynolds_number: float) -> str:
    """
    Classifies the flow regime based on the Reynolds number.
    """
    if reynolds_number < 2000:
        return "Laminar"
    elif 2000 <= reynolds_number <= 4000:
        return "Transitional"
    else:
        return "Turbulent"

def calculate_friction_factor(reynolds_number: float) -> float:
    """
    Estimates the Darcy friction factor.
    Uses 64/Re for Laminar and the Blasius correlation for Turbulent (assuming smooth pipes).
    """
    if reynolds_number < 2000:
        return 64.0 / reynolds_number
    else:
        return 0.3164 * (reynolds_number ** -0.25)

def calculate_pressure_drop(friction_factor: float, length: float, diameter: float, density: float, velocity: float) -> float:
    """
    Calculates the expected pressure drop using the Darcy-Weisbach equation.
    """
    if diameter <= 0:
        raise ValueError("Pipe diameter must be greater than zero.")
        
    return friction_factor * (length / diameter) * (density * (velocity ** 2) / 2)