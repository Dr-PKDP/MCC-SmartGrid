"""
energy_model.py
===============
Marginal energy cost model for MCC device participation (Equation 4) and
comparison against a conventional dedicated-edge-node baseline.

MCC energy figures are sourced from:
    Patterson, D., Gilbert, J.M., Gruteser, M., Robles, E., Sekar, K.,
    Wei, Y., and Zhu, T. (2024). Energy and Emissions of Machine Learning
    on Smartphones vs. the Cloud. Communications of the ACM, 67(2), 87–95.

Conventional edge figures represent a 100-node IoT sensing deployment
operating at continuous 5 W per node, consistent with ruggedised
distribution-edge gateways and smart meter concentrators.
"""

from dataclasses import dataclass


@dataclass
class EnergyConfig:
    """
    Parameters for the marginal energy cost calculation.

    Attributes
    ----------
    e_session_j : float
        Energy per MCC FL/sensing session on a smartphone (Joules).
        From Patterson et al. (2024), Table 3: lightweight ~10K-parameter
        model; breakdown: wakelock 0.5 J, Wi-Fi 0.27 J, CPU 2.2 J = 2.97 J.
    sessions_per_day : int
        Sessions per device per day. Conservative estimate: 12 sessions
        (one per 2 hours during active-use periods), drawn from typical
        participant availability windows.
    edge_nodes : int
        Number of dedicated edge nodes in the conventional baseline.
    edge_node_power_w : float
        Continuous power draw of one conventional edge node (Watts).
        Typical ruggedised IoT edge concentrator or smart meter gateway.
    """
    e_session_j: float = 2.97        # Patterson et al. (2024), Table 3
    sessions_per_day: int = 12
    edge_nodes: int = 100
    edge_node_power_w: float = 5.0   # W, continuous


def mcc_daily_energy_wh(n_devices: int, config: EnergyConfig) -> float:
    """
    Total daily MCC operational energy across all participating devices (Wh).

    This corresponds to the aggregate ΔEᵢ summed over all participants
    across all sessions in one day. For sessions scheduled during
    maintenance mode (device fully charged, plugged in), the per-session
    ΔEᵢ is ≈1.2 J above the charger baseline; the full 2.97 J figure is
    used here as an upper bound (conservative).

    Parameters
    ----------
    n_devices : int
        Number of participating devices.
    config : EnergyConfig

    Returns
    -------
    float : Daily energy in Wh.
    """
    total_j = n_devices * config.sessions_per_day * config.e_session_j
    return total_j / 3600.0   # J → Wh


def edge_daily_energy_wh(config: EnergyConfig) -> float:
    """
    Daily energy consumed by the conventional dedicated-edge baseline (Wh).

    Parameters
    ----------
    config : EnergyConfig

    Returns
    -------
    float : Daily energy in Wh.
    """
    return config.edge_nodes * config.edge_node_power_w * 24.0   # W × h


def energy_ratio(n_devices: int, config: EnergyConfig) -> float:
    """
    Ratio of conventional edge daily energy to MCC daily energy.

    A ratio of R means the conventional baseline consumes R times more
    energy per day than the MCC configuration with n_devices participants.

    Parameters
    ----------
    n_devices : int
    config : EnergyConfig

    Returns
    -------
    float
    """
    mcc = mcc_daily_energy_wh(n_devices, config)
    edge = edge_daily_energy_wh(config)
    return edge / mcc if mcc > 0 else float("inf")
