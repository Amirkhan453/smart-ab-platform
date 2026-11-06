"""
Синтетический генератор данных «как ВИТ» (ТЗ v4, раздел 11; дизайн-заметка фазы 0).

    from ab_platform.simulation.vit import NetworkConfig, generate_network, inject

    net = generate_network(NetworkConfig.preset("pessimistic", n_restaurants=1000, seed=42))
    net2, truth = inject(net, units=ids, start="2026-04-13", end="2026-04-26", lift=0.03)

Параметры по умолчанию — допущения порядка величины, а не данные сети.
"""
from .config import GuestParams, KioskParams, NetworkConfig, PanelParams
from .effects import break_data, inject
from .guests import adopt_feature, apply_guest_effect, generate_guests
from .io import load_network, save_network
from .kiosk import apply_kiosk_ui, generate_kiosk_sessions
from .network import Network, generate_network
from .panel import simulate_reference_revenue
from .truth import Truth

__all__ = [
    "GuestParams", "KioskParams", "NetworkConfig", "PanelParams", "Network", "Truth",
    "generate_network", "inject", "break_data", "generate_guests", "apply_guest_effect",
    "adopt_feature", "generate_kiosk_sessions", "apply_kiosk_ui", "save_network", "load_network",
    "simulate_reference_revenue",
]
