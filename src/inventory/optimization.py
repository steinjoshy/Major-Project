import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


class InventoryOptimization:
    """
    Calculate inventory optimization parameters:
      - Safety Stock
      - Reorder Point (ROP)
      - Economic Order Quantity (EOQ, optional)
      - Projected inventory levels

    Formulas:
        Safety Stock  = Z × σ_d × √(Lead Time)
        Reorder Point = (μ_d × Lead Time) + Safety Stock

    where Z is the service-level Z-score, σ_d is the standard deviation of
    daily demand, and μ_d is the mean daily demand.
    """

    def __init__(self, service_level=0.95):
        """
        Args:
            service_level: Target service level probability (0–1). Default 0.95.
        """
        if not 0 < service_level < 1:
            raise ValueError("service_level must be between 0 and 1 exclusive.")
        self.service_level = service_level
        self.z_score = float(scipy_stats.norm.ppf(service_level))

    # ------------------------------------------------------------------
    # Demand Statistics
    # ------------------------------------------------------------------

    def calculate_demand_statistics(self, demand_data):
        """
        Calculate descriptive statistics from historical demand.

        Returns:
            Dict with mean, std, min, max, coefficient of variation
        """
        arr = np.asarray(demand_data, dtype=float)
        arr = arr[np.isfinite(arr) & (arr >= 0)]
        mean_d = float(np.mean(arr))
        std_d = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        return {
            'mean_demand': mean_d,
            'std_demand': std_d,
            'min_demand': float(np.min(arr)),
            'max_demand': float(np.max(arr)),
            'cv': std_d / mean_d if mean_d > 0 else 0.0,
            'n_periods': int(len(arr)),
        }

    # ------------------------------------------------------------------
    # Core Calculations
    # ------------------------------------------------------------------

    def calculate_safety_stock(self, demand_std, lead_time=1):
        """
        Safety Stock = Z-score × σ_demand × √(Lead Time)

        Args:
            demand_std:  Standard deviation of daily demand
            lead_time:   Lead time in days

        Returns:
            Safety stock quantity (rounded up)
        """
        ss = self.z_score * demand_std * np.sqrt(lead_time)
        return max(0.0, float(ss))

    def calculate_reorder_point(self, avg_demand, lead_time, safety_stock):
        """
        Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock

        Args:
            avg_demand:    Mean daily demand
            lead_time:     Lead time in days
            safety_stock:  Calculated safety stock

        Returns:
            Reorder point quantity
        """
        return float(avg_demand * lead_time + safety_stock)

    def calculate_economic_order_quantity(self, annual_demand, holding_cost,
                                          ordering_cost):
        """
        EOQ = √(2 × D × S / H)

        Args:
            annual_demand:  Total annual demand (units)
            holding_cost:   Cost to hold 1 unit for 1 year
            ordering_cost:  Fixed cost per order

        Returns:
            Economic order quantity
        """
        if holding_cost <= 0 or ordering_cost <= 0 or annual_demand <= 0:
            return None
        return float(np.sqrt(2 * annual_demand * ordering_cost / holding_cost))

    # ------------------------------------------------------------------
    # Forecast-based Safety Stock
    # ------------------------------------------------------------------

    def calculate_forecast_based_safety_stock(self, forecast_array, lead_time=1):
        """
        Safety Stock derived from forecast variability rather than historical std.

        Uses the standard deviation of the forecast values as a proxy for
        demand uncertainty over the forecast horizon.

        Args:
            forecast_array: Array of forecasted demand values
            lead_time:      Lead time in days

        Returns:
            Safety stock quantity
        """
        fc = np.asarray(forecast_array, dtype=float)
        fc_std = float(np.std(fc, ddof=1)) if len(fc) > 1 else 0.0
        return self.calculate_safety_stock(fc_std, lead_time)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def generate_inventory_recommendations(self, demand_data, lead_time=7,
                                           forecast_data=None,
                                           annual_demand=None,
                                           holding_cost=None,
                                           ordering_cost=None):
        """
        Generate comprehensive inventory recommendations.

        Args:
            demand_data:    Historical demand array
            lead_time:      Lead time in days
            forecast_data:  Forecast array (used for forecast-based SS if provided)
            annual_demand:  For EOQ calculation
            holding_cost:   For EOQ calculation
            ordering_cost:  For EOQ calculation

        Returns:
            Dict with all inventory parameters
        """
        ds = self.calculate_demand_statistics(demand_data)

        # Prefer forecast-based safety stock if forecast is available
        if forecast_data is not None and len(forecast_data) > 1:
            ss = self.calculate_forecast_based_safety_stock(forecast_data, lead_time)
            ss_basis = 'forecast variability'
        else:
            ss = self.calculate_safety_stock(ds['std_demand'], lead_time)
            ss_basis = 'historical demand variability'

        rop = self.calculate_reorder_point(ds['mean_demand'], lead_time, ss)

        recommendations = {
            'average_daily_demand': round(ds['mean_demand'], 2),
            'demand_std_dev': round(ds['std_demand'], 2),
            'lead_time_days': lead_time,
            'service_level_pct': round(self.service_level * 100, 1),
            'z_score': round(self.z_score, 4),
            'safety_stock': round(ss, 1),
            'reorder_point': round(rop, 1),
            'safety_stock_basis': ss_basis,
            'demand_during_lead_time': round(ds['mean_demand'] * lead_time, 1),
        }

        # EOQ (optional)
        if all(v is not None and v > 0
               for v in [annual_demand, holding_cost, ordering_cost]):
            eoq = self.calculate_economic_order_quantity(
                annual_demand, holding_cost, ordering_cost
            )
            if eoq:
                recommendations['economic_order_quantity'] = round(eoq, 1)

        return recommendations

    # ------------------------------------------------------------------
    # Inventory Projection
    # ------------------------------------------------------------------

    def forecast_inventory_levels(self, current_stock, forecast_demand,
                                  reorder_point, lead_time=7,
                                  safety_stock=None):
        """
        Simulate projected inventory levels over the forecast horizon.

        Args:
            current_stock:    Starting inventory level (units)
            forecast_demand:  Forecasted daily demand array
            reorder_point:    When to trigger a replenishment order
            lead_time:        Days until a placed order arrives
            safety_stock:     Used to size replenishment order (default: 30-day avg demand)

        Returns:
            DataFrame: Period, Forecast_Demand, Inventory_Level, Order_Placed
        """
        forecast_demand = np.asarray(forecast_demand, dtype=float)
        avg_demand = float(np.mean(forecast_demand)) if len(forecast_demand) > 0 else 1.0
        order_qty = max(avg_demand * 30, reorder_point * 2)  # sensible order quantity

        inventory_levels = []
        orders_placed = []
        level = float(current_stock)
        pending_orders = {}  # {arrival_day: qty}

        for i, demand in enumerate(forecast_demand):
            # Receive any pending orders
            if i in pending_orders:
                level += pending_orders[i]

            # Check if reorder needed
            order_placed = False
            if level <= reorder_point:
                arrival_day = i + lead_time
                pending_orders[arrival_day] = pending_orders.get(arrival_day, 0) + order_qty
                order_placed = True

            # Consume demand (floor at 0 — no negative stock in simulation)
            level = max(0.0, level - demand)

            inventory_levels.append(round(level, 1))
            orders_placed.append(order_placed)

        return pd.DataFrame({
            'Period': range(len(forecast_demand)),
            'Forecast_Demand': np.round(forecast_demand, 1),
            'Inventory_Level': inventory_levels,
            'Order_Placed': orders_placed,
        })

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_optimization_report(self, recommendations):
        """Generate a human-readable inventory optimization report."""
        r = recommendations
        lines = [
            "=" * 50,
            " INVENTORY OPTIMIZATION REPORT",
            "=" * 50,
            "",
            f"Service Level Target : {r['service_level_pct']:.1f}%",
            f"Z-Score (normal dist): {r['z_score']:.4f}",
            f"Average Daily Demand : {r['average_daily_demand']:.2f} units",
            f"Demand Std Deviation : {r['demand_std_dev']:.2f} units",
            f"Lead Time            : {r['lead_time_days']} days",
            "",
            "FORMULAS USED",
            "-" * 40,
            "Safety Stock  = Z × σ_demand × √(Lead Time)",
            "             = {z:.4f} × {std:.2f} × √{lt}".format(
                z=r['z_score'], std=r['demand_std_dev'], lt=r['lead_time_days']
            ),
            f"             = {r['safety_stock']:.1f} units  [{r.get('safety_stock_basis','')}]",
            "",
            "Reorder Point = (Avg Demand × Lead Time) + Safety Stock",
            "             = ({avg:.2f} × {lt}) + {ss:.1f}".format(
                avg=r['average_daily_demand'],
                lt=r['lead_time_days'],
                ss=r['safety_stock']
            ),
            f"             = {r['reorder_point']:.1f} units",
            "",
            "RECOMMENDATIONS",
            "-" * 40,
            f"Safety Stock  : {r['safety_stock']:.1f} units",
            f"Reorder Point : {r['reorder_point']:.1f} units",
        ]

        if r.get('economic_order_quantity') is not None:
            lines += [
                "",
                "EOQ = √(2 × D × S / H)",
                f"Economic Order Quantity : {r['economic_order_quantity']:.1f} units",
            ]

        lines += ["", "=" * 50]
        return "\n".join(lines)


if __name__ == "__main__":
    print("Inventory Optimization module loaded successfully")
