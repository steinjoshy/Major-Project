import numpy as np
import pandas as pd
from scipy import stats


class InventoryOptimization:
    """Calculate inventory optimization metrics: Safety Stock and Reorder Point."""
    
    def __init__(self, service_level=0.95):
        """
        Initialize inventory optimization.
        
        Args:
            service_level: Desired service level (0-1), default 0.95 (95%)
        """
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)  # Z-score for service level
    
    def calculate_demand_statistics(self, demand_data):
        """
        Calculate demand statistics needed for inventory optimization.
        
        Args:
            demand_data: Array of historical demand values
        
        Returns:
            Dictionary with demand statistics
        """
        return {
            'mean_demand': np.mean(demand_data),
            'std_demand': np.std(demand_data),
            'min_demand': np.min(demand_data),
            'max_demand': np.max(demand_data),
            'cv': np.std(demand_data) / np.mean(demand_data)  # Coefficient of variation
        }
    
    def calculate_safety_stock(self, demand_std, lead_time=1):
        """
        Calculate Safety Stock.
        
        Safety Stock = Z-score * Standard Deviation of Demand * sqrt(Lead Time)
        
        Args:
            demand_std: Standard deviation of demand
            lead_time: Lead time in days/periods
        
        Returns:
            Safety stock quantity
        """
        safety_stock = self.z_score * demand_std * np.sqrt(lead_time)
        return safety_stock
    
    def calculate_reorder_point(self, avg_demand, lead_time, safety_stock):
        """
        Calculate Reorder Point.
        
        Reorder Point = (Average Demand * Lead Time) + Safety Stock
        
        Args:
            avg_demand: Average demand per period
            lead_time: Lead time in periods
            safety_stock: Calculated safety stock
        
        Returns:
            Reorder point quantity
        """
        reorder_point = (avg_demand * lead_time) + safety_stock
        return reorder_point
    
    def calculate_economic_order_quantity(self, annual_demand, holding_cost, 
                                         ordering_cost):
        """
        Calculate Economic Order Quantity (EOQ).
        
        EOQ = sqrt(2 * D * S / H)
        where D = annual demand, S = ordering cost, H = holding cost per unit
        
        Args:
            annual_demand: Total annual demand
            holding_cost: Cost to hold one unit per year
            ordering_cost: Cost per order
        
        Returns:
            Economic order quantity
        """
        if holding_cost <= 0 or ordering_cost <= 0:
            return annual_demand / 12  # Default: monthly ordering
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return eoq
    
    def generate_inventory_recommendations(self, demand_data, lead_time=1, 
                                         annual_demand=None, holding_cost=None, 
                                         ordering_cost=None):
        """
        Generate comprehensive inventory optimization recommendations.
        
        Args:
            demand_data: Historical demand values
            lead_time: Lead time in periods
            annual_demand: Annual demand (for EOQ calculation)
            holding_cost: Cost to hold one unit per year
            ordering_cost: Cost per order
        
        Returns:
            Dictionary with inventory recommendations
        """
        # Calculate demand statistics
        stats_dict = self.calculate_demand_statistics(demand_data)
        
        # Calculate Safety Stock
        ss = self.calculate_safety_stock(stats_dict['std_demand'], lead_time)
        
        # Calculate Reorder Point
        rp = self.calculate_reorder_point(stats_dict['mean_demand'], 
                                         lead_time, ss)
        
        recommendations = {
            'average_daily_demand': stats_dict['mean_demand'],
            'demand_std_dev': stats_dict['std_demand'],
            'lead_time': lead_time,
            'service_level': self.service_level * 100,
            'z_score': self.z_score,
            'safety_stock': round(ss, 2),
            'reorder_point': round(rp, 2),
            'min_stock_recommended': round(rp, 2),
            'max_stock_recommended': round(rp + ss, 2)
        }
        
        # Add EOQ if costs provided
        if annual_demand and holding_cost and ordering_cost:
            eoq = self.calculate_economic_order_quantity(
                annual_demand, holding_cost, ordering_cost
            )
            recommendations['economic_order_quantity'] = round(eoq, 2)
        
        return recommendations
    
    def forecast_inventory_levels(self, current_stock, forecast_demand, 
                                 reorder_point, lead_time=1):
        """
        Forecast future inventory levels based on demand forecast.
        
        Args:
            current_stock: Current inventory level
            forecast_demand: Forecasted demand values
            reorder_point: Calculated reorder point
            lead_time: Lead time for orders
        
        Returns:
            DataFrame with inventory forecast
        """
        inventory_levels = []
        current_level = current_stock
        orders_placed = []
        
        for i, demand in enumerate(forecast_demand):
            # Check if reorder needed
            order_placed = False
            if current_level <= reorder_point:
                # Place order (arrives after lead_time)
                order_placed = True
                current_level += np.mean(forecast_demand) * 30  # Order 30 days worth
            
            # Update inventory
            current_level -= demand
            
            inventory_levels.append(current_level)
            orders_placed.append(order_placed)
        
        forecast_df = pd.DataFrame({
            'Period': range(len(forecast_demand)),
            'Forecast_Demand': forecast_demand,
            'Inventory_Level': inventory_levels,
            'Order_Placed': orders_placed
        })
        
        return forecast_df
    
    def generate_optimization_report(self, recommendations):
        """Generate readable inventory optimization report."""
        report = f"""
        === INVENTORY OPTIMIZATION REPORT ===
        
        Service Level: {recommendations['service_level']:.1f}%
        Average Daily Demand: {recommendations['average_daily_demand']:.2f} units
        Demand Std Dev: {recommendations['demand_std_dev']:.2f} units
        Lead Time: {recommendations['lead_time']} days
        
        RECOMMENDED INVENTORY LEVELS:
        - Safety Stock: {recommendations['safety_stock']} units
        - Reorder Point: {recommendations['reorder_point']} units
        - Minimum Stock: {recommendations['min_stock_recommended']} units
        - Maximum Stock: {recommendations['max_stock_recommended']} units
        """
        
        if 'economic_order_quantity' in recommendations:
            report += f"\n- Economic Order Quantity (EOQ): {recommendations['economic_order_quantity']} units"
        
        return report


if __name__ == "__main__":
    print("Inventory Optimization module loaded successfully")
