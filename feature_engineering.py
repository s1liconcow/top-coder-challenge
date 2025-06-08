import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features based on interview insights and error analysis.
    Shared between training and prediction.
    
    Key insights from interviews:
    - Efficiency (miles per day) is crucial, with sweet spot around 180-220 miles/day
    - 5-day trips often get bonuses
    - Spending patterns vary by trip length
    - Small receipts can be penalized
    - Distance has non-linear effects on mileage reimbursement
    - Combined factors create different calculation paths
    """
    engineered_df = df.copy()
    
    # 1. Enhanced Efficiency Features (from Kevin's analysis)
    engineered_df['miles_per_day'] = engineered_df['miles_traveled'] / engineered_df['trip_duration_days'].replace(0, 1)
    engineered_df['efficiency_bonus_zone'] = ((engineered_df['miles_per_day'] >= 180) & 
                                            (engineered_df['miles_per_day'] <= 220)).astype(int)
    engineered_df['efficiency_penalty_zone'] = ((engineered_df['miles_per_day'] < 100) | 
                                              (engineered_df['miles_per_day'] > 400)).astype(int)
    
    # 2. Distance-based Features (from Lisa and Marcus's insights)
    engineered_df['distance_tier_1'] = (engineered_df['miles_traveled'] <= 100).astype(int)  # Full rate zone
    engineered_df['distance_tier_2'] = ((engineered_df['miles_traveled'] > 100) & 
                                       (engineered_df['miles_traveled'] <= 500)).astype(int)  # Reduced rate
    engineered_df['distance_tier_3'] = (engineered_df['miles_traveled'] > 500).astype(int)  # Further reduced
    engineered_df['miles_per_day_squared'] = engineered_df['miles_per_day'] ** 2  # Non-linear effect
    
    # 3. Duration-based Features (from Jennifer and Lisa's insights)
    engineered_df['is_5_day_trip'] = (engineered_df['trip_duration_days'] == 5).astype(int)  # Sweet spot
    engineered_df['is_4_to_6_day_trip'] = ((engineered_df['trip_duration_days'] >= 4) & 
                                          (engineered_df['trip_duration_days'] <= 6)).astype(int)
    engineered_df['is_extended_trip'] = (engineered_df['trip_duration_days'] >= 8).astype(int)
    engineered_df['is_short_trip'] = (engineered_df['trip_duration_days'] <= 2).astype(int)
    
    # 4. Receipt-based Features (from multiple interviews)
    engineered_df['receipts_per_day'] = engineered_df['total_receipts_amount'] / engineered_df['trip_duration_days'].replace(0, 1)
    
    # Spending tiers based on trip length (from Kevin's analysis)
    engineered_df['optimal_spending_short'] = ((engineered_df['trip_duration_days'] <= 3) & 
                                             (engineered_df['receipts_per_day'] <= 75)).astype(int)
    engineered_df['optimal_spending_medium'] = ((engineered_df['trip_duration_days'] >= 4) & 
                                              (engineered_df['trip_duration_days'] <= 6) & 
                                              (engineered_df['receipts_per_day'] <= 120)).astype(int)
    engineered_df['optimal_spending_long'] = ((engineered_df['trip_duration_days'] >= 7) & 
                                            (engineered_df['receipts_per_day'] <= 90)).astype(int)
    
    # Small receipt penalties (from Dave and Lisa's insights)
    engineered_df['has_small_receipts'] = (engineered_df['total_receipts_amount'] < 50).astype(int)
    engineered_df['receipts_to_miles_ratio'] = engineered_df['total_receipts_amount'] / (engineered_df['miles_traveled'] + 1)
    
    # 5. Combined Features (from Kevin's analysis of interaction effects)
    engineered_df['sweet_spot_combo'] = ((engineered_df['trip_duration_days'] == 5) & 
                                       (engineered_df['miles_per_day'] >= 180) & 
                                       (engineered_df['receipts_per_day'] <= 100)).astype(int)
    
    engineered_df['vacation_penalty'] = ((engineered_df['trip_duration_days'] >= 8) & 
                                       (engineered_df['receipts_per_day'] > 100)).astype(int)
    
    engineered_df['efficiency_bonus'] = ((engineered_df['miles_per_day'] >= 180) & 
                                       (engineered_df['receipts_per_day'] <= 100)).astype(int)
    
    # 6. Trip Complexity Features
    engineered_df['trip_complexity'] = engineered_df['trip_duration_days'] * engineered_df['miles_per_day']
    engineered_df['spending_efficiency'] = engineered_df['total_receipts_amount'] / (engineered_df['miles_traveled'] + 1)
    
    # 7. Receipt Pattern Features
    engineered_df['receipt_intensity'] = engineered_df['total_receipts_amount'] / (engineered_df['trip_duration_days'] + 1)
    engineered_df['is_high_spending'] = (engineered_df['receipts_per_day'] > 200).astype(int)
    engineered_df['is_low_spending'] = (engineered_df['receipts_per_day'] < 50).astype(int)
    
    # 8. Advanced Anomaly/Edge Case Features (based on high error analysis)
    
    # Catches Case 996: 1 day, 1082 miles, $1809.49 receipts
    engineered_df['extreme_miles_per_day_short_trip'] = (
        (engineered_df['miles_per_day'] > 600) & 
        (engineered_df['trip_duration_days'] <= 2)
    ).astype(int)
    engineered_df['single_day_extreme_receipts'] = (
        (engineered_df['trip_duration_days'] == 1) &
        (engineered_df['total_receipts_amount'] > 1000)
    ).astype(int)

    # Catches Case 152 (4 days, 69 miles, $2321.49 receipts) and Case 115 (5 days, 195.73 miles, $1228.49 receipts)
    engineered_df['very_low_efficiency_high_spending'] = (
        (engineered_df['miles_per_day'] < 50) & 
        (engineered_df['receipts_per_day'] > 200)
    ).astype(int)

    # Catches Case 711 (5 days, 516 miles, $1878.49 receipts)
    engineered_df['moderate_efficiency_very_high_spending'] = (
        (engineered_df['miles_per_day'] >= 50) & 
        (engineered_df['miles_per_day'] < 150) & 
        (engineered_df['receipts_per_day'] > 300) # Higher threshold for very high spending
    ).astype(int)
    
    # Addresses Case 684 (8 days, 795 miles, $1645.99 receipts) - supplements existing vacation_penalty
    engineered_df['long_trip_low_miles_high_spending'] = (
        (engineered_df['trip_duration_days'] >= 7) & 
        (engineered_df['miles_per_day'] < 120) & 
        (engineered_df['receipts_per_day'] > 150)
    ).astype(int)

    # General feature for trips with disproportionately high receipts for mileage
    engineered_df['high_receipts_to_mileage_ratio_short_trip'] = (
        (engineered_df['receipts_to_miles_ratio'] > 5) & # e.g. $5 in receipts for every mile
        (engineered_df['trip_duration_days'] <=3) &
        (engineered_df['miles_traveled'] > 10) # Avoid division by near zero if miles are very low
    ).astype(int)

    return engineered_df