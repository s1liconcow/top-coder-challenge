import pandas as pd
import numpy as np
import json
import joblib
import os
import sys
import argparse
import socket
from feature_engineering import engineer_features

class ReimbursementPredictor:
    def __init__(self, model_dir='models'):
        """
        Initialize the predictor by loading saved model and metadata.
        
        Args:
            model_dir (str): Directory containing the saved model and metadata
        """
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        self.trained_columns = metadata['trained_columns']
        
        # Load model
        model_path = os.path.join(model_dir, 'model.joblib')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise RuntimeError("Model not found. Please ensure the model is trained and saved.")

    def predict(self, trip_data: dict) -> float:
        """
        Predict reimbursement for a single trip.
        
        Args:
            trip_data (dict): Dictionary containing trip information with required fields:
                - trip_duration_days (int)
                - miles_traveled (float)
                - total_receipts_amount (float)
        
        Returns:
            float: Predicted reimbursement amount
        """
        # Convert input to DataFrame with explicit column names
        df = pd.DataFrame([trip_data], columns=['trip_duration_days', 'miles_traveled', 'total_receipts_amount'])
        
        # Engineer features
        processed_trip = engineer_features(df)
        
        # Align columns with training data
        final_trip = processed_trip.reindex(columns=self.trained_columns, fill_value=0)
        
        # Get prediction
        prediction = self.model.predict(final_trip)[0]
        
        return prediction

def start_server(host='localhost', port=5000):
    """
    Start a server that listens for prediction requests.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    
    print(f"Server started on {host}:{port}", file=sys.stderr)
    
    # Initialize predictor once
    predictor = ReimbursementPredictor()
    
    while True:
        try:
            client, addr = server.accept()
            
            # Read the request data
            data = client.recv(1024).decode().strip()
            if not data:
                continue
                
            # Parse space-delimited input
            duration, miles, receipts = map(float, data.split())
            
            # Create trip data dictionary
            trip_data = {
                'trip_duration_days': int(duration),
                'miles_traveled': miles,
                'total_receipts_amount': receipts
            }
            
            # Make prediction
            prediction = predictor.predict(trip_data)
            
            # Send back the prediction
            response = f"{prediction:.2f}"
            client.sendall(response.encode())
            client.close()
            
        except KeyboardInterrupt:
            print("\nShutting down server...", file=sys.stderr)
            break
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            continue

def main():
    parser = argparse.ArgumentParser(description='Predict reimbursement amount for a business trip.')
    parser.add_argument('--server', action='store_true', help='Start the prediction server')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')
    parser.add_argument('duration', type=int, nargs='?', help='Trip duration in days')
    parser.add_argument('miles', type=float, nargs='?', help='Miles traveled')
    parser.add_argument('receipts', type=float, nargs='?', help='Total receipts amount')
    
    args = parser.parse_args()
    
    if args.server:
        start_server(args.host, args.port)
    elif all(v is not None for v in [args.duration, args.miles, args.receipts]):
        predictor = ReimbursementPredictor()
        trip_data = {
            'trip_duration_days': args.duration,
            'miles_traveled': args.miles,
            'total_receipts_amount': args.receipts
        }
        prediction = predictor.predict(trip_data)
        print(f"Predicted reimbursement: ${prediction:.2f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 