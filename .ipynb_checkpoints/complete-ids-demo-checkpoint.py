# Complete IDS Demonstration System
# Simulates CICIDS attacks and demonstrates ML model detection

import threading
import time
import random
import socket
from scapy.all import *
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import requests
import paramiko
from collections import defaultdict

class AttackSimulator:
    """Simulates various network attacks from CICIDS dataset"""
    
    def __init__(self, target_ip="127.0.0.1"):
        self.target_ip = target_ip
        self.running = False
        
    def start_simulation(self):
        """Start all attack simulations in separate threads"""
        self.running = True
        
        # Start different attack threads
        threading.Thread(target=self.simulate_port_scan, daemon=True).start()
        threading.Thread(target=self.simulate_syn_flood, daemon=True).start()
        threading.Thread(target=self.simulate_brute_force, daemon=True).start()
        threading.Thread(target=self.simulate_botnet, daemon=True).start()
        threading.Thread(target=self.simulate_normal_traffic, daemon=True).start()
        
        print("[SIMULATOR] Attack simulation started...")
        
    def stop_simulation(self):
        """Stop all simulations"""
        self.running = False
        print("[SIMULATOR] Attack simulation stopped...")
        
    def simulate_port_scan(self):
        """Simulate port scanning attack"""
        while self.running:
            try:
                ports = [22, 23, 25, 53, 80, 443, 993, 995]
                for port in random.sample(ports, 3):  # Random 3 ports
                    pkt = IP(dst=self.target_ip)/TCP(dport=port, flags="S")
                    send(pkt, verbose=0)
                    time.sleep(0.1)
                time.sleep(random.randint(10, 20))  # Wait before next scan
            except Exception as e:
                pass
                
    def simulate_syn_flood(self):
        """Simulate SYN flood DoS attack"""
        while self.running:
            try:
                for _ in range(random.randint(5, 15)):
                    src_port = random.randint(1024, 65535)
                    pkt = IP(dst=self.target_ip)/TCP(sport=src_port, dport=80, flags="S")
                    send(pkt, verbose=0)
                    time.sleep(0.05)
                time.sleep(random.randint(15, 30))  # Wait before next flood
            except Exception as e:
                pass
                
    def simulate_brute_force(self):
        """Simulate brute force attack (conceptual)"""
        while self.running:
            try:
                # Simulate failed login attempts by creating TCP connections
                for i in range(random.randint(3, 8)):
                    pkt = IP(dst=self.target_ip)/TCP(dport=22, flags="S")  # SSH port
                    send(pkt, verbose=0)
                    time.sleep(0.5)
                time.sleep(random.randint(20, 40))
            except Exception as e:
                pass
                
    def simulate_botnet(self):
        """Simulate botnet C2 communication"""
        while self.running:
            try:
                # Simulate outbound connections to C2 server
                pkt = IP(dst="8.8.8.8")/TCP(dport=8080, flags="S")  # Fake C2
                send(pkt, verbose=0)
                time.sleep(random.randint(30, 60))  # Periodic beaconing
            except Exception as e:
                pass
                
    def simulate_normal_traffic(self):
        """Simulate normal web browsing traffic"""
        while self.running:
            try:
                # Normal HTTP traffic
                pkt = IP(dst=self.target_ip)/TCP(dport=80, flags="A")
                send(pkt, verbose=0)
                # Normal HTTPS traffic
                pkt = IP(dst=self.target_ip)/TCP(dport=443, flags="A")
                send(pkt, verbose=0)
                time.sleep(random.randint(2, 8))
            except Exception as e:
                pass

class FeatureExtractor:
    """Extract features from network packets for ML model"""
    
    def extract_features(self, packet):
        """Extract key features from packet (simplified for demo)"""
        features = {}
        
        if IP in packet:
            features['src_ip'] = packet[IP].src
            features['dst_ip'] = packet[IP].dst
            features['protocol'] = packet[IP].proto
            features['packet_len'] = len(packet)
            
        if TCP in packet:
            features['src_port'] = packet[TCP].sport
            features['dst_port'] = packet[TCP].dport
            features['tcp_flags'] = packet[TCP].flags
            features['tcp_window'] = packet[TCP].window
            
        # Convert to numerical features for ML model
        numerical_features = self.convert_to_numerical(features)
        return numerical_features
        
    def convert_to_numerical(self, features):
        """Convert features to numerical format for ML model"""
        # Simplified feature vector (replace with your actual feature engineering)
        try:
            feature_vector = [
                hash(features.get('src_ip', '')) % 1000,  # Source IP hash
                hash(features.get('dst_ip', '')) % 1000,  # Dest IP hash
                features.get('src_port', 0),
                features.get('dst_port', 0),
                features.get('protocol', 0),
                features.get('packet_len', 0),
                features.get('tcp_flags', 0),
                features.get('tcp_window', 0)
            ]
            return np.array(feature_vector).reshape(1, -1)
        except:
            return np.zeros((1, 8))  # Default feature vector

class IDSDetector:
    """IDS system for real-time attack detection"""
    
    def __init__(self, model_path=None):
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.detection_stats = defaultdict(int)
        self.running = False
        
        # Load trained model if provided
        if model_path:
            try:
                self.model = joblib.load(model_path)
                print(f"[IDS] Model loaded from {model_path}")
            except:
                print("[IDS] Could not load model, using demo classifier")
                self.model = None
        
    def start_monitoring(self):
        """Start packet monitoring and detection"""
        self.running = True
        print("[IDS] Starting network monitoring...")
        
        # Start packet sniffing in separate thread
        threading.Thread(target=self._sniff_packets, daemon=True).start()
        
        # Start statistics display
        threading.Thread(target=self._display_stats, daemon=True).start()
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        print("[IDS] Monitoring stopped")
        
    def _sniff_packets(self):
        """Sniff packets and analyze them"""
        def packet_handler(packet):
            if not self.running:
                return
                
            try:
                # Extract features
                features = self.feature_extractor.extract_features(packet)
                
                # Classify packet
                if self.model:
                    prediction = self.model.predict(features)[0]
                    confidence = max(self.model.predict_proba(features)[0])
                else:
                    # Demo classifier based on simple rules
                    prediction, confidence = self._demo_classifier(packet)
                
                # Log detection
                self._log_detection(packet, prediction, confidence)
                
            except Exception as e:
                pass
                
        # Start sniffing
        sniff(filter="tcp", prn=packet_handler, store=0)
        
    def _demo_classifier(self, packet):
        """Demo classifier when no ML model is loaded"""
        if TCP in packet:
            tcp_flags = packet[TCP].flags
            dst_port = packet[TCP].dport
            
            # Simple rule-based classification for demo
            if tcp_flags == 2:  # SYN flag
                if dst_port in [22, 23, 25, 53, 80, 443]:
                    return "PortScan", 0.85
                elif dst_port == 80:
                    return "DoS", 0.75
            elif dst_port == 8080:
                return "Botnet", 0.90
            elif dst_port == 22 and tcp_flags == 2:
                return "BruteForce", 0.80
                
        return "Normal", 0.95
        
    def _log_detection(self, packet, prediction, confidence):
        """Log detection results"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if prediction != "Normal":
            src_ip = packet[IP].src if IP in packet else "Unknown"
            dst_ip = packet[IP].dst if IP in packet else "Unknown"
            dst_port = packet[TCP].dport if TCP in packet else "Unknown"
            
            print(f"[{timestamp}] ALERT: {prediction} detected!")
            print(f"    Source: {src_ip} -> Destination: {dst_ip}:{dst_port}")
            print(f"    Confidence: {confidence:.2f}")
            print("-" * 50)
            
            self.detection_stats[prediction] += 1
        else:
            self.detection_stats["Normal"] += 1
            
    def _display_stats(self):
        """Display detection statistics periodically"""
        while self.running:
            time.sleep(30)  # Update every 30 seconds
            if self.detection_stats:
                print("\n" + "="*50)
                print("DETECTION STATISTICS (Last 30s)")
                print("="*50)
                for attack_type, count in self.detection_stats.items():
                    print(f"{attack_type}: {count} detections")
                print("="*50 + "\n")
                
                # Reset stats
                self.detection_stats.clear()

class IDSDemo:
    """Main demo controller"""
    
    def __init__(self, target_ip="127.0.0.1", model_path=None):
        self.simulator = AttackSimulator(target_ip)
        self.detector = IDSDetector(model_path)
        
    def run_demo(self, duration=300):  # 5 minutes default
        """Run the complete IDS demonstration"""
        print("="*60)
        print("INTRUSION DETECTION SYSTEM DEMONSTRATION")
        print("="*60)
        print(f"Target IP: {self.simulator.target_ip}")
        print(f"Demo Duration: {duration} seconds")
        print("="*60)
        
        try:
            # Start IDS monitoring
            self.detector.start_monitoring()
            time.sleep(2)
            
            # Start attack simulation
            self.simulator.start_simulation()
            
            # Run demo for specified duration
            print(f"\nDemo running... (Press Ctrl+C to stop early)")
            time.sleep(duration)
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            
        finally:
            # Clean shutdown
            print("\nShutting down demo...")
            self.simulator.stop_simulation()
            self.detector.stop_monitoring()
            print("Demo completed!")

if __name__ == "__main__":
    # Configuration
    TARGET_IP = "127.0.0.1"  # Change to your target IP
    MODEL_PATH = None  # Path to your trained model (optional)
    DEMO_DURATION = 180  # 3 minutes
    
    # Run the demonstration
    demo = IDSDemo(target_ip=TARGET_IP, model_path=MODEL_PATH)
    demo.run_demo(duration=DEMO_DURATION)