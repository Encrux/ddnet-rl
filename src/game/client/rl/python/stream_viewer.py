"""
Stream viewer for remote frame streaming.

Usage:
    python stream_viewer.py [--host HOST] [--port PORT]
"""

import argparse
import struct
import cv2
import numpy as np

try:
    import zmq
except ImportError:
    print("ZeroMQ not installed. Install with: pip install pyzmq")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description='Teeworlds Frame Stream Viewer')
    parser.add_argument('--host', type=str, default='localhost', help='Host to connect to')
    parser.add_argument('--port', type=int, default=5555, help='Port to connect to')
    args = parser.parse_args()
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{args.host}:{args.port}")
    socket.setsockopt(zmq.SUBSCRIBE, b'frame')
    
    print(f"Connected to tcp://{args.host}:{args.port}")
    print("Press 'q' to quit.")
    
    while True:
        try:
            topic, tick_data, frame_data = socket.recv_multipart()
            tick = struct.unpack('<I', tick_data)[0]
            
            # Decode JPEG
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # Resize for display
            display_size = (frame.shape[1] * 4, frame.shape[0] * 4)
            frame_display = cv2.resize(frame, display_size, interpolation=cv2.INTER_NEAREST)
            
            # Add tick info
            cv2.putText(frame_display, f"Tick: {tick}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Teeworlds Stream', frame_display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except zmq.ZMQError:
            break
    
    cv2.destroyAllWindows()
    socket.close()
    context.term()


if __name__ == '__main__':
    main()
