import cv2
import numpy as np

def create_transition(cap, start_frame, end_frame, transition_type='fade', duration_frames=30):
    """
    Create a transition between two frames in a video.
    
    Parameters:
    cap: cv2.VideoCapture object
    start_frame: int, starting frame number
    end_frame: int, ending frame number
    transition_type: str, type of transition ('fade', 'wipe_left', 'wipe_right', 'dissolve')
    duration_frames: int, number of frames for the transition
    
    Returns:
    list of frames containing the transition
    """
    # Save original position
    original_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    # Get the two frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret, frame2 = cap.read()
    
    if not ret or frame1 is None or frame2 is None:
        raise ValueError("Could not read frames")
    
    # Convert frames to float32 for better transition quality
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)
    
    transition_frames = []
    
    for i in range(duration_frames):
        progress = i / (duration_frames - 1)
        
        if transition_type == 'fade':
            # Simple fade transition
            frame = cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)
            
        elif transition_type == 'wipe_left':
            # Wipe from left to right
            width = frame1.shape[1]
            cut_point = int(width * progress)
            frame = frame1.copy()
            frame[:, :cut_point] = frame2[:, :cut_point]
            
        elif transition_type == 'wipe_right':
            # Wipe from right to left
            width = frame1.shape[1]
            cut_point = int(width * (1 - progress))
            frame = frame1.copy()
            frame[:, cut_point:] = frame2[:, cut_point:]
            
        elif transition_type == 'dissolve':
            # Dissolve with random pixels
            mask = np.random.random(frame1.shape[:2]) < progress
            mask = np.stack([mask] * 3, axis=2)
            frame = np.where(mask, frame2, frame1)
            
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")
        
        # Convert back to uint8 for display
        frame_uint8 = frame.astype(np.uint8)
        transition_frames.append(frame_uint8)
    
    # Restore original position
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
    
    return transition_frames

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create video transitions between frames')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--start-frame', type=int, default=100,
                        help='Starting frame number (default: 100)')
    parser.add_argument('--end-frame', type=int, default=200,
                        help='Ending frame number (default: 200)')
    parser.add_argument('--transition-type', type=str, default='fade',
                        choices=['fade', 'wipe_left', 'wipe_right', 'dissolve'],
                        help='Type of transition effect (default: fade)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Number of frames for the transition (default: 30)')
    
    args = parser.parse_args()
    
    # Open the video file
    cap = cv2.VideoCapture(args.video_path)
    
    try:
        # Create transition using command line arguments
        transition_frames = create_transition(cap, args.start_frame, args.end_frame,
                                           transition_type=args.transition_type,
                                           duration_frames=args.duration)
        
        # Create video writer for saving the transition
        first_frame = transition_frames[0]
        out = cv2.VideoWriter('transition.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            30,
                            (first_frame.shape[1], first_frame.shape[0]))
        
        # Display and save the transition frames
        for frame in transition_frames:
            # Display frame
            cv2.imshow('Transition', frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):  # Exit on 'q' press
                break
                
            # Save frame
            out.write(frame)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        # Clean up resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
