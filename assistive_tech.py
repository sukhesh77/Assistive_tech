import cv2
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr
import time

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')  # Pre-trained YOLO model for object detection

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize speech recognition
recognizer = sr.Recognizer()

# Define known object heights (in meters) for distance estimation
KNOWN_HEIGHTS = {
    "person": 1.7,  # Average human height
    "car": 1.5,     # Average car height
    "chair": 0.9,   # Average chair height
    "cat": 0.25,    # Average cat height
    # Add more objects and their heights here
}

# Camera parameters (adjust based on your camera)
FOCAL_LENGTH = 600  # Focal length in pixels (example value)

# Function to estimate distance to an object
def estimate_distance(object_height_px, object_type):
    if object_type not in KNOWN_HEIGHTS:
        return None  # Skip if object height is not known
    real_height = KNOWN_HEIGHTS[object_type]
    distance = (real_height * FOCAL_LENGTH) / object_height_px
    return distance

# Function to provide contextual feedback
def provide_contextual_feedback(object_type, distance):
    if distance < 1:
        return f"A {object_type} is very close, about {distance:.2f} meters in front of you."
    elif distance < 3:
        return f"A {object_type} is about {distance:.2f} meters away, to your left."
    else:
        return f"A {object_type} is far away, about {distance:.2f} meters to your right."

# Function to listen for voice commands
def listen_for_command():
    try:
        with sr.Microphone() as source:
            print("Listening for command...")
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
            try:
                audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds
                command = recognizer.recognize_google(audio)
                print(f"Command: {command}")
                return command.lower()
            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")
                return None
            except sr.RequestError:
                print("Speech recognition service is unavailable. Please check your internet connection.")
                return None
    except OSError as e:
        print(f"Microphone error: {e}. Please check your microphone settings.")
        return None

# Main function
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Listen for voice command
        command = listen_for_command()
        if command:
            if "what is in front of me" in command or "describe the scene" in command:
                # Capture frame from webcam
                success, frame = cap.read()
                if not success:
                    print("Error: Could not read frame from webcam.")
                    continue

                # Run YOLOv8 inference on the frame
                results = model(frame)

                # Process detection results
                for result in results:
                    if len(result.boxes) > 0:  # Check if objects are detected
                        for box, cls in zip(result.boxes.xywh, result.boxes.cls):
                            object_type = model.names[int(cls)]  # Get object type
                            object_height_px = box[3].item()    # Get bounding box height in pixels

                            # Estimate distance to the object
                            distance = estimate_distance(object_height_px, object_type)
                            if distance is not None:
                                # Provide contextual feedback
                                feedback = provide_contextual_feedback(object_type, distance)
                                print(feedback)  # Print feedback to console
                                engine.say(feedback)  # Speak feedback
                                engine.runAndWait()

                # Display the annotated frame with a larger window
                annotated_frame = results[0].plot()  # Draw bounding boxes on the frame
                cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)  # Resizable window
                cv2.resizeWindow("YOLOv8 Object Detection", 800, 600)  # Set window size
                cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()