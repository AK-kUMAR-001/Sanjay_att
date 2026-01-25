from flask import Flask, render_template, Response, request, redirect, url_for, flash
import cv2
import face_recognition
import numpy as np
from face_recognizer import FaceRecognizer
from database import mark_attendance, init_db, get_attendance_records, generate_session_report
import os
import time
from datetime import datetime
from face_encoder import generate_encodings
from twilio.rest import Client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import base64
import io

app = Flask(__name__)
app.secret_key = "secret_key_here"

# Initialize DB
init_db()

# Initialize Recognizer
recognizer = FaceRecognizer()

# Global state for UI hold logic: {student_id: {'until': timestamp, 'name': name}}
display_state = {}

# Constants
HOLD_DURATION = 4  # Seconds to hold the green frame
COOLDOWN_DURATION = 5 # Seconds before re-detecting same person (starts after hold)

# --- Configuration ---
# Twilio WhatsApp (Replace with your SID and Token from Twilio Console)
TWILIO_SID =  "" 
TWILIO_AUTH_TOKEN = ""
TWILIO_FROM = "whatsapp:+14155238886"
TWILIO_TO = "whatsapp:+919791005567" # Staff number

# Email Configuration (Gmail)
# NOTE: For Gmail, use an App Password if 2FA is enabled, or allow "Less Secure Apps"
EMAIL_SENDER = "akshayprabhu19012005@gmail.com"
EMAIL_PASSWORD = "qixo ixhb txqf qvtq"  # App Password
EMAIL_RECEIVER = "akshayprabhu19012005@gmail.com" # Sending to self by default

# Global Camera Management
video_capture = None
is_registering = False
is_attendance_active = False
session_start_time = None
current_session_name = "Session"
marked_students = set()

def send_email_report(file_path, summary):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = f"Attendance Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        body = f"Please find the attached attendance report for the recent session.\n\nSummary:\n{summary}"
        msg.attach(MIMEText(body, 'plain'))
        
        # Attachment
        if file_path and os.path.exists(file_path):
            attachment = open(file_path, "rb")
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= " + os.path.basename(file_path))
            msg.attach(part)
            attachment.close()
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, text)
        server.quit()
        return True, "Email sent successfully."
    except Exception as e:
        print(f"Email Error: {e}")
        return False, str(e)

def get_camera():
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        # PRIORITIZE DSHOW (DirectShow) on Windows for stability
        # MSMF (default) often causes Heap Corruption/Crashes
        print("Opening camera with DSHOW...")
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # If DSHOW fails, try default (MSMF)
        if not video_capture.isOpened():
             print("DSHOW failed. Trying default backend...")
             video_capture = cv2.VideoCapture(0)
             
        # If index 0 fails, try index 1 with DSHOW
        if not video_capture.isOpened():
            print("Camera index 0 failed. Trying index 1 (DSHOW)...")
            video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            
    return video_capture

def release_camera():
    global video_capture
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
        video_capture = None

def generate_frames():
    global is_registering, is_attendance_active
    
    while True:
        if is_registering:
            time.sleep(0.5)
            continue
            
        if not is_attendance_active:
            # Yield a black placeholder frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Attendance Stopped", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            cv2.putText(frame, "Click Start to Resume", (170, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Release camera resource while inactive
            release_camera()
            time.sleep(0.5)
            continue
            
        camera = get_camera()
        if camera is None or not camera.isOpened():
            print("Waiting for camera...")
            time.sleep(1)
            continue

        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera. Resetting...")
            release_camera()
            time.sleep(0.5)
            continue
            
        current_time = time.time()
        
        # Check if we are in a "HOLD" state for any student
        active_hold = None
        for sid, state in list(display_state.items()):
            if current_time < state['until']:
                active_hold = state
                break
            else:
                # Hold expired
                del display_state[sid]

        if active_hold:
            # === HOLD STATE ===
            # Use 0.5 scale for better long-range detection (more pixels = better detection)
            scale_factor = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # If faces found, assume the largest one is our guy
            if face_locations:
                 # Just use the first face for simplicity in hold mode
                top, right, bottom, left = face_locations[0]
                
                # Scale up
                inv_scale = int(1/scale_factor)
                top *= inv_scale
                right *= inv_scale
                bottom *= inv_scale
                left *= inv_scale
                
                color = (0, 255, 0) # Green
                name = active_hold['name']
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, f"{name} - Mark Success", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                
                # Add "Verified" text on screen center
                cv2.putText(frame, "VERIFIED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # === NORMAL RECOGNITION STATE ===
            # Use 0.5 scale for better long-range detection
            scale_factor = 0.5
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            face_names = []
            
            for face_encoding in face_encodings:
                name, student_id = recognizer.recognize_face(face_encoding)
                
                if name != "Unknown" and student_id:
                    display_name = f"{name} ({student_id})"
                    
                    # Mark attendance
                    if student_id not in marked_students:
                        mark_attendance(student_id, name)
                        marked_students.add(student_id)
                        
                        # Trigger HOLD logic
                        display_state[student_id] = {
                            'until': current_time + HOLD_DURATION,
                            'name': name
                        }
                    else:
                        pass
                        
                else:
                    display_name = "Unknown"
                    
                face_names.append(display_name)
                
            # Draw results
            inv_scale = int(1/scale_factor)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up
                top *= inv_scale
                right *= inv_scale
                bottom *= inv_scale
                left *= inv_scale
                
                color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html', is_active=is_attendance_active, session_name=current_session_name)

@app.route('/start_attendance', methods=['GET', 'POST'])
def start_attendance():
    global is_attendance_active, session_start_time, current_session_name, marked_students
    
    if request.method == 'POST':
        current_session_name = request.form.get('session_name', 'Session')
    
    is_attendance_active = True
    session_start_time = datetime.now()
    marked_students.clear() # Reset for new session
    
    flash(f"{current_session_name} Started at {session_start_time.strftime('%H:%M:%S')}")
    return redirect(url_for('index'))

@app.route('/stop_attendance')
def stop_attendance():
    global is_attendance_active, session_start_time, current_session_name
    is_attendance_active = False
    
    msg = f"{current_session_name} Stopped."
    
    # Generate Report if session was active
    if session_start_time:
        # Pass marked_students to filter only unique attendees for this session
        # Or generate_session_report handles it. Let's inspect generate_session_report.
        # Actually, if marked_students prevented duplicates in DB, we are good.
        # But marked_students only prevents re-marking in THIS runtime.
        # If generate_session_report queries DB by time, it might pick up multiple if app restarted?
        # But we want ONE entry per person per session.
        
        file_path, summary, attendees = generate_session_report(session_start_time, current_session_name)
        if file_path:
            msg += f" {summary}"
            
            # Auto-send Email with Excel Attachment
            email_success, email_status = send_email_report(file_path, summary)
            
            if email_success:
                msg += " [Email Sent]"
            else:
                print(f"Email Failed: {email_status}") # Log full error to console
                msg += " [Email Failed]" # Show simple message to user

            # Auto-send via Twilio (Text Summary)
            try:
                client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
                
                # Construct message
                whatsapp_msg = f"🛑 {current_session_name} Report\n"
                whatsapp_msg += f"📅 {datetime.now().strftime('%Y-%m-%d')}\n"
                whatsapp_msg += f"⏰ {session_start_time.strftime('%H:%M')} - {datetime.now().strftime('%H:%M')}\n\n"
                
                if attendees:
                    whatsapp_msg += "✅ Present:\n"
                    for student in attendees:
                        # student = (name, id, time)
                        whatsapp_msg += f"• {student[0]} ({student[1]}) - {student[2]}\n"
                else:
                    whatsapp_msg += "No students detected.\n"
                
                whatsapp_msg += "\n(Excel file saved locally)"

                message = client.messages.create(
                    from_=TWILIO_FROM,
                    body=whatsapp_msg,
                    to=TWILIO_TO
                )
                msg += f" (WhatsApp sent: {message.sid})"
            except Exception as e:
                print(f"Twilio Auto-Send Error: {e}")
                msg += " (WhatsApp failed - Check console)"
        else:
            msg += " No records found in this session."
            
    session_start_time = None # Reset
    flash(msg)
    return redirect(url_for('index'))

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        global is_registering
        student_id = request.form['student_id']
        name = request.form['name']
        
        folder_name = f"{student_id}_{name}"
        # Use absolute path for dataset to avoid CWD issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(base_dir, "dataset")
        folder_path = os.path.join(dataset_dir, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        # Capture images using global camera
        is_registering = True
        time.sleep(1) # Wait for generator to pause
        
        cap = get_camera()
        count = 0
        while count < 20:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            cv2.imwrite(os.path.join(folder_path, f"{count}.jpg"), frame)
            count += 1
            time.sleep(0.2)
        
        is_registering = False
        
        flash(f"Registered {name} successfully! Captured {count} images.")
        return redirect(url_for('index'))
        
    return render_template('register.html')

@app.route('/send_report')
def send_report():
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        
        # Get today's attendance summary
        records = get_attendance_records()
        today_str = time.strftime("%Y-%m-%d")
        today_records = [r for r in records if r[3] == today_str]
        
        msg_body = f"Attendance Report for {today_str}:\n"
        if not today_records:
            msg_body += "No attendance marked today."
        else:
            for r in today_records:
                msg_body += f"- {r[2]} ({r[1]}) at {r[4]}\n"
        
        message = client.messages.create(
            from_=TWILIO_FROM,
            body=msg_body,
            to=TWILIO_TO
        )
        flash(f"Report sent via WhatsApp! SID: {message.sid}")
    except Exception as e:
        flash("Failed to send report. Check console for details.")
        print(f"Twilio Error: {e}")
        
    return redirect(url_for('attendance'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        generate_encodings()
        # Reload recognizer
        global recognizer
        recognizer = FaceRecognizer()
        flash("Model trained successfully!")
        return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/attendance')
def attendance():
    records = get_attendance_records()
    return render_template('attendance.html', records=records)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
